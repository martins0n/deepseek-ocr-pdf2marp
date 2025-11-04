#!/usr/bin/env python3
"""
Complete PDF to Marp Pipeline with Debug Output

This script handles the full pipeline:
1. Convert PDF to images
2. Run OCR on each slide
3. Extract plots using bounding boxes
4. Create Marp presentation
5. Generate debug visualizations

Uses the same transform logic as single_slide_converter.py for accurate bounding boxes.
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
PROMPT = "<image>\n<|grounding|>Convert the slide to markdown."
DEFAULT_BASE = 1024
LATEX_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
LATEX_INLINE = re.compile(r"\\\((.+?)\\\)")
REGION_PATTERN = re.compile(
    r"<\|ref\|>([^<]+)<\|/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|/det\|>(.*?)(?=<\|ref\||$)",
    re.DOTALL,
)

COLORS = {
    "title": (255, 0, 0),
    "sub_title": (255, 128, 0),
    "text": (0, 255, 0),
    "equation": (0, 255, 255),
    "image": (255, 0, 255),
    "image_caption": (128, 0, 255),
    "table": (255, 255, 0),
}


def solve_linear(src_vals, dst_vals, fallback_scale):
    """Fit a linear transform from source to destination coordinates."""
    if len(src_vals) < 2:
        return fallback_scale, 0.0
    A = np.vstack([src_vals, np.ones(len(src_vals))]).T
    try:
        slope, intercept = np.linalg.lstsq(A, dst_vals, rcond=None)[0]
    except np.linalg.LinAlgError:
        return fallback_scale, 0.0
    if not np.isfinite(slope) or slope <= 0 or slope > fallback_scale * 2:
        slope, intercept = fallback_scale, 0.0
    if not np.isfinite(intercept):
        intercept = 0.0
    return float(slope), float(intercept)


def compute_transform(gray, regions, base_size=DEFAULT_BASE, threshold=240):
    """Compute coordinate transform from OCR space to image space."""
    height, width = gray.shape
    default_scale_x = width / base_size
    default_scale_y = height / base_size

    xs_src, xs_dst = [], []
    ys_src, ys_dst = [], []

    # Filter regions for transform computation to avoid outliers
    # Skip boxes that are too large or at suspicious coordinates
    filtered_regions = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Skip boxes that are too large (> 75% of base_size)
        # These are often full-slide images or incorrectly detected regions
        if box_width > base_size * 0.75 or box_height > base_size * 0.75:
            continue
            
        # Skip boxes with invalid or suspicious coordinates
        # Boxes at (0,0) or spanning nearly the full coordinate space
        if (x1 <= 5 and y1 <= 5) or (x2 >= base_size - 5 and y2 >= base_size - 5):
            continue
            
        filtered_regions.append(region)

    # If we filtered out too many regions, fall back to using all regions
    if len(filtered_regions) < 2:
        filtered_regions = regions

    for region in filtered_regions:
        x1, y1, x2, y2 = region["bbox"]
        ax1 = max(0, int(round(x1 * default_scale_x)) - 10)
        ax2 = min(width, int(round(x2 * default_scale_x)) + 10)
        ay1 = max(0, int(round(y1 * default_scale_y)) - 10)
        ay2 = min(height, int(round(y2 * default_scale_y)) + 10)
        if ax2 <= ax1 or ay2 <= ay1:
            continue
        region_arr = gray[ay1:ay2, ax1:ax2]
        if region_arr.size == 0:
            continue
        mask = region_arr < threshold
        if not mask.any():
            continue
        cols = np.where(mask.any(axis=0))[0]
        rows = np.where(mask.any(axis=1))[0]
        if cols.size:
            xs_src.extend([x1, x2])
            xs_dst.extend([ax1 + int(cols[0]), ax1 + int(cols[-1])])
        if rows.size:
            ys_src.extend([y1, y2])
            ys_dst.extend([ay1 + int(rows[0]), ay1 + int(rows[-1])])

    scale_x, offset_x = solve_linear(xs_src, xs_dst, default_scale_x)
    scale_y, offset_y = solve_linear(ys_src, ys_dst, default_scale_y)
    return scale_x, offset_x, scale_y, offset_y


def apply_transform(bbox, transform, img_width, img_height):
    """Apply coordinate transform to a bounding box."""
    x1, y1, x2, y2 = bbox
    scale_x, offset_x, scale_y, offset_y = transform
    nx1 = int(round(scale_x * x1 + offset_x))
    nx2 = int(round(scale_x * x2 + offset_x))
    ny1 = int(round(scale_y * y1 + offset_y))
    ny2 = int(round(scale_y * y2 + offset_y))
    nx1, nx2 = sorted((nx1, nx2))
    ny1, ny2 = sorted((ny1, ny2))
    nx1 = max(0, min(img_width, nx1))
    nx2 = max(0, min(img_width, nx2))
    ny1 = max(0, min(img_height, ny1))
    ny2 = max(0, min(img_height, ny2))
    return nx1, ny1, nx2, ny2


def refine_bbox(gray, bbox, margin=6, threshold=240):
    """Refine bounding box by finding actual content edges."""
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, x1), min(gray.shape[1], x2)
    y1, y2 = max(0, y1), min(gray.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return bbox
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return bbox
    mask = region < threshold
    if not mask.any():
        return bbox
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    ny1 = y1 + int(rows[0])
    ny2 = y1 + int(rows[-1]) + 1
    nx1 = x1 + int(cols[0])
    nx2 = x1 + int(cols[-1]) + 1
    nx1 = max(0, nx1 - margin)
    ny1 = max(0, ny1 - margin)
    nx2 = min(gray.shape[1], nx2 + margin)
    ny2 = min(gray.shape[0], ny2 + margin)
    return nx1, ny1, nx2, ny2


def convert_pdf_to_images(pdf_path, output_dir, dpi=200):
    """Convert PDF to PNG images using pdftoppm."""
    print(f"Converting PDF to images (DPI: {dpi})...")
    os.makedirs(output_dir, exist_ok=True)

    output_prefix = os.path.join(output_dir, "slide")

    cmd = [
        "pdftoppm",
        "-png",
        "-r", str(dpi),
        pdf_path,
        output_prefix
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error converting PDF: {result.stderr}")
        return None

    slide_files = sorted(Path(output_dir).glob("slide-*.png"))
    renamed_files = []

    for idx, slide_file in enumerate(slide_files, 1):
        new_name = os.path.join(output_dir, f"slide_{idx:03d}.png")
        os.rename(slide_file, new_name)
        renamed_files.append(new_name)

    print(f"✓ Converted {len(renamed_files)} pages to images")
    return renamed_files


def run_ocr_on_slide(image_path, model, tokenizer):
    """Run OCR on a single slide and return result."""
    print(f"  Processing: {os.path.basename(image_path)}")

    captured_output = io.StringIO()
    start_time = time.time()

    try:
        with contextlib.redirect_stdout(captured_output):
            result = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=image_path,
                output_path=os.path.dirname(image_path),
                base_size=DEFAULT_BASE,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False
            )

        output_text = captured_output.getvalue()
        elapsed = time.time() - start_time
        print(f"    ✓ Completed in {elapsed:.1f}s")

        text = result if isinstance(result, str) and result.strip() else output_text
        return text.strip()

    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def process_all_slides_ocr(slide_files, output_dir):
    """Run OCR on all slides and combine results."""
    print("\nRunning OCR on all slides...")

    print("Loading DeepSeek-OCR model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device
    ).eval()
    print(f"✓ Model loaded on {device}\n")

    all_results = []

    for idx, slide_file in enumerate(slide_files, 1):
        ocr_result = run_ocr_on_slide(slide_file, model, tokenizer)

        if ocr_result:
            all_results.append({
                'slide_num': idx,
                'content': ocr_result
            })

    os.makedirs(output_dir, exist_ok=True)

    combined_md = "# Lecture Slides\n\nGenerated with DeepSeek-OCR\n\n---\n\n"

    for result in all_results:
        combined_md += f"# Slide {result['slide_num']}/{len(slide_files)}\n\n---\n\n"
        combined_md += result['content']
        combined_md += "\n\n---\n\n"

    md_file = os.path.join(output_dir, "lecture_all_slides.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(combined_md)

    print(f"\n✓ OCR completed for {len(all_results)} slides")
    print(f"✓ Combined output: {md_file}")

    return md_file


def parse_ocr_regions(ocr_file):
    """Parse OCR output to extract regions with bounding boxes."""
    with open(ocr_file, 'r', encoding='utf-8') as f:
        content = f.read()

    slide_pattern = r'# Slide (\d+)/\d+\n+---\n+(.*?)(?=\n# Slide |\Z)'
    slides = re.findall(slide_pattern, content, re.DOTALL)

    slides_data = {}

    for slide_num_str, slide_content in slides:
        slide_num = int(slide_num_str)

        # Remove debug output
        slide_content = re.sub(r'={20,}.*?={20,}', '', slide_content, flags=re.DOTALL)

        regions = []

        for match in REGION_PATTERN.finditer(slide_content):
            region_type = match.group(1)
            bbox_str = match.group(2)
            region_content = match.group(3).strip()

            try:
                coords = [int(x.strip()) for x in bbox_str.split(',')]
                if len(coords) == 4:
                    regions.append({
                        'type': region_type,
                        'bbox': tuple(coords),
                        'content': region_content
                    })
            except ValueError:
                continue

        slides_data[slide_num] = {
            'regions': regions,
            'full_content': slide_content
        }

    return slides_data


def extract_plots(slides_dir, output_dir, slides_data, padding_pct=0.05, min_size=150):
    """Extract plots using bounding boxes with proper transform and refinement."""
    print("\nExtracting plots from slides...")
    os.makedirs(output_dir, exist_ok=True)

    extracted = {}
    stats = {'total': 0, 'extracted': 0, 'skipped': 0}

    for slide_num, data in sorted(slides_data.items()):
        image_regions = [r for r in data['regions'] if r['type'] == 'image']

        if not image_regions:
            continue

        slide_file = os.path.join(slides_dir, f'slide_{slide_num:03d}.png')

        if not os.path.exists(slide_file):
            continue

        img = Image.open(slide_file).convert('RGB')
        img_width, img_height = img.size
        gray = np.array(img.convert('L'))
        
        # Compute transform from OCR coordinates to actual image coordinates
        transform = compute_transform(gray, data['regions'], base_size=DEFAULT_BASE)

        print(
            f"  Slide {slide_num}: {len(image_regions)} image(s), "
            f"scale=({transform[0]:.3f}, {transform[2]:.3f})"
        )

        extracted[slide_num] = []

        for idx, region in enumerate(image_regions, 1):
            stats['total'] += 1
            
            # Apply transform and refine bbox
            mapped = apply_transform(region['bbox'], transform, img_width, img_height)
            refined = refine_bbox(gray, mapped)

            width = refined[2] - refined[0]
            height = refined[3] - refined[1]

            if width < min_size or height < min_size:
                stats['skipped'] += 1
                continue

            # Add padding relative to refined box
            pad_x = int(width * padding_pct)
            pad_y = int(height * padding_pct)

            x1_padded = max(0, refined[0] - pad_x)
            y1_padded = max(0, refined[1] - pad_y)
            x2_padded = min(img_width, refined[2] + pad_x)
            y2_padded = min(img_height, refined[3] + pad_y)

            if x2_padded <= x1_padded or y2_padded <= y1_padded:
                stats['skipped'] += 1
                print(f"    Skipped image {idx}: invalid bbox")
                continue

            # Crop and save
            cropped = img.crop((x1_padded, y1_padded, x2_padded, y2_padded))

            output_file = f'slide_{slide_num:03d}_img_{idx:02d}.png'
            output_path = os.path.join(output_dir, output_file)
            cropped.save(output_path, optimize=True, quality=95)

            extracted[slide_num].append(output_file)
            stats['extracted'] += 1

            print(f"    ✓ {output_file} ({x2_padded-x1_padded}x{y2_padded-y1_padded})")

        img.close()

    print(f"\n✓ Extracted {stats['extracted']}/{stats['total']} plots")
    return extracted, stats


def create_debug_visualization(slides_dir, debug_dir, slides_data):
    """Create debug slides with bounding boxes using proper transform."""
    print("\nCreating debug visualizations...")
    os.makedirs(debug_dir, exist_ok=True)

    for slide_num, data in sorted(slides_data.items()):
        slide_file = os.path.join(slides_dir, f'slide_{slide_num:03d}.png')

        if not os.path.exists(slide_file):
            continue

        img = Image.open(slide_file).convert('RGB')
        gray = np.array(img.convert('L'))
        width, height = img.size
        
        # Compute transform
        transform = compute_transform(gray, data['regions'], base_size=DEFAULT_BASE)

        draw = ImageDraw.Draw(img)
        default_scale = max(width / DEFAULT_BASE, height / DEFAULT_BASE)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                max(12, int(24 * default_scale))
            )
        except Exception:
            font = ImageFont.load_default()

        line_width = max(1, int(4 * default_scale))

        # Draw bounding boxes
        for region in data['regions']:
            # Skip drawing boxes that are clearly problematic
            x1, y1, x2, y2 = region['bbox']
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Skip extremely large boxes that likely indicate OCR errors
            if box_width > DEFAULT_BASE * 0.9 or box_height > DEFAULT_BASE * 0.9:
                continue
            
            mapped = apply_transform(region['bbox'], transform, width, height)
            refined = refine_bbox(gray, mapped)

            # Additional validation: skip if refined box is still too large or invalid
            refined_width = refined[2] - refined[0]
            refined_height = refined[3] - refined[1]
            if refined_width < 5 or refined_height < 5:
                continue
            if refined_width > width * 0.95 or refined_height > height * 0.95:
                continue

            color = COLORS.get(region['type'], (255, 255, 255))

            for offset in range(line_width):
                draw.rectangle(
                    [
                        refined[0] + offset,
                        refined[1] + offset,
                        refined[2] - offset,
                        refined[3] - offset
                    ],
                    outline=color
                )

            label = region['type']
            label_x = refined[0] + line_width
            label_y = refined[1] + line_width
            
            draw.text((label_x, label_y), label, fill=color, font=font)

        # Save
        output_file = os.path.join(debug_dir, f'slide_{slide_num:03d}_debug.png')
        img.save(output_file, quality=95)
        print(f"  ✓ Slide {slide_num}: {len(data['regions'])} regions")

        img.close()

    print(f"✓ Debug images saved to {debug_dir}")


def create_marp_presentation(slides_data, extracted_images, output_file):
    """Create final Marp presentation."""
    print("\nCreating Marp presentation...")

    def clean_content(content, slide_num):
        # Remove OCR tags
        content = re.sub(r'<\|ref\|>[^<]*<\|/ref\|>', '', content)
        content = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', content)
        content = re.sub(r'^\s*---\s*$', '', content, flags=re.MULTILINE)

        # Fix LaTeX
        content = LATEX_DISPLAY.sub(r'$$\1$$', content)
        content = LATEX_INLINE.sub(r'$\1$', content)

        content = re.sub(r'\n{3,}', '\n\n', content).strip()

        # Add images
        if slide_num in extracted_images:
            if content:
                content += "\n\n"
            for img_file in extracted_images[slide_num]:
                content += f"![plot](plots/{img_file})\n\n"

        return content.strip()

    marp_content = """---
marp: true
theme: default
paginate: true
math: mathjax
---

"""

    slides_output = []
    for slide_num in sorted(slides_data.keys()):
        content = clean_content(slides_data[slide_num]['full_content'], slide_num)
        if content.strip():
            slides_output.append(content)

    marp_content += '\n\n---\n\n'.join(slides_output)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(marp_content)

    print(f"✓ Marp presentation: {output_file}")
    return output_file


def main():
    """Main pipeline."""
    print("="*70)
    print("Complete PDF to Marp Pipeline")
    print("="*70)
    print()

    if len(sys.argv) < 2:
        print("Usage: python pdf_to_marp_full.py <pdf_file> [output_dir]")
        print("Example: python pdf_to_marp_full.py lecture.pdf lecture_output")
        return 1

    pdf_file = sys.argv[1]
    output_base = sys.argv[2] if len(sys.argv) > 2 else "lecture_output"

    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found: {pdf_file}")
        return 1

    # Setup directories
    images_dir = os.path.join(output_base, "images")
    plots_dir = os.path.join(output_base, "plots")
    debug_dir = os.path.join(output_base, "debug")

    print(f"PDF: {pdf_file}")
    print(f"Output: {output_base}")
    print()

    # Step 1: Convert PDF to images
    slide_files = convert_pdf_to_images(pdf_file, images_dir, dpi=200)
    if not slide_files:
        return 1

    # Step 2: Run OCR on all slides
    ocr_file = process_all_slides_ocr(slide_files, output_base)

    # Step 3: Parse OCR output
    print("\nParsing OCR output...")
    slides_data = parse_ocr_regions(ocr_file)
    print(f"✓ Parsed {len(slides_data)} slides")

    # Step 4: Extract plots
    extracted_images, plot_stats = extract_plots(
        images_dir, plots_dir, slides_data
    )

    # Step 5: Create debug visualization
    create_debug_visualization(images_dir, debug_dir, slides_data)

    # Step 6: Create Marp presentation
    marp_file = os.path.join(output_base, "lecture_marp.md")
    create_marp_presentation(slides_data, extracted_images, marp_file)

    # Summary
    print()
    print("="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Slides processed: {len(slide_files)}")
    print(f"Plots extracted: {plot_stats['extracted']}")
    print()
    print("Output files:")
    print(f"  Marp presentation: {marp_file}")
    print(f"  OCR output: {ocr_file}")
    print(f"  Plots: {plots_dir}/")
    print(f"  Debug images: {debug_dir}/")
    print()
    print("Next steps:")
    print(f"  marp {marp_file} -o presentation.pdf")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
