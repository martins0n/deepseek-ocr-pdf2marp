#!/usr/bin/env python3
"""
Single Slide Converter

Convert a single PDF page or image into a Marp slide and generate a debug
image with DeepSeek-OCR bounding boxes.
"""

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer
import contextlib

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None


MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
PROMPT = "<image>\n<|grounding|>Convert the slide to markdown."
DEFAULT_BASE = 1024
LATEX_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
LATEX_INLINE = re.compile(r"\\\((.+?)\\\)")
REGION_PATTERN = re.compile(
    r"<\|ref\|>([^<]+)<\|/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|/det\|>(.*?)(?=<\|ref\||$)",
    re.DOTALL,
)

# Bounding box validation thresholds
BBOX_TRANSFORM_SIZE_THRESHOLD = 0.75  # Filter boxes > 75% of base_size from transform
BBOX_DRAW_SIZE_THRESHOLD = 0.9  # Don't draw boxes > 90% of base_size
BBOX_EDGE_MARGIN = 5  # Pixels from edges to consider suspicious
BBOX_MIN_SIZE = 5  # Minimum refined box size in pixels
BBOX_MAX_RATIO = 0.95  # Maximum refined box size as ratio of image


def load_model(device: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    return model, tokenizer


def run_ocr(model, tokenizer, image_path: Path):
    output_dir = image_path.parent
    captured = io.StringIO()
    with torch.inference_mode():
        with contextlib.redirect_stdout(captured):
            result = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=DEFAULT_BASE,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False,
            )
    text = result if isinstance(result, str) and result.strip() else captured.getvalue()
    return text.strip()


def prepare_image(input_path: Path, page: int, dpi: int, output_dir: Path) -> Path:
    if input_path.suffix.lower() == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF support (pip install pymupdf)")
        doc = fitz.open(str(input_path))
        if page < 1 or page > len(doc):
            total = len(doc)
            doc.close()
            raise ValueError(f"Page {page} out of range (1-{total})")
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = doc[page - 1].get_pixmap(matrix=matrix)
        image_path = output_dir / f"{input_path.stem}_page_{page:03d}.png"
        pix.save(str(image_path))
        doc.close()
        return image_path
    return input_path


def extract_slide_block(text: str):
    cleaned = re.sub(r"={20,}.*?={20,}", "", text, flags=re.DOTALL)
    match = re.search(r"# Slide (\d+)/\d+\n+---\n+(.*?)(?=\n# Slide |\Z)", cleaned, re.DOTALL)
    if match:
        slide_num = int(match.group(1))
        body = match.group(2)
    else:
        slide_num = 1
        body = cleaned
    return slide_num, body.strip()


def parse_regions(slide_body: str):
    regions = []
    for match in REGION_PATTERN.finditer(slide_body):
        region_type = match.group(1)
        bbox_str = match.group(2)
        try:
            coords = [int(x.strip()) for x in bbox_str.split(',')]
            if len(coords) == 4:
                regions.append({
                    "type": region_type,
                    "bbox": (coords[0], coords[1], coords[2], coords[3]),
                })
        except ValueError:
            continue
    return regions


def clean_slide_content(slide_body: str):
    text = re.sub(r"<\|ref\|>[^<]*<\|/ref\|>", "", slide_body)
    text = re.sub(r"<\|det\|>\[\[.*?\]\]<\|/det\|>", "", text)
    text = LATEX_DISPLAY.sub(r"$$\1$$", text)
    text = LATEX_INLINE.sub(r"$\1$", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def solve_linear(src_vals, dst_vals, fallback_scale):
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
        
        # Skip boxes that are too large
        # These are often full-slide images or incorrectly detected regions
        if box_width > base_size * BBOX_TRANSFORM_SIZE_THRESHOLD or box_height > base_size * BBOX_TRANSFORM_SIZE_THRESHOLD:
            continue
            
        # Skip boxes with invalid or suspicious coordinates
        # Boxes at (0,0) or spanning nearly the full coordinate space
        if (x1 <= BBOX_EDGE_MARGIN and y1 <= BBOX_EDGE_MARGIN) or (x2 >= base_size - BBOX_EDGE_MARGIN and y2 >= base_size - BBOX_EDGE_MARGIN):
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


COLORS = {
    "title": (255, 0, 0),
    "sub_title": (255, 128, 0),
    "text": (0, 255, 0),
    "equation": (0, 255, 255),
    "image": (255, 0, 255),
    "image_caption": (128, 0, 255),
    "table": (255, 255, 0),
}


def draw_debug(image_path: Path, regions, transform, output_path: Path):
    img = Image.open(image_path).convert("RGB")
    gray = np.array(img.convert("L"))
    width, height = img.size
    draw = ImageDraw.Draw(img)

    default_scale = max(width / DEFAULT_BASE, height / DEFAULT_BASE)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            max(12, int(24 * default_scale)),
        )
    except Exception:
        font = ImageFont.load_default()

    for region in regions:
        # Skip drawing boxes that are clearly problematic
        x1, y1, x2, y2 = region["bbox"]
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Skip extremely large boxes that likely indicate OCR errors
        if box_width > DEFAULT_BASE * BBOX_DRAW_SIZE_THRESHOLD or box_height > DEFAULT_BASE * BBOX_DRAW_SIZE_THRESHOLD:
            continue
        
        bbox = apply_transform(region["bbox"], transform, width, height)
        bbox = refine_bbox(gray, bbox)
        
        # Additional validation: skip if refined box is still too large or invalid
        refined_width = bbox[2] - bbox[0]
        refined_height = bbox[3] - bbox[1]
        if refined_width < BBOX_MIN_SIZE or refined_height < BBOX_MIN_SIZE:
            continue
        if refined_width > width * BBOX_MAX_RATIO or refined_height > height * BBOX_MAX_RATIO:
            continue
        
        color = COLORS.get(region["type"], (255, 255, 255))
        line_width = max(1, int(4 * default_scale))
        for offset in range(line_width):
            draw.rectangle(
                [bbox[0] + offset, bbox[1] + offset, bbox[2] - offset, bbox[3] - offset],
                outline=color,
            )
        label = region["type"]
        label_pos = (bbox[0] + line_width, bbox[1] + line_width)
        draw.text(label_pos, label, fill=color, font=font)

    img.save(output_path, quality=95)
    return output_path


def build_marp(markdown_body: str, theme: str = "default"):
    return (
        f"---\nmarp: true\ntheme: {theme}\npaginate: true\nmath: mathjax\n---\n\n"
        + markdown_body.strip()
        + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a single slide to Marp")
    parser.add_argument("input", type=Path, help="PDF or image path")
    parser.add_argument("--page", type=int, default=1, help="PDF page (1-indexed)")
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    parser.add_argument("--output-dir", type=Path, default=Path("single_slide_output"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Target device (defaults to auto)")
    parser.add_argument("--theme", default="default", help="Marp theme")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    image_path = prepare_image(input_path, args.page, args.dpi, output_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"

    try:
        model, tokenizer = load_model(device)
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    try:
        ocr_text = run_ocr(model, tokenizer, image_path)
    except Exception as exc:
        print(f"OCR failed: {exc}", file=sys.stderr)
        return 1

    if not ocr_text:
        print("No OCR output produced", file=sys.stderr)
        return 1

    ocr_path = output_dir / f"{image_path.stem}_ocr.md"
    ocr_path.write_text(ocr_text, encoding="utf-8")

    slide_num, slide_body = extract_slide_block(ocr_text)
    regions = parse_regions(slide_body)
    cleaned = clean_slide_content(slide_body)
    marp_doc = build_marp(cleaned, theme=args.theme)

    marp_path = output_dir / f"slide_{slide_num:03d}_marp.md"
    marp_path.write_text(marp_doc, encoding="utf-8")

    image = Image.open(image_path)
    gray = np.array(image.convert("L"))
    transform = compute_transform(gray, regions)
    debug_path = output_dir / f"slide_{slide_num:03d}_debug.png"
    draw_debug(image_path, regions, transform, debug_path)

    print(f"✓ OCR saved to {ocr_path}")
    print(f"✓ Marp slide: {marp_path}")
    print(f"✓ Debug image: {debug_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
