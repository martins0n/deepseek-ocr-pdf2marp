# DeepSeek OCR v2 - PDF to Marp Converter

Convert PDF slides to Marp presentations with plot extraction and debug visualizations.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process Full PDF

```bash
./process_pdf.sh input.pdf [output_dir]
```

Output: Marp presentation, extracted plots, debug images with bounding boxes.

### Process Single Slide

```bash
python single_slide_converter.py input.pdf --page 1 --output-dir output
```

Output: Marp slide, debug visualization, OCR markdown.

## Example

See `example/` folder for sample PDFs and expected outputs.

## Files

- `process_pdf.sh` - Batch process entire PDFs
- `single_slide_converter.py` - Convert individual slides
- `pdf_to_marp_full.py` - Main pipeline script
- `requirements.txt` - Python dependencies