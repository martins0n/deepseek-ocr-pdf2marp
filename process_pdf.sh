#!/bin/bash
# Complete PDF to Marp pipeline with debug output
# Usage: ./process_pdf.sh <pdf_file> [output_dir]

set -e

PYTHON="~/.pyenv/versions/deepseek-ocr/bin/python"
SCRIPT="pdf_to_marp_full.py"

echo "======================================================================="
echo "PDF to Marp - Complete Pipeline"
echo "======================================================================="
echo ""

# Check if Python environment exists
if [ ! -f ~/.pyenv/versions/deepseek-ocr/bin/python ]; then
    echo "Error: Python environment not found"
    echo "Please run setup.sh first"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <pdf_file> [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 lecture.pdf"
    echo "  $0 lecture.pdf custom/output/dir"
    exit 1
fi

PDF_FILE="$1"
OUTPUT_DIR="${2:-lecture/lecture_slides}"

if [ ! -f "$PDF_FILE" ]; then
    echo "Error: PDF file not found: $PDF_FILE"
    exit 1
fi

echo "Processing: $PDF_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This will:"
echo "  1. Convert PDF to images"
echo "  2. Run OCR on each slide (may take several minutes)"
echo "  3. Extract plots and charts"
echo "  4. Create Marp presentation"
echo "  5. Generate debug visualizations"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run the pipeline
~/.pyenv/versions/deepseek-ocr/bin/python "$SCRIPT" "$PDF_FILE" "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "SUCCESS!"
    echo "======================================================================="
    echo ""
    echo "Your files are ready:"
    echo "  📄 Presentation: $OUTPUT_DIR/lecture_marp.md"
    echo "  📊 Plots: $OUTPUT_DIR/plots/"
    echo "  🔍 Debug: $OUTPUT_DIR/debug/"
    echo ""
    echo "Export to PDF:"
    echo "  marp $OUTPUT_DIR/lecture_marp.md -o presentation.pdf"
    echo ""
fi

exit $EXIT_CODE
