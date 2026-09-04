import os
from nano_extractor.core import NanoPDFExtractor

def run_basic_test(pdf_file_path: str):
    print(f"--- Testing NanoPDFExtractor on: {pdf_file_path} ---")
    
    # 1. Initialize
    extractor = NanoPDFExtractor()
    
    # 2. Test Page 1 Extraction
    print("[1/3] Extracting Page 1...")
    results = extractor.process_pdf(pdf_file_path, pages=[0], extract_mode="hybrid")
    
    page = results["pages"][0]
    print(f"✓ Page Number: {page['page_number']}")
    print(f"✓ Extracted Text Length: {len(page['text'])} characters")
    print(f"✓ Detected Visual/Chart Elements: {len(page['visual_analysis'])}")
    
    # Print preview of extracted text
    print("\n--- Text Preview (First 300 chars) ---")
    print(page['text'][:300] + ("..." if len(page['text']) > 300 else ""))
    
    # Print OpenCV detection results if any charts were found
    if page['visual_analysis']:
        print("\n--- Chart/Visual Analysis Output ---")
        for idx, item in enumerate(page['visual_analysis']):
            print(f"Figure {item['figure_index']} (BBox: {item['bbox']}):")
            print(item['data'])
    else:
        print("\n(No chart/graph bounding boxes detected on this page)")

    # 3. Test Exporting
    print("\n[3/3] Exporting test results...")
    extractor.export(results, output_fmt="both", output_prefix="test_output")
    print("✓ Output saved to test_output.json and test_output.md")

if __name__ == "__main__":
    # Replace with the path to a local PDF file
    SAMPLE_PDF = "sample.pdf" 
    
    if os.path.exists(SAMPLE_PDF):
        run_basic_test(SAMPLE_PDF)
    else:
        print(f"Please place a test PDF file at '{SAMPLE_PDF}' to run the test.")