import os
import glob
from nano_extractor.core import NanoPDFExtractor

def test_data_folder():
    # Construct path directly to data/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, "data")
    output_folder = os.path.join(base_dir, "test_results")

    os.makedirs(output_folder, exist_ok=True)

    print(f"=== Searching for PDFs in: {data_folder} ===")
    
    # Locate all .pdf files inside pdf_extractor/data
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{data_folder}'.")
        print("Please ensure your PDF files are placed directly inside 'pdf_extractor/data/'.")
        return

    print(f"✓ Found {len(pdf_files)} PDF(s) to process.\n")

    # Initialize extractor (reads GEMINI_API_KEY directly from environment)
    extractor = NanoPDFExtractor()

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        clean_name = os.path.splitext(file_name)[0]
        output_prefix = os.path.join(output_folder, f"{clean_name}_extracted")

        print(f"--------------------------------------------------")
        print(f"Processing: {file_name}")
        print(f"--------------------------------------------------")

        try:
            # Process page 1 in hybrid mode (Text + OpenCV detection + Vision API)
            results = extractor.process_pdf(pdf_path, pages=[0], extract_mode="hybrid")
            page_data = results["pages"][0]

            print(f"✓ Native Text Extracted: {len(page_data['text'])} chars")
            print(f"✓ OpenCV Chart Regions Detected: {len(page_data['visual_analysis'])}")

            # Print native text sample
            text_preview = page_data['text'][:200].replace('\n', ' ')
            print(f"  └─ Text Preview: \"{text_preview}...\"")

            # Print chart extraction sample if any were detected
            if page_data['visual_analysis']:
                print("  └─ Detected Chart Analysis:")
                for figure in page_data['visual_analysis']:
                    print(f"     • Figure {figure['figure_index']} (Box: {figure['bbox']})")
                    analysis_snippet = figure['data'][:150].replace('\n', ' ')
                    print(f"       Data Snippet: {analysis_snippet}...")

            # Save Markdown and JSON outputs to test_results/
            extractor.export(results, output_fmt="both", output_prefix=output_prefix)
            print(f"✓ Saved exports to 'test_results/{clean_name}_extracted.md|json'\n")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}\n")

if __name__ == "__main__":
    test_data_folder()