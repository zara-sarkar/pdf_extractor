import os
import json
from io import BytesIO
from typing import Literal, Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from .env file
load_dotenv()


class NanoPDFExtractor:
    """A hybrid PDF extraction module combining vector text extraction and computer vision.

    Uses `pdfplumber` for deterministic text/footnote parsing and OpenCV bounding-box 
    heuristics coupled with Gemini Vision models to extract structured data from charts and graphs.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """Initializes the NanoPDFExtractor with GenAI API credentials and settings.

        Args:
            api_key: Optional explicit Gemini API key. If omitted, falls back to
                `GEMINI_API_KEY` or `gemini_key` environment variables.
            model_name: The Gemini model identifier to use for visual extraction.
                Defaults to "gemini-2.5-flash".
        """
        resolved_key = (
            api_key 
            or os.environ.get("GEMINI_API_KEY") 
            or os.environ.get("gemini_key")
        )
        
        self.client = genai.Client(api_key=resolved_key)
        self.model_name = model_name

    def detect_chart_bounding_boxes(
        self, 
        pil_image: Image.Image, 
        min_area_ratio: float = 0.03, 
        max_area_ratio: float = 0.85
    ) -> List[Tuple[int, int, int, int]]:
        """Detects potential chart, graph, or figure regions in an image using OpenCV heuristics.

        Args:
            pil_image: PIL Image object of the rendered PDF page.
            min_area_ratio: Minimum visual region size relative to total page area (0.0 to 1.0).
            max_area_ratio: Maximum visual region size relative to total page area (0.0 to 1.0).

        Returns:
            A list of bounding box tuples in `(x, y, width, height)` pixel format.
        """
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        img_height, img_width, _ = open_cv_image.shape
        total_area = img_height * img_width

        # Grayscale and adaptive thresholding to isolate structural lines/axes
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological dilation to group nearby chart elements (bars, keys, axes)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            aspect_ratio = float(w) / h

            # Filter candidate boxes based on area and reasonable chart proportions
            if (total_area * min_area_ratio) < box_area < (total_area * max_area_ratio) and (0.2 < aspect_ratio < 5.0):
                # Edge density check to distinguish graphical figures from text blocks
                roi = gray[y:y+h, x:x+w]
                edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(edges > 0) / (w * h)

                if edge_density > 0.04:
                    margin = 10
                    x_m = max(0, x - margin)
                    y_m = max(0, y - margin)
                    w_m = min(img_width - x_m, w + (margin * 2))
                    h_m = min(img_height - y_m, h + (margin * 2))
                    bounding_boxes.append((x_m, y_m, w_m, h_m))

        return bounding_boxes

    def _extract_page_text_native(self, page: pdfplumber.page.Page) -> Dict[str, Any]:
        """Extracts text and native vector tables directly via PDF vector layers.

        Args:
            page: Open `pdfplumber.page.Page` instance.

        Returns:
            Dictionary containing extracted raw text and extracted native markdown tables.
        """
        text = page.extract_text(layout=True) or ""
        tables = page.extract_tables()
        
        md_tables = []
        for table in tables:
            clean_table = [[str(cell or "").strip() for cell in row] for row in table if any(row)]
            if len(clean_table) > 1:
                header = clean_table[0]
                rows = clean_table[1:]
                
                md_str = "| " + " | ".join(header) + " |\n"
                md_str += "| " + " | ".join(["---"] * len(header)) + " |\n"
                for row in rows:
                    md_str += "| " + " | ".join(row) + " |\n"
                md_tables.append(md_str)
                
        return {"text": text, "tables": md_tables}

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        retry=retry_if_exception_type(APIError)
    )
    def _analyze_image_patch_with_nano(self, image_patch: Image.Image, prompt: str) -> str:
        """Sends an isolated high-res crop to the vision LLM via Chat session.

        Follows Google GenAI guidelines by using `client.chats.create` and `chat.send_message`
        to prevent Automatic Function Calling (AFC) deprecation warnings.
        """
        system_instruction = (
            "You are an expert document extraction system. "
            "Your task is to analyze the image crop and extract structured information:\n"
            "1. IF IT IS A TABLE: Extrapolate and render it directly as a standard, valid Markdown table.\n"
            "2. IF IT IS A CHART OR GRAPH: Extract all x/y-axis labels, legend keys, trends, and list key values as bullet points or a structured Markdown table.\n"
            "3. IF IT IS PLAIN TEXT OR INSTRUCTIONS: Summarize or reproduce the text concisely.\n"
            "Do not include conversational filler or intros like 'Here is the extracted data:'."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1
        )

        # Uses recommended chat API to avoid AFC warning on generate_content
        chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )
        
        response = chat.send_message([image_patch, prompt])
        return response.text

    def process_pdf(
        self, 
        pdf_path: str, 
        pages: Optional[List[int]] = None, 
        extract_mode: Literal["text", "hybrid"] = "hybrid"
    ) -> Dict[str, Any]:
        """Main pipeline to extract structured content from target PDF pages.

        Args:
            pdf_path: Filepath to target PDF document.
            pages: List of 0-indexed page numbers to process. If None, processes all pages.
            extract_mode: Extraction strategy:
                - `"text"`: Native text layer extraction only.
                - `"hybrid"`: Native text layer + OpenCV chart detection + LLM visual extraction.

        Returns:
            Structured dictionary containing extracted text, native tables, and visual analysis.
        """
        results = {"pages": []}
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            target_pages = pages if pages is not None else list(range(total_pages))
            
            page_images_map = {}
            if extract_mode == "hybrid":
                min_page = min(target_pages) + 1
                max_page = max(target_pages) + 1
                rendered_images = convert_from_path(
                    pdf_path, 
                    first_page=min_page, 
                    last_page=max_page, 
                    dpi=300
                )
                
                for idx, page_idx in enumerate(range(min_page - 1, max_page)):
                    if page_idx in target_pages:
                        page_images_map[page_idx] = rendered_images[idx]
            
            for page_idx in target_pages:
                page = pdf.pages[page_idx]
                native_data = self._extract_page_text_native(page)
                
                page_data = {
                    "page_number": page_idx + 1, 
                    "text": native_data["text"],
                    "native_tables": native_data["tables"],
                    "visual_analysis": []
                }
                
                if extract_mode == "hybrid" and page_idx in page_images_map:
                    page_img = page_images_map[page_idx]
                    chart_boxes = self.detect_chart_bounding_boxes(page_img)
                    
                    prompt = (
                        "Analyze this image crop. "
                        "Format any tables using standard Markdown tables (`| ... |`). "
                        "Format charts/graphs using structured Markdown lists or tables."
                    )
                    
                    for box_idx, (x, y, w, h) in enumerate(chart_boxes):
                        crop = page_img.crop((x, y, x + w, y + h))
                        analysis = self._analyze_image_patch_with_nano(crop, prompt)
                        page_data["visual_analysis"].append({
                            "figure_index": box_idx + 1,
                            "bbox": [x, y, w, h],
                            "data": analysis
                        })

                results["pages"].append(page_data)
                
        return results

    def export(self, data: Dict[str, Any], output_fmt: Literal["json", "md", "both"], output_prefix: str) -> None:
        """Exports extraction results into specified structured file formats.

        Args:
            data: Dictionary results produced by `process_pdf()`.
            output_fmt: Desired export format (`"json"`, `"md"`, or `"both"`).
            output_prefix: Target file path prefix without extension.
        """
        if output_fmt in ["json", "both"]:
            with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        if output_fmt in ["md", "both"]:
            md_content = ""
            for page in data["pages"]:
                md_content += f"# Page {page['page_number']}\n\n"
                md_content += f"## Extracted Text & Footnotes\n\n{page['text']}\n\n"
                
                if page["native_tables"]:
                    md_content += "## Extracted Native Tables\n\n"
                    for idx, tbl in enumerate(page["native_tables"]):
                        md_content += f"### Native Table {idx + 1}\n\n{tbl}\n\n"
                        
                if page["visual_analysis"]:
                    md_content += "## Extracted Visual Figures & Visual Tables\n\n"
                    for visual in page["visual_analysis"]:
                        md_content += f"### Figure/Table Crop {visual['figure_index']}\n"
                        md_content += f"**Bounding Box (x, y, w, h):** `{visual['bbox']}`\n\n"
                        md_content += f"{visual['data']}\n\n"
            
            with open(f"{output_prefix}.md", "w", encoding="utf-8") as f:
                f.write(md_content)