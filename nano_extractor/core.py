import os
import json
from io import BytesIO
from typing import Literal, Dict, Any, List, Tuple
import cv2
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
from google import genai


class NanoPDFExtractor:
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model_name = "gemini-2.5-flash"

    def detect_chart_bounding_boxes(
        self, 
        pil_image: Image.Image, 
        min_area_ratio: float = 0.03, 
        max_area_ratio: float = 0.85
    ) -> List[Tuple[int, int, int, int]]:
        """Detects potential chart/graph regions using OpenCV contour heuristics."""
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        img_height, img_width, _ = open_cv_image.shape
        total_area = img_height * img_width

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            aspect_ratio = float(w) / h

            if (total_area * min_area_ratio) < box_area < (total_area * max_area_ratio) and (0.2 < aspect_ratio < 5.0):
                margin = 10
                x_m = max(0, x - margin)
                y_m = max(0, y - margin)
                w_m = min(img_width - x_m, w + (margin * 2))
                h_m = min(img_height - y_m, h + (margin * 2))
                bounding_boxes.append((x_m, y_m, w_m, h_m))

        return bounding_boxes

    def _extract_page_text_native(self, page) -> str:
        """Extracts text directly via PDF vector layers (capturing small print/footnotes)."""
        return page.extract_text(layout=True) or ""

    def _analyze_image_patch_with_nano(self, image_patch: Image.Image, prompt: str) -> str:
        """Sends an isolated high-res crop to the vision model."""
        buffer = BytesIO()
        image_patch.save(buffer, format="PNG")
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                {"mime_type": "image/png", "data": buffer.getvalue()},
                prompt
            ]
        )
        return response.text

    def process_pdf(
        self, 
        pdf_path: str, 
        pages: List[int] = None, 
        extract_mode: Literal["text", "hybrid"] = "hybrid"
    ) -> Dict[str, Any]:
        results = {"pages": []}
        
        with pdfplumber.open(pdf_path) as pdf:
            target_pages = pages if pages else list(range(len(pdf.pages)))
            
            for page_idx in target_pages:
                page = pdf.pages[page_idx]
                page_data = {
                    "page_number": page_idx + 1, 
                    "text": "", 
                    "visual_analysis": []
                }
                
                # Step 1: Deterministic PDF layer extraction for perfect text & footnotes
                page_data["text"] = self._extract_page_text_native(page)
                
                # Step 2: High-DPI OpenCV detection + Vision model extraction for graphs
                if extract_mode == "hybrid":
                    # Render page at 300 DPI for high fidelity
                    page_images = convert_from_path(
                        pdf_path, 
                        first_page=page_idx + 1, 
                        last_page=page_idx + 1, 
                        dpi=300
                    )
                    
                    if page_images:
                        page_img = page_images[0]
                        chart_boxes = self.detect_chart_bounding_boxes(page_img)
                        
                        prompt = (
                            "Analyze this isolated chart, graph, or figure crop. "
                            "Extract all key data points, legend mappings, x/y-axis labels, "
                            "and values into a clean, structured JSON format."
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

    def export(self, data: Dict[str, Any], output_fmt: Literal["json", "md", "both"], output_prefix: str):
        if output_fmt in ["json", "both"]:
            with open(f"{output_prefix}.json", "w") as f:
                json.dump(data, f, indent=2)
                
        if output_fmt in ["md", "both"]:
            md_content = ""
            for page in data["pages"]:
                md_content += f"# Page {page['page_number']}\n\n"
                md_content += f"## Extracted Text & Footnotes\n\n{page['text']}\n\n"
                if page["visual_analysis"]:
                    md_content += "## Extracted Charts & Visuals\n\n"
                    for visual in page["visual_analysis"]:
                        md_content += f"### Figure {visual['figure_index']}\n"
                        md_content += f"**Bounding Box:** `{visual['bbox']}`\n\n"
                        md_content += f"{visual['data']}\n\n"
            
            with open(f"{output_prefix}.md", "w") as f:
                f.write(md_content)