import os
import json
from io import BytesIO
from pathlib import Path
from typing import Literal, Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import pymupdf  # PyMuPDF engine
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Storage integration
from nano_extractor.storage import BaseStorageProvider, LocalStorageProvider

# Load environment variables from .env file
load_dotenv()


class NanoPDFExtractor:
    """A hierarchical hybrid PDF extraction module combining vector text parsing and computer vision.

    Uses `pymupdf` (PyMuPDF) as Layer 1 for ultra-fast text/block layout extraction, `pdfplumber`
    as Layer 2 for deterministic grid table reconstruction, OpenCV layout heuristics as Layer 3
    for secondary visual DLA bounding-box isolation, and Gemini 2.5 Flash as Layer 4 for multimodal
    interpretation of charts, graphs, and complex diagrams.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: str = "gemini-2.5-flash",
        storage_provider: Optional[BaseStorageProvider] = None
    ):
        """Initializes the NanoPDFExtractor with GenAI API credentials and storage backend.

        Args:
            api_key: Gemini API key. Defaults to environment variable GEMINI_API_KEY or gemini_key.
            model_name: Vision LLM model identifier. Defaults to 'gemini-2.5-flash'.
            storage_provider: Abstract storage provider for cropped figures. Defaults to LocalStorageProvider.
        """
        resolved_key = (
            api_key 
            or os.environ.get("GEMINI_API_KEY") 
            or os.environ.get("gemini_key")
        )
        
        self.client = genai.Client(api_key=resolved_key)
        self.model_name = model_name
        self.storage = storage_provider or LocalStorageProvider()

    def detect_chart_bounding_boxes(
        self, 
        pil_image: Image.Image, 
        min_area_ratio: float = 0.02, 
        max_area_ratio: float = 0.85
    ) -> List[Tuple[int, int, int, int]]:
        """Detects potential chart, graph, or figure regions in an image using OpenCV heuristics (Layer 3 DLA).

        Args:
            pil_image: Target rendered page image.
            min_area_ratio: Minimum area fraction relative to total page area to qualify as a figure.
            max_area_ratio: Maximum area fraction relative to total page area to filter full-page containers.

        Returns:
            List of bounding box tuples in (x, y, w, h) format.
        """
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        img_height, img_width, _ = open_cv_image.shape
        total_area = img_height * img_width

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological structuring element to group graphical visual structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            aspect_ratio = float(w) / h

            # Filter out ultra-wide text banners or tall vertical margins
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                continue

            if (total_area * min_area_ratio) < box_area < (total_area * max_area_ratio):
                roi = gray[y:y+h, x:x+w]
                edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(edges > 0) / float(w * h)

                # Lowered density threshold to capture clean/sparse vector diagrams
                if edge_density > 0.015:
                    margin = 10
                    x_m = max(0, x - margin)
                    y_m = max(0, y - margin)
                    w_m = min(img_width - x_m, w + (margin * 2))
                    h_m = min(img_height - y_m, h + (margin * 2))
                    bounding_boxes.append((x_m, y_m, w_m, h_m))

        return bounding_boxes

    def _extract_page_text_pymupdf(self, pymupdf_page: pymupdf.Page) -> List[Dict[str, Any]]:
        """Layer 1: Extracts structured text blocks directly via PyMuPDF (pymupdf) C-engine."""
        blocks = pymupdf_page.get_text("blocks")
        text_blocks = []
        
        for b in blocks:
            # b tuple format: (x0, y0, x1, y1, "text", block_no, block_type)
            # block_type 0 = text, 1 = image
            text_content = b[4].strip()
            if text_content and b[6] == 0:
                text_blocks.append({
                    "bbox": [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                    "text": text_content
                })
                
        return text_blocks

    def _extract_native_tables_plumber(self, plumber_page: pdfplumber.page.Page) -> List[str]:
        """Layer 2: Extracts native vector grid tables via pdfplumber into Markdown tables."""
        tables = plumber_page.extract_tables()
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
                
        return md_tables

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        retry=retry_if_exception_type(APIError)
    )
    def _analyze_image_patch_with_nano(self, image_patch: Image.Image, prompt: str) -> str:
        """Layer 4: Sends an isolated visual crop to Gemini 2.5 Flash with retry handling."""
        system_instruction = (
            "You are an expert document extraction system. "
            "Your task is to analyze the image crop and extract structured visual information:\n"
            "1. IF THE IMAGE CONTAINS ONLY PLAIN TEXT, PARAGRAPHS, OR INSTRUCTIONS: "
            "Respond ONLY with the exact key phrase: NO_GRAPH\n"
            "2. IF IT IS A TABLE: Extrapolate and render it directly as a standard Markdown table.\n"
            "3. IF IT IS A CHART, GRAPH, OR DIAGRAM: Extract all x/y-axis labels, legend keys, trends, and list key values as bullet points or a structured Markdown table.\n"
            "Do not include conversational filler or intros like 'Here is the extracted data:'."
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1
        )

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
        """Hierarchical pipeline processing target PDF pages.

        Args:
            pdf_path: Path to the target PDF file.
            pages: Zero-indexed list of page numbers to process. If None, processes all pages.
            extract_mode: 'text' for rapid vector text parsing, or 'hybrid' to activate DLA & Vision LLM.

        Returns:
            Dict containing per-page hierarchical extraction structures.
        """
        results = {"pages": []}
        pdf_stem = Path(pdf_path).stem
        
        doc_pymupdf = pymupdf.open(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf_plumber:
            total_pages = len(doc_pymupdf)
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
                pymupdf_page = doc_pymupdf[page_idx]
                plumber_page = pdf_plumber.pages[page_idx]
                
                # Layer 1: PyMuPDF Block Text
                text_blocks = self._extract_page_text_pymupdf(pymupdf_page)
                combined_text = "\n\n".join([b["text"] for b in text_blocks])
                
                # Layer 2: pdfplumber Table Parsing
                native_tables = self._extract_native_tables_plumber(plumber_page)
                
                page_data = {
                    "page_number": page_idx + 1, 
                    "text": combined_text,
                    "text_blocks": text_blocks,
                    "native_tables": native_tables,
                    "visual_analysis": []
                }
                
                # Layer 3 & 4: OpenCV DLA + Gemini Vision Analysis
                if extract_mode == "hybrid" and page_idx in page_images_map:
                    page_img = page_images_map[page_idx]
                    chart_boxes = self.detect_chart_bounding_boxes(page_img)
                    
                    prompt = (
                        "Analyze this image crop. "
                        "If it contains a graph, chart, or visual table, extract its structure into Markdown. "
                        "If it contains only regular text or instructions, return 'NO_GRAPH'."
                    )
                    
                    fig_counter = 1
                    for box_idx, (x, y, w, h) in enumerate(chart_boxes):
                        crop = page_img.crop((x, y, x + w, y + h))
                        
                        analysis = self._analyze_image_patch_with_nano(crop, prompt)
                        
                        # Filter out non-graph crops flagged by Gemini Vision
                        if "NO_GRAPH" in analysis.strip():
                            continue

                        # Persist image via storage adapter
                        rel_key = f"crops/{pdf_stem}/p{page_idx + 1}_fig{fig_counter}.png"
                        image_uri = self.storage.save_image(crop, rel_key)
                        
                        page_data["visual_analysis"].append({
                            "figure_index": fig_counter,
                            "bbox": [x, y, w, h],
                            "image_path": image_uri,
                            "data": analysis
                        })
                        fig_counter += 1

                results["pages"].append(page_data)
                
        doc_pymupdf.close()
        return results

    def export(self, data: Dict[str, Any], output_fmt: Literal["json", "md", "both"], output_prefix: str) -> None:
        """Exports hierarchical extraction results into JSON, Markdown, or both formats.

        Args:
            data: Structured output payload generated by process_pdf().
            output_fmt: Desired export format ('json', 'md', or 'both').
            output_prefix: Output filepath prefix (without extensions).
        """
        if output_fmt in ["json", "both"]:
            with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        if output_fmt in ["md", "both"]:
            md_content = ""
            for page in data["pages"]:
                md_content += f"# Page {page['page_number']}\n\n"
                md_content += f"## Extracted Text & Footnotes\n\n{page['text']}\n\n"
                
                if page.get("native_tables"):
                    md_content += "## Extracted Native Tables\n\n"
                    for idx, tbl in enumerate(page["native_tables"]):
                        md_content += f"### Native Table {idx + 1}\n\n{tbl}\n\n"
                        
                if page.get("visual_analysis"):
                    md_content += "## Extracted Visual Figures & Visual Tables\n\n"
                    for visual in page["visual_analysis"]:
                        md_content += f"### Figure {visual['figure_index']}\n\n"
                        md_content += f"![Page {page['page_number']} Figure {visual['figure_index']}]({visual['image_path']})\n\n"
                        md_content += f"**Bounding Box (x, y, w, h):** `{visual['bbox']}`\n\n"
                        md_content += f"{visual['data']}\n\n"
            
            with open(f"{output_prefix}.md", "w", encoding="utf-8") as f:
                f.write(md_content)