import os
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Literal, Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import pymupdf  # PyMuPDF engine
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import onnxruntime as ort
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Storage integration
from nano_extractor.storage import BaseStorageProvider, LocalStorageProvider

# Load environment variables from .env file
load_dotenv()


class MLLayoutDetector:
    """Layer 3: ONNX-based DocLayout-YOLO engine for ML document layout analysis."""

    CLASSES = ["Text", "Title", "Header", "Footer", "Figure", "Table", "Equation"]

    def __init__(self, model_path: Optional[str] = "models/doclayout_yolo.onnx"):
        self.session = None
        if model_path and os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(
                    model_path, 
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            except Exception as e:
                print(f"[Warning] Failed to load ML model: {e}. Falling back to OpenCV heuristics.")

    def detect_layout(self, pil_image: Image.Image, conf_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Executes ML inference on a rendered page image to identify document elements."""
        if not self.session:
            return []

        # Resize image for standard object detection model input
        img_w, img_h = pil_image.size
        resized = pil_image.resize((640, 640))
        img_data = np.array(resized).astype(np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(img_data, axis=0)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        detected_regions = []
        # Expecting tensor shape [1, N, 6] -> (x1, y1, x2, y2, conf, class_id)
        if len(outputs) > 0 and len(outputs[0]) > 0:
            for pred in outputs[0][0]:
                if len(pred) < 6:
                    continue
                score = float(pred[4])
                if score < conf_threshold:
                    continue

                cls_id = int(pred[5])
                label = self.CLASSES[cls_id] if cls_id < len(self.CLASSES) else "Figure"

                # Scale coordinates back to original image dimensions
                x1 = int((pred[0] / 640.0) * img_w)
                y1 = int((pred[1] / 640.0) * img_h)
                x2 = int((pred[2] / 640.0) * img_w)
                y2 = int((pred[3] / 640.0) * img_h)

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                detected_regions.append({
                    "label": label,
                    "bbox": (x1, y1, w, h),
                    "confidence": score
                })

        return detected_regions


class NanoPDFExtractor:
    """A 5-layer hierarchical hybrid PDF extraction pipeline.

    - Layer 1: PyMuPDF (pymupdf) -> Ultra-fast block text extraction
    - Layer 2: pdfplumber -> Deterministic grid table parsing
    - Layer 3: DocLayout-YOLO (ONNX) -> ML-driven layout region classification
    - Layer 4: OpenCV -> Boundary contour refinement & secondary edge heuristics
    - Layer 5: Gemini 2.5 Flash -> Multimodal vision analysis of isolated visual crops
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: str = "gemini-2.5-flash",
        layout_model_path: Optional[str] = "models/doclayout_yolo.onnx",
        storage_provider: Optional[BaseStorageProvider] = None,
        embed_base64: bool = True
    ):
        resolved_key = (
            api_key 
            or os.environ.get("GEMINI_API_KEY") 
            or os.environ.get("gemini_key")
        )
        
        self.client = genai.Client(api_key=resolved_key)
        self.model_name = model_name
        self.storage = storage_provider or LocalStorageProvider()
        self.embed_base64 = embed_base64
        
        # Layer 3 ML Layout Model
        self.ml_detector = MLLayoutDetector(model_path=layout_model_path)

    @staticmethod
    def _image_to_base64(pil_img: Image.Image) -> str:
        """Helper to serialize PIL images as Base64 Data URIs."""
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def refine_bbox_with_opencv(
        self, 
        pil_image: Image.Image, 
        ml_bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Layer 4: Refines and snaps ML bounding boxes tightly around actual inner contours."""
        x, y, w, h = ml_bbox
        img_w, img_h = pil_image.size
        
        crop = pil_image.crop((x, y, x + w, y + h))
        cv_img = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return ml_bbox

        min_x = min([cv2.boundingRect(c)[0] for c in contours])
        min_y = min([cv2.boundingRect(c)[1] for c in contours])
        max_r = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
        max_b = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

        margin = 8
        ref_x = max(0, x + min_x - margin)
        ref_y = max(0, y + min_y - margin)
        ref_w = min(img_w - ref_x, (max_r - min_x) + (margin * 2))
        ref_h = min(img_h - ref_y, (max_b - min_y) + (margin * 2))

        return (ref_x, ref_y, ref_w, ref_h)

    def detect_chart_bounding_boxes(
        self, 
        pil_image: Image.Image, 
        min_area_ratio: float = 0.02, 
        max_area_ratio: float = 0.85
    ) -> List[Tuple[int, int, int, int]]:
        """Layer 4 Fallback: OpenCV edge heuristics for visual region detection."""
        open_cv_image = np.array(pil_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        img_height, img_width, _ = open_cv_image.shape
        total_area = img_height * img_width

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            aspect_ratio = float(w) / h

            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                continue

            if (total_area * min_area_ratio) < box_area < (total_area * max_area_ratio):
                roi = gray[y:y+h, x:x+w]
                edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(edges > 0) / float(w * h)

                if edge_density > 0.015:
                    margin = 10
                    x_m = max(0, x - margin)
                    y_m = max(0, y - margin)
                    w_m = min(img_width - x_m, w + (margin * 2))
                    h_m = min(img_height - y_m, h + (margin * 2))
                    bounding_boxes.append((x_m, y_m, w_m, h_m))

        return bounding_boxes

    def _extract_page_text_pymupdf(self, pymupdf_page: pymupdf.Page) -> List[Dict[str, Any]]:
        """Layer 1: Fast block text extraction using PyMuPDF."""
        blocks = pymupdf_page.get_text("blocks")
        text_blocks = []
        
        for b in blocks:
            text_content = b[4].strip()
            if text_content and b[6] == 0:
                text_blocks.append({
                    "bbox": [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                    "text": text_content,
                    "type": "text"
                })
                
        return text_blocks

    def _extract_native_tables_plumber(self, plumber_page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
        """Layer 2: Extract vector tables via pdfplumber into Markdown strings with bounding box."""
        tables = plumber_page.find_tables()
        extracted_tables = []
        
        for t in tables:
            raw_table = t.extract()
            clean_table = [[str(cell or "").strip() for cell in row] for row in raw_table if any(row)]
            if len(clean_table) > 1:
                header = clean_table[0]
                rows = clean_table[1:]
                
                md_str = "| " + " | ".join(header) + " |\n"
                md_str += "| " + " | ".join(["---"] * len(header)) + " |\n"
                for row in rows:
                    md_str += "| " + " | ".join(row) + " |\n"
                
                extracted_tables.append({
                    "bbox": list(t.bbox),
                    "markdown": md_str,
                    "type": "native_table"
                })
                
        return extracted_tables

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        retry=retry_if_exception_type(APIError)
    )
    def _analyze_image_patch_with_nano(self, image_patch: Image.Image, prompt: str) -> str:
        """Layer 5: Analyze image crop with Gemini 2.5 Flash."""
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
        """Executes the 5-layer pipeline on target PDF pages."""
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
                    "native_tables": [t["markdown"] for t in native_tables],
                    "visual_analysis": []
                }
                
                # Layer 3, 4 & 5: ML + OpenCV + Gemini Analysis
                if extract_mode == "hybrid" and page_idx in page_images_map:
                    page_img = page_images_map[page_idx]
                    
                    # Layer 3: ML Layout Detection
                    ml_regions = self.ml_detector.detect_layout(page_img)
                    figure_boxes = [r["bbox"] for r in ml_regions if r["label"] in ["Figure", "Table", "Equation"]]
                    
                    # Layer 4 Fallback: Use OpenCV heuristics if ML session is not available
                    if not figure_boxes and not self.ml_detector.session:
                        figure_boxes = self.detect_chart_bounding_boxes(page_img)
                    
                    prompt = (
                        "Analyze this image crop. "
                        "If it contains a graph, chart, or visual table, extract its structure into Markdown. "
                        "If it contains only regular text or instructions, return 'NO_GRAPH'."
                    )
                    
                    fig_counter = 1
                    for box_idx, ml_box in enumerate(figure_boxes):
                        refined_bbox = self.refine_bbox_with_opencv(page_img, ml_box)
                        x, y, w, h = refined_bbox
                        
                        crop = page_img.crop((x, y, x + w, y + h))
                        
                        # Layer 5: Gemini Multimodal Analysis
                        analysis = self._analyze_image_patch_with_nano(crop, prompt)
                        
                        if "NO_GRAPH" in analysis.strip():
                            continue

                        # Extract image URI (Base64 data URI by default, or storage relative path)
                        if self.embed_base64:
                            image_uri = self._image_to_base64(crop)
                        else:
                            rel_key = f"crops/{pdf_stem}/p{page_idx + 1}_fig{fig_counter}.png"
                            image_uri = self.storage.save_image(crop, rel_key)
                        
                        page_data["visual_analysis"].append({
                            "figure_index": fig_counter,
                            "bbox": [x, y, w, h],
                            "image_path": image_uri,
                            "type": "visual_figure",
                            "data": analysis
                        })
                        fig_counter += 1

                results["pages"].append(page_data)
                
        doc_pymupdf.close()
        return results

    def export(self, data: Dict[str, Any], output_fmt: Literal["json", "md", "both"], output_prefix: str) -> None:
        """Exports hierarchical extraction results to JSON or Markdown, interleaving visual figures inline."""
        if output_fmt in ["json", "both"]:
            with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        if output_fmt in ["md", "both"]:
            md_content = ""
            for page in data["pages"]:
                md_content += f"# Page {page['page_number']}\n\n"
                
                # Consolidate all elements for dynamic vertical interleaving
                elements = []
                
                for b in page.get("text_blocks", []):
                    elements.append({
                        "y": b["bbox"][1], 
                        "content": b["text"]
                    })
                    
                for idx, visual in enumerate(page.get("visual_analysis", [])):
                    img_md = f"### Figure {visual['figure_index']}\n\n"
                    img_md += f"![Page {page['page_number']} Figure {visual['figure_index']}]({visual['image_path']})\n\n"
                    img_md += f"**Visual Analysis:**\n{visual['data']}\n"
                    
                    elements.append({
                        "y": visual["bbox"][1],
                        "content": img_md
                    })

                # Sort elements based on vertical position (top to bottom)
                elements.sort(key=lambda item: item["y"])
                
                # Render vertically interleaved page output
                for elem in elements:
                    md_content += f"{elem['content']}\n\n"

                md_content += "---\n\n"
            
            with open(f"{output_prefix}.md", "w", encoding="utf-8") as f:
                f.write(md_content)