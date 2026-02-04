"""
Poster2Data AI Dashboard — B2B Internal Tool
Single-file Streamlit app with DeepSeek API integration and local API key management.
"""

import os
import sys
import json
import io
import re
import base64
from pathlib import Path
import html

import numpy as np
import streamlit as st
import pandas as pd
from openai import OpenAI
import pdfplumber

# PIL Image - always import for type hints
from PIL import Image

# OCR for PDF: pdf2image (needs poppler) or PyMuPDF (no poppler). Both need tesseract.
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_FILENAME = "config.json"
CONFIG_KEY = "deepseek_api_key"
CONFIG_KEY_OPENROUTER = "openrouter_api_key"
FUTU_BLUE = "#2A55E5"
MAX_UPLOAD_MB = 25
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".pdf")

# DeepSeek API (text only)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Vision API Configuration
# DeepSeek's official API doesn't support vision yet, so we use OpenRouter for vision models
# Set USE_VISION_API = True and provide OPENROUTER_API_KEY to use vision
USE_VISION_API = True  # Set to True to use Vision API instead of OCR

# OpenRouter supports many vision models: claude-3-5-sonnet, gpt-4o, gemini-pro-vision, etc.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Recommended vision models (in order of preference for Chinese OCR):
# - "google/gemini-2.0-flash-001" (fast, good for Chinese, generous free tier)
# - "anthropic/claude-3-5-sonnet" (excellent quality, paid)
# - "openai/gpt-4o" (excellent quality, paid)
# - "google/gemini-pro-vision" (good quality, free tier available)
VISION_MODEL = "google/gemini-2.0-flash-001"

# Strict output schema (exact column names only)
OUTPUT_COLUMNS = ["Source File", "Event Name", "Date", "Time", "Venue"]

# OCR: prefer Traditional Chinese + English (chi_tra+eng). Requires tesseract lang data: e.g. brew install tesseract-lang
OCR_LANG = "chi_tra+eng"
# Higher DPI for better OCR accuracy
OCR_DPI = 300
# Tesseract config: PSM 3 = fully automatic page segmentation (default), OEM 3 = default LSTM
TESSERACT_CONFIG = r'--oem 3 --psm 3'

SYSTEM_PROMPT = """You are a data extraction assistant. **The input text is primarily in Traditional Chinese (繁體中文).** Preserve Chinese for event_name and venue; do not translate to English. Extract 活動名稱、日期、時間、地點 accordingly. The text may come from OCR (posters, screenshots) and can contain typos or mixed Chinese/English. Extract event details as best you can.

Rules:
- Return ONLY valid JSON, no markdown, no explanation, no code fences.
- One event: {"event_name": "...", "date": "YYYY-MM-DD", "time": "HH:MM", "venue": "..."}
- Multiple events: array of such objects.
- Use empty string "" only when the field is truly absent. Do NOT leave "venue" empty if the text contains any location/address info.
- **Venue (地點/地址/場地) is required when present:** Look for labels like 地點、地址、場地、Location、Venue, or any line that looks like an address (e.g. 台中市...區...路...號, 基督書院地下操場, building names + 操場/廣場). Copy or infer the venue string even if OCR is slightly wrong. Only use "" for venue when no location is mentioned at all.
- date: normalize to YYYY-MM-DD (e.g. 2025/04/27 → "2025-04-27", 2019/6月7日 → "2019-06-07").
- time: normalize to 24h HH:MM (e.g. 12NN-6PM → "12:00"-"18:00", 早上9:00~晚上6:00 → "09:00"-"18:00").
- If the text describes an event (title, date, time, or venue), always return at least one object. Guess reasonable values from context when OCR is noisy.
- Keep event_name and venue in 繁體中文 when the source text is in Chinese (e.g. "微笑多肉市集", "基督書院地下操場", "台中市神岡區大豐路4段52號")."""

# Hint prepended to user message so the model treats content as Traditional Chinese
USER_TEXT_PREFIX = "以下為繁體中文活動海報／截圖的 OCR 文字，請擷取活動名稱、日期、時間、地點（保留中文）：\n\n"

# Vision API prompt (for direct image understanding, no OCR)
VISION_SYSTEM_PROMPT = """你是一個活動資訊擷取助手。請直接從圖片中讀取活動海報的內容，提取活動名稱、日期、時間、地點。

重要規則：
- 只回傳 JSON，不要 markdown、不要解釋、不要 code fence
- 單一活動: {"event_name": "...", "date": "YYYY-MM-DD", "time": "HH:MM", "venue": "..."}
- 多個活動: 回傳陣列
- **活動名稱 (event_name)**: 請直接讀取圖片上的主標題文字，保留繁體中文原文（例如「微笑多肉市集」「社區樂動展共融」），不要翻譯成英文
- **地點 (venue)**: 尋找「地點」「地址」「場地」等標籤，或任何看起來像地址的文字
- **日期 (date)**: 正規化為 YYYY-MM-DD 格式
- **時間 (time)**: 正規化為 24小時制 HH:MM 格式（例如 12NN-6PM → "12:00-18:00"）
- 如果某欄位在圖片中找不到，使用空字串 ""
- 保留所有中文原文，不要翻譯"""

VISION_USER_PROMPT = "請從這張繁體中文活動海報圖片中，擷取活動名稱、日期、時間、地點。直接讀取圖片上的文字，保留中文原文。"

# ---------------------------------------------------------------------------
# 3A. Security & Configuration — Local API Key Management
# ---------------------------------------------------------------------------


def get_config_path() -> str:
    """
    Return absolute path to config.json.
    For PyInstaller: uses POSTER2DATA_CONFIG_DIR env var or executable's directory.
    For development: uses script's directory.
    """
    # Check for environment variable first (set by PyInstaller launcher)
    config_dir = os.environ.get('POSTER2DATA_CONFIG_DIR')
    if config_dir:
        return os.path.join(config_dir, CONFIG_FILENAME)
    
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, CONFIG_FILENAME)


def load_api_key(key_name: str = CONFIG_KEY) -> str | None:
    """
    Load API key from config.json.
    Returns the key string if found and non-empty, else None.
    """
    path = get_config_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = (data or {}).get(key_name)
        return (key or "").strip() or None
    except (json.JSONDecodeError, OSError):
        return None


def load_openrouter_key() -> str | None:
    """Load OpenRouter API key from config.json."""
    return load_api_key(CONFIG_KEY_OPENROUTER)


def save_api_key(api_key: str, key_name: str = CONFIG_KEY) -> bool:
    """
    Write API key to config.json. Creates file if missing.
    Pass empty string to clear the key. Returns True on success, False on error.
    """
    path = get_config_path()
    try:
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
            except (json.JSONDecodeError, OSError):
                pass
        data[key_name] = (api_key or "").strip()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError:
        return False


def save_openrouter_key(api_key: str) -> bool:
    """Save OpenRouter API key to config.json."""
    return save_api_key(api_key, CONFIG_KEY_OPENROUTER)


def api_key_is_set() -> bool:
    """True if a non-empty DeepSeek API key exists in config.json."""
    return load_api_key() is not None


def openrouter_key_is_set() -> bool:
    """True if a non-empty OpenRouter API key exists in config.json."""
    return load_openrouter_key() is not None


# ---------------------------------------------------------------------------
# Text Extraction — PDF (OCR first, then fallback) / Image OCR
# ---------------------------------------------------------------------------


def _preprocess_image_for_ocr(img: Image.Image, aggressive: bool = False) -> Image.Image:
    """
    Preprocess image for better OCR accuracy:
    - Convert to grayscale
    - Increase contrast
    - Optionally apply adaptive thresholding (binarization)
    - Resize if too small
    
    Args:
        img: PIL Image to preprocess
        aggressive: If True, apply binarization (better for text-heavy docs, worse for posters)
    """
    # Convert to RGB first if needed (handles RGBA, P mode, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Convert to grayscale
    if img.mode == 'RGB':
        img = img.convert('L')
    
    # Resize if image is too small (minimum 1500px width for good OCR)
    min_width = 1500
    if img.width < min_width:
        ratio = min_width / img.width
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Increase contrast using histogram stretching
    img_array = np.array(img)
    p_low, p_high = np.percentile(img_array, (2, 98))
    if p_high > p_low:
        img_array = np.clip((img_array - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Only apply binarization in aggressive mode (can hurt poster OCR)
    if aggressive:
        # Otsu-like threshold using mean
        threshold = int(np.mean(img_array))
        img = img.point(lambda x: 255 if x > threshold else 0, mode='L')
    
    return img


def _ocr_single_image(img: Image.Image, preprocess: bool = True) -> str:
    """
    Run OCR on a single PIL Image with optional preprocessing.
    Tries multiple strategies to get the best result.
    Returns extracted text.
    """
    results = []
    
    # Strategy 1: Light preprocessing (contrast enhancement, no binarization)
    # Better for colorful posters
    if preprocess:
        try:
            img_light = _preprocess_image_for_ocr(img.copy(), aggressive=False)
        except Exception:
            img_light = img.copy()
    else:
        img_light = img.copy()
    
    # Try with Chinese + English first
    try:
        text = pytesseract.image_to_string(img_light, lang=OCR_LANG, config=TESSERACT_CONFIG)
        if text and text.strip():
            results.append(text.strip())
    except Exception:
        pass
    
    # Strategy 2: Aggressive preprocessing (with binarization)
    # Better for text-heavy documents
    if preprocess:
        try:
            img_aggressive = _preprocess_image_for_ocr(img.copy(), aggressive=True)
            text = pytesseract.image_to_string(img_aggressive, lang=OCR_LANG, config=TESSERACT_CONFIG)
            if text and text.strip():
                results.append(text.strip())
        except Exception:
            pass
    
    # Strategy 3: Original image without preprocessing
    # Sometimes works better for high-quality scans
    try:
        text = pytesseract.image_to_string(img, lang=OCR_LANG, config=TESSERACT_CONFIG)
        if text and text.strip():
            results.append(text.strip())
    except Exception:
        pass
    
    # Fallback to English only if no Chinese results
    if not results:
        try:
            text = pytesseract.image_to_string(img_light, lang="eng", config=TESSERACT_CONFIG)
            if text and text.strip():
                results.append(text.strip())
        except Exception:
            pass
    
    if not results:
        return ""
    
    # Return the longest result (usually the most complete extraction)
    return max(results, key=len)


def _pdf_text_via_ocr(raw: bytes) -> tuple[str | None, str]:
    """
    Extract text from PDF using OCR. Tries pdf2image first (needs poppler), then PyMuPDF (no poppler).
    Returns (text or None, error_message for debug).
    """
    if not _TESSERACT_AVAILABLE:
        return None, "pytesseract/PIL not installed"

    def ocr_image(img) -> str:
        """Run OCR with preprocessing and Chinese+English support."""
        return _ocr_single_image(img, preprocess=True)

    # 1) pdf2image (requires poppler on system)
    err = ""
    if _PDF2IMAGE_AVAILABLE:
        try:
            images = convert_from_bytes(raw)
            parts = []
            for img in images:
                text = ocr_image(img)
                if text and text.strip():
                    parts.append(text.strip())
            out = "\n".join(parts).strip() or None
            if out:
                return out, ""
            err = "pdf2image: no text from OCR"
        except Exception as e:
            err = f"pdf2image: {type(e).__name__}: {e}"
    else:
        err = "pdf2image not installed"

    # 2) PyMuPDF (no poppler needed) — render each page to image then OCR
    if _PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
            parts = []
            for page in doc:
                # Use higher DPI for better OCR accuracy
                pix = page.get_pixmap(dpi=OCR_DPI)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                text = ocr_image(img)
                if text and text.strip():
                    parts.append(text.strip())
            doc.close()
            out = "\n".join(parts).strip() or None
            return out, (err + "; " if err else "") + ("" if out else "PyMuPDF OCR: no text")
        except Exception as e:
            return None, (err + "; " if err else "") + f"PyMuPDF: {type(e).__name__}: {e}"

    return None, err or "PyMuPDF not installed"


def _image_text_via_ocr(raw: bytes) -> tuple[str | None, str]:
    """
    Extract text from image file (JPG, PNG) using OCR.
    Returns (text or None, error_message for debug).
    """
    if not _TESSERACT_AVAILABLE:
        return None, "pytesseract/PIL not installed"
    
    try:
        img = Image.open(io.BytesIO(raw))
        text = _ocr_single_image(img, preprocess=True)
        if text and text.strip():
            return text.strip(), ""
        return None, "OCR returned no text from image"
    except Exception as e:
        return None, f"Image OCR error: {type(e).__name__}: {e}"


def _pdf_text_via_plumber(raw: bytes) -> str:
    """Extract text from PDF using pdfplumber (direct text layer)."""
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return "\n".join(parts).strip() or "(No text extracted from PDF)"
    except Exception as e:
        return f"(PDF extraction failed: {e})"


def extract_text_from_file(uploaded_file, extraction_debug: dict | None = None) -> str:
    """
    Extract text from uploaded file.
    PDF: try OCR (pdf2image then PyMuPDF), then pdfplumber.
    Image: use OCR directly.
    Set extraction_debug["ocr_error"] if provided.
    """
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".pdf"):
        ocr_text, ocr_err = _pdf_text_via_ocr(raw)
        if extraction_debug is not None:
            extraction_debug["ocr_error"] = ocr_err or ""
            extraction_debug["ocr_available"] = _TESSERACT_AVAILABLE and (_PDF2IMAGE_AVAILABLE or _PYMUPDF_AVAILABLE)
        if ocr_text and len(ocr_text.strip()) > 20:
            return ocr_text
        # Fallback to pdfplumber for PDFs with embedded text layer
        plumber_text = _pdf_text_via_plumber(raw)
        if extraction_debug is not None and not ocr_text:
            extraction_debug["ocr_error"] = (ocr_err or "") + " | Fallback to pdfplumber"
        return plumber_text

    if name.endswith((".jpg", ".jpeg", ".png")):
        ocr_text, ocr_err = _image_text_via_ocr(raw)
        if extraction_debug is not None:
            extraction_debug["ocr_error"] = ocr_err or ""
            extraction_debug["ocr_available"] = _TESSERACT_AVAILABLE
        if ocr_text and ocr_text.strip():
            return ocr_text
        return "(No text extracted from image via OCR)"

    return "(Unsupported file type)"


# ---------------------------------------------------------------------------
# DeepSeek API & Processing
# ---------------------------------------------------------------------------


def call_deepseek(api_key: str, user_text: str) -> str:
    """Call DeepSeek API; returns raw response content or raises."""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def _image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _get_images_from_file(uploaded_file) -> list[Image.Image]:
    """
    Extract images from uploaded file.
    For PDF: render each page as image.
    For image files: return the image directly.
    Returns list of PIL Images.
    """
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    
    images = []
    
    if name.endswith(".pdf"):
        # Try pdf2image first
        if _PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_bytes(raw, dpi=OCR_DPI)
                if images:
                    return images
            except Exception:
                pass
        
        # Fallback to PyMuPDF
        if _PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=raw, filetype="pdf")
                for page in doc:
                    pix = page.get_pixmap(dpi=OCR_DPI)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    images.append(img)
                doc.close()
                return images
            except Exception:
                pass
    
    elif name.endswith((".jpg", ".jpeg", ".png")):
        try:
            img = Image.open(io.BytesIO(raw))
            return [img]
        except Exception:
            pass
    
    return images


def call_vision_api(api_key: str, images: list[Image.Image]) -> str:
    """
    Call Vision API (via OpenRouter) with images.
    Sends images as base64-encoded content for direct visual understanding.
    Returns raw response content or raises.
    """
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    
    # Build message content with images
    content = []
    
    # Add text prompt first (some models prefer this order)
    content.append({
        "type": "text",
        "text": VISION_USER_PROMPT
    })
    
    # Add all images
    for img in images:
        base64_img = _image_to_base64(img, format="PNG")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}"
            }
        })
    
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json_from_response(raw: str) -> list[dict]:
    """
    Parse JSON from API response. Tolerates markdown code fences.
    Returns list of event dicts (keys: event_name, date, time, venue).
    """
    raw = raw.strip()
    # Strip optional markdown code block
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _row_from_event(event: dict, source_file: str) -> dict:
    """Map one event dict to strict schema row. Use 'Error' for invalid/missing."""
    def get(key: str, default: str = "Error") -> str:
        v = event.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return str(v).strip()

    return {
        "Source File": source_file,
        "Event Name": get("event_name"),
        "Date": get("date"),
        "Time": get("time"),
        "Venue": get("venue"),
    }


def process_file(uploaded_file, deepseek_key: str | None, openrouter_key: str | None = None, debug_out: list | None = None):
    """
    Process file using Vision API (preferred, via OpenRouter) or OCR + text API (via DeepSeek).
    Returns (DataFrame, debug_dict).
    debug_out: if provided, append one dict with extraction/API/parse info for F12 console.
    """
    source_name = uploaded_file.name or "unknown"
    
    # Determine which method to use based on available keys
    use_vision = USE_VISION_API and openrouter_key and openrouter_key.strip()
    
    debug = {
        "file": source_name,
        "stage": "extract",
        "text_len": 0,
        "text_preview": "",
        "ocr_error": "",
        "ocr_available": False,
        "api_ok": False,
        "api_error": None,
        "response_preview": "",
        "parse_ok": False,
        "events_count": 0,
        "method": "vision" if use_vision else "ocr",
        "model": VISION_MODEL if use_vision else DEEPSEEK_MODEL,
    }

    # Check if we have the necessary API key
    if use_vision and not openrouter_key:
        debug["stage"] = "api"
        debug["api_error"] = "No OpenRouter API key for Vision API"
        if debug_out is not None:
            debug_out.append(debug)
        return (
            pd.DataFrame(
                [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                columns=OUTPUT_COLUMNS,
            ),
            debug,
        )
    
    if not use_vision and not (deepseek_key and deepseek_key.strip()):
        debug["stage"] = "api"
        debug["api_error"] = "No DeepSeek API key in config"
        if debug_out is not None:
            debug_out.append(debug)
        return (
            pd.DataFrame(
                [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                columns=OUTPUT_COLUMNS,
            ),
            debug,
        )

    # =====================================================
    # METHOD 1: Vision API (preferred - directly reads images via OpenRouter)
    # =====================================================
    if use_vision:
        try:
            images = _get_images_from_file(uploaded_file)
            if not images:
                debug["stage"] = "extract"
                debug["api_error"] = "Could not extract images from file"
                if debug_out is not None:
                    debug_out.append(debug)
                return (
                    pd.DataFrame(
                        [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                        columns=OUTPUT_COLUMNS,
                    ),
                    debug,
                )
            
            debug["text_preview"] = f"[Vision API] {len(images)} image(s) → {VISION_MODEL}"
            debug["text_len"] = len(images)
            
            raw_response = call_vision_api(openrouter_key, images)
            debug["api_ok"] = True
            debug["response_preview"] = (raw_response[:400] + "…") if len(raw_response) > 400 else raw_response
            
        except Exception as e:
            debug["stage"] = "api"
            debug["api_error"] = f"Vision API: {type(e).__name__}: {str(e)}"
            # Fallback to OCR method if DeepSeek key is available
            if deepseek_key and deepseek_key.strip():
                debug["method"] = "ocr_fallback"
                uploaded_file.seek(0)
                return _process_file_ocr(uploaded_file, deepseek_key, debug, debug_out)
            if debug_out is not None:
                debug_out.append(debug)
            return (
                pd.DataFrame(
                    [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                    columns=OUTPUT_COLUMNS,
                ),
                debug,
            )
    
    # =====================================================
    # METHOD 2: OCR + Text API (fallback via DeepSeek)
    # =====================================================
    else:
        return _process_file_ocr(uploaded_file, deepseek_key, debug, debug_out)

    # Parse response (common for both methods)
    events = _parse_json_from_response(raw_response)
    debug["parse_ok"] = len(events) > 0
    debug["events_count"] = len(events)

    if not events:
        debug["stage"] = "parse"
        debug["api_error"] = "JSON parse failed or empty events"
        if debug_out is not None:
            debug_out.append(debug)
        return (
            pd.DataFrame(
                [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                columns=OUTPUT_COLUMNS,
            ),
            debug,
        )

    debug["stage"] = "ok"
    rows = [_row_from_event(ev, source_name) for ev in events]
    if debug_out is not None:
        debug_out.append(debug)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), debug


def _process_file_ocr(uploaded_file, api_key: str, debug: dict, debug_out: list | None):
    """
    Process file using OCR + text API (legacy method).
    """
    source_name = uploaded_file.name or "unknown"
    extraction_debug = {}
    text = extract_text_from_file(uploaded_file, extraction_debug=extraction_debug)
    debug["text_len"] = len(text)
    debug["text_preview"] = (text[:500] + "…") if len(text) > 500 else text
    debug["ocr_error"] = extraction_debug.get("ocr_error", "")
    debug["ocr_available"] = extraction_debug.get("ocr_available", False)

    if "(Unsupported file type)" in text:
        debug["stage"] = "skip"
        debug["api_error"] = "Unsupported file type"
        if debug_out is not None:
            debug_out.append(debug)
        return (
            pd.DataFrame(
                [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                columns=OUTPUT_COLUMNS,
            ),
            debug,
        )

    try:
        user_message = USER_TEXT_PREFIX + text
        raw_response = call_deepseek(api_key, user_message)
        debug["api_ok"] = True
        debug["response_preview"] = (raw_response[:400] + "…") if len(raw_response) > 400 else raw_response
    except Exception as e:
        debug["stage"] = "api"
        debug["api_error"] = f"{type(e).__name__}: {str(e)}"
        if debug_out is not None:
            debug_out.append(debug)
        return (
            pd.DataFrame(
                [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                columns=OUTPUT_COLUMNS,
            ),
            debug,
        )

    events = _parse_json_from_response(raw_response)
    debug["parse_ok"] = len(events) > 0
    debug["events_count"] = len(events)

    if not events:
        debug["stage"] = "parse"
        if (debug.get("text_preview") or "").strip() == "(No text extracted from PDF)":
            debug["api_error"] = "No text from PDF. " + (debug.get("ocr_error") or "Install tesseract (and poppler for pdf2image, or use PyMuPDF).")
        else:
            debug["api_error"] = "JSON parse failed or empty events"
        if debug_out is not None:
            debug_out.append(debug)
        return (
            pd.DataFrame(
                [{c: (source_name if c == "Source File" else "Error") for c in OUTPUT_COLUMNS}],
                columns=OUTPUT_COLUMNS,
            ),
            debug,
        )

    debug["stage"] = "ok"
    rows = [_row_from_event(ev, source_name) for ev in events]
    if debug_out is not None:
        debug_out.append(debug)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), debug


def get_empty_dataframe() -> pd.DataFrame:
    """Return DataFrame with strict columns only, no rows."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Custom CSS — Enterprise SaaS Look
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* Hide Streamlit default chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Tighten spacing: less gap between header and main content */
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 1rem !important;
    }

    /* Root and layout */
    .stApp { background: #f5f6f8; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8eaef 0%, #dfe2e8 100%);
        min-width: 280px;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #1a1d21; }

    /* Style native file uploader as single Drag & Drop card */
    [data-testid="stFileUploader"] {
        border: 2px dashed #c5c9d1 !important;
        border-radius: 12px !important;
        padding: 24px 20px !important;
        text-align: center !important;
        background: #fafbfc !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stFileUploader"] section div[data-testid="stFileUploadDropzone"] {
        padding: 16px 12px !important;
        text-align: center !important;
    }
    [data-testid="stFileUploader"] small {
        display: block;
        margin-top: 8px;
        color: #5a5e66;
        font-size: 0.85rem;
    }

    /* Primary accent — Futu Blue */
    .primary-accent { color: #2A55E5; }
    .primary-bg { background-color: #2A55E5; }
    .primary-bg:hover { background-color: #2349c4; }

    /* Header bar */
    .p2d-logo {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1d21;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .p2d-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .p2d-badge-online {
        background: #d4edda;
        color: #155724;
    }
    .p2d-badge-offline {
        background: #f8d7da;
        color: #721c24;
    }
    .p2d-badge-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .p2d-badge-online .p2d-badge-dot { background: #28a745; }
    .p2d-badge-offline .p2d-badge-dot { background: #dc3545; }
    .p2d-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #e2e5e9;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6c757d;
        font-weight: 600;
    }

    /* Left column sections */
    .p2d-sidebar-title { font-weight: 700; font-size: 1rem; color: #1a1d21; margin-bottom: 6px; }
    .p2d-upload-section { margin-bottom: 8px; }

    /* Recent Activity — small row: icon | text | status dot */
    .p2d-activity-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid #e8eaef;
        font-size: 0.875rem;
    }
    .p2d-activity-row:last-child { border-bottom: none; }
    .p2d-activity-icon { flex-shrink: 0; }
    .p2d-activity-text { flex: 1; min-width: 0; color: #1a1d21; }
    .p2d-activity-meta { font-size: 0.75rem; color: #6c757d; margin-top: 2px; }
    .p2d-activity-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .p2d-activity-dot--done { background: #28a745; }
    .p2d-activity-dot--scanning { background: #ffc107; }
    .p2d-activity-dot--failed { background: #dc3545; }

    /* Main content */
    .p2d-main-title { font-size: 1.35rem; font-weight: 700; color: #1a1d21; margin-bottom: 4px; }
    .p2d-main-subtitle { color: #6c757d; font-size: 0.9rem; margin-bottom: 16px; }
</style>
"""


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------


def init_session_state():
    if "extracted_df" not in st.session_state:
        st.session_state.extracted_df = get_empty_dataframe()
    if "recent_activity" not in st.session_state:
        st.session_state.recent_activity = []
    if "last_processed_batch" not in st.session_state:
        st.session_state.last_processed_batch = []
    if "uploader_reset_key" not in st.session_state:
        st.session_state.uploader_reset_key = 0
    if "last_debug" not in st.session_state:
        st.session_state.last_debug = []


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Poster2Data AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    deepseek_key = load_api_key()
    openrouter_key = load_openrouter_key()
    deepseek_key_set = api_key_is_set()
    openrouter_key_set = openrouter_key_is_set()

    # ----- Configuration in Streamlit Sidebar -----
    with st.sidebar:
        st.markdown("**Configuration**")
        
        # Vision API (OpenRouter) - Primary for better accuracy
        if USE_VISION_API:
            st.markdown("##### 🖼️ Vision API (推薦)")
            if openrouter_key_set:
                st.success(f"✅ Vision AI Online ({VISION_MODEL.split('/')[-1]})")
                if st.button("Clear Vision key", use_container_width=True, key="clear_openrouter"):
                    save_openrouter_key("")
                    st.rerun()
            else:
                st.info("💡 Vision API 可直接讀取圖片，準確度更高")
                st.caption("從 [openrouter.ai](https://openrouter.ai/keys) 免費取得 API Key")
                openrouter_input = st.text_input(
                    "OpenRouter API Key",
                    type="password",
                    placeholder="sk-or-v1-...",
                    key="openrouter_key_input",
                    label_visibility="collapsed",
                )
                if st.button("Save Vision Key", type="primary", use_container_width=True, key="save_openrouter"):
                    if openrouter_input and openrouter_input.strip():
                        if save_openrouter_key(openrouter_input):
                            st.success("Key saved. Refreshing...")
                            st.rerun()
                        else:
                            st.error("Could not write config.json.")
                    else:
                        st.error("Please enter a non-empty key.")
        
        st.divider()
        
        # DeepSeek API - Fallback / OCR mode
        st.markdown("##### 📝 DeepSeek API (OCR 備用)")
        if deepseek_key_set:
            st.success("✅ DeepSeek Online")
            if st.button("Clear DeepSeek key", use_container_width=True, key="clear_deepseek"):
                save_api_key("")
                st.rerun()
        else:
            st.warning("⚠️ 需要 DeepSeek API Key 作為備用")
            deepseek_input = st.text_input(
                "DeepSeek API Key",
                type="password",
                placeholder="sk-...",
                key="deepseek_key_input",
                label_visibility="collapsed",
            )
            if st.button("Save DeepSeek Key", use_container_width=True, key="save_deepseek"):
                if deepseek_input and deepseek_input.strip():
                    if save_api_key(deepseek_input):
                        st.success("Key saved. Refreshing...")
                        st.rerun()
                    else:
                        st.error("Could not write config.json.")
                else:
                    st.error("Please enter a non-empty key.")

    # ----- Header: Logo | Badge | Export | Avatar -----
    # System is operational if we have at least one working key
    system_operational = (USE_VISION_API and openrouter_key_set) or deepseek_key_set
    badge_class = "p2d-badge-online" if system_operational else "p2d-badge-offline"
    badge_text = "System Operational" if system_operational else "API Key Required"
    h1, h2, h3 = st.columns([3, 1, 1])
    with h1:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px;">
                <span class="p2d-logo">📊 Poster2Data AI</span>
                <span class="p2d-badge {badge_class}">
                    <span class="p2d-badge-dot"></span>
                    {badge_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h2:
        csv_bytes = st.session_state.extracted_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export to CSV",
            data=csv_bytes,
            file_name="poster2data_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with h3:
        st.markdown('<div class="p2d-avatar">U</div>', unsafe_allow_html=True)

    # ----- Layout: 30% Left (Upload + Recent Activity) / 70% Right (Data Preview) -----
    col_left, col_main = st.columns([3, 7])

    with col_left:
        st.markdown('<p class="p2d-sidebar-title">Upload Sources</p>', unsafe_allow_html=True)
        uploader_key = f"main_uploader_{st.session_state.get('uploader_reset_key', 0)}"
        uploaded_files = st.file_uploader(
            "Drag & Drop posters here — JPG, PNG, PDF up to 25MB",
            type=["jpg", "jpeg", "png", "pdf"],
            label_visibility="collapsed",
            key=uploader_key,
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.session_state.last_processed_batch = []

        current_batch = [(f.name, f.size) for f in uploaded_files] if uploaded_files else []
        last_batch = st.session_state.get("last_processed_batch", [])
        new_files = [f for f in (uploaded_files or []) if (f.name, f.size) not in last_batch]

        if new_files:
            use_vision = USE_VISION_API and openrouter_key_set
            processing_msg = f"🖼️ Vision AI Processing ({VISION_MODEL.split('/')[-1]})..." if use_vision else "🔄 OCR Scanning & AI Processing..."
            st.info(processing_msg)
            debug_list = []
            with st.spinner("Extracting data from file(s)..."):
                for f in new_files:
                    df, _ = process_file(f, deepseek_key, openrouter_key, debug_out=debug_list)
                    existing = st.session_state.extracted_df
                    if len(df) > 0:
                        st.session_state.extracted_df = pd.concat(
                            [existing, df], ignore_index=True
                        )
                    size_mb = f"{f.size / (1024 * 1024):.1f} MB"
                    entry = {"name": f.name, "status": "done", "size": size_mb}
                    if not any(a.get("name") == f.name for a in st.session_state.recent_activity):
                        st.session_state.recent_activity = [entry] + st.session_state.recent_activity[:9]
                st.session_state.last_processed_batch = current_batch
                st.session_state.last_debug = debug_list
                # Clear upload box next run by using a new key
                st.session_state.uploader_reset_key = st.session_state.get("uploader_reset_key", 0) + 1
                # Also print to terminal (where streamlit run is executed)
                for d in debug_list:
                    print("[Poster2Data Debug]", json.dumps(d, ensure_ascii=False, indent=2))
            st.success("✅ Done")
            st.rerun()

        # Recent Activity — scrollable container
        st.markdown('<p class="p2d-sidebar-title" style="margin-top:12px;">RECENT ACTIVITY</p>', unsafe_allow_html=True)
        activity_rows = []
        for item in st.session_state.recent_activity[:10]:  # Show up to 10 items
            status = item.get("status", "")
            name = item.get("name", "File")
            icon = "📄" if name.lower().endswith(".pdf") else "🖼️"
            if status == "done":
                meta = f"Processed • {item.get('size', '')}"
                dot_class = "p2d-activity-dot--done"
            elif status == "scanning":
                meta = f"OCR Scanning... {item.get('pct', 0)}%"
                dot_class = "p2d-activity-dot--scanning"
            elif status == "failed":
                meta = "Upload Failed"
                dot_class = "p2d-activity-dot--failed"
            else:
                meta = ""
                dot_class = "p2d-activity-dot--done"
            activity_rows.append(
                f'<div class="p2d-activity-row">'
                f'<span class="p2d-activity-icon">{icon}</span>'
                f'<div class="p2d-activity-text"><span>{html.escape(name)}</span><div class="p2d-activity-meta">{html.escape(meta)}</div></div>'
                f'<span class="p2d-activity-dot {dot_class}" title="{html.escape(status)}"></span></div>'
            )
        # Scrollable container with max height
        st.markdown(
            f'<div style="max-height: 250px; overflow-y: auto; padding-right: 5px;">{"".join(activity_rows)}</div>',
            unsafe_allow_html=True
        )

    with col_main:
        st.markdown('<p class="p2d-main-title">Extracted Data Preview</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="p2d-main-subtitle">Review and edit the structured data extracted from your files.</p>',
            unsafe_allow_html=True,
        )

        df = st.session_state.extracted_df
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="data_editor",
        )
        st.session_state.extracted_df = edited

        n = len(edited)
        if n == 0:
            st.caption("No data yet. Upload a PDF to extract event details.")
        else:
            st.caption(f"Showing {n} item(s)" if n != 1 else "Showing 1 item")

        # Debug: show last run info and mirror to F12 console
        last_debug = st.session_state.get("last_debug", [])
        if last_debug:
            with st.expander("🔧 Debug (F12 Console)", expanded=False):
                for i, d in enumerate(last_debug):
                    method = d.get("method", "ocr")
                    model = d.get("model", "")
                    method_label = "🖼️ Vision API" if method == "vision" else ("🔄 OCR (fallback)" if method == "ocr_fallback" else "📝 OCR")
                    st.markdown(f"**File:** `{d.get('file', '')}` · **Method:** {method_label}")
                    if model:
                        st.markdown(f"- **Model:** `{model}`")
                    st.markdown(f"- **Stage:** `{d.get('stage', '')}` · **Text/Images:** {d.get('text_len', 0)}")
                    if method != "vision" and (d.get("ocr_error") or d.get("ocr_available") is not None):
                        st.markdown(f"- **OCR available:** {d.get('ocr_available', False)} · **OCR error:** `{d.get('ocr_error', '')}`")
                    if d.get("text_preview"):
                        label = "Vision API input" if method == "vision" else "Text preview (sent to LLM)"
                        st.text_area(label, d.get("text_preview", ""), height=80, key=f"debug_text_{i}")
                    if d.get("api_error"):
                        st.error(f"API/Parse: {d.get('api_error')}")
                    if d.get("api_ok") and d.get("response_preview"):
                        st.text_area("API response preview", d.get("response_preview", ""), height=100, key=f"debug_resp_{i}")
                    st.markdown(f"- **Parse OK:** {d.get('parse_ok')} · **Events count:** {d.get('events_count', 0)}")
                    st.divider()
                # Print same debug to F12 Console (base64 to avoid escaping issues)
                try:
                    payload_b64 = base64.b64encode(
                        json.dumps(last_debug, ensure_ascii=False).encode("utf-8")
                    ).decode("ascii")
                    st.components.v1.html(
                        f'<script>'
                        f'try {{ var j = JSON.parse(atob("{payload_b64}")); '
                        f'window.top.console.log("[Poster2Data Debug]", j); }} '
                        f'catch(e) {{ window.top.console.error("[Poster2Data Debug]", e); }}'
                        f'</script>',
                        height=0,
                    )
                except Exception as ex:
                    st.caption(f"Console log skipped: {ex}. See expander above.")


if __name__ == "__main__":
    main()
