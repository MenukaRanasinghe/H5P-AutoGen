import math
import os
import re
import json
import copy
import glob
# Image sourcing: Wikimedia Commons (no API key required)

import shutil
import zipfile
import tempfile
import uuid
import time
import random
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from pypdf import PdfReader


# ----------------------------
# Free image sourcing (Wikimedia Commons)
# ----------------------------
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = os.environ.get("H5P_IMG_USER_AGENT", "H5PActivityGenerator/1.0 (contact: content@imperiallearning.co.uk)")
LLM_API_KEY = os.getenv("LLM_API_KEY")
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY")

# Pull keys from Streamlit secrets (Cloud + local secrets.toml)
if "LLM_API_KEY" in st.secrets and not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = st.secrets["LLM_API_KEY"]

if "FREEPIK_API_KEY" in st.secrets and not os.environ.get("FREEPIK_API_KEY"):
    os.environ["FREEPIK_API_KEY"] = st.secrets["FREEPIK_API_KEY"]

# Re-read keys after Streamlit secrets injection (globals above were read before os.environ was updated)
LLM_API_KEY = os.getenv("LLM_API_KEY") or LLM_API_KEY
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY") or FREEPIK_API_KEY

# Freepik API configuration
FREEPIK_API_BASE = os.getenv("FREEPIK_API_BASE", "https://api.freepik.com/v1").rstrip("/")
FREEPIK_LANG = os.getenv("FREEPIK_LANG", "en-US")
FREEPIK_DEFAULT_IMAGE_SIZE = os.getenv("FREEPIK_IMAGE_SIZE", "large")  # small|medium|large|original or px string

# Optional AI image generation (OpenAI Images API)
# Set USE_AI_IMAGES=1 to enable (may incur API costs). Falls back to Freepik/Wikimedia on failure.
USE_AI_IMAGES = os.getenv("USE_AI_IMAGES", "1")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")

# OpenAI TTS configuration (used for Dictation activities)
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_TTS_NORMAL_SPEED = float(os.getenv("OPENAI_TTS_NORMAL_SPEED", "1.0"))
OPENAI_TTS_SLOW_SPEED = float(os.getenv("OPENAI_TTS_SLOW_SPEED", "0.75"))

# Filter out files that are likely to contain text overlays, logos, icons, diagrams, banners, etc.
_BAD_TITLE_TERMS = {
    "logo","icon","diagram","chart","word","text","banner","label","seal","flag","coat","crest",
    "poster","sign","notice","warning","infographic","map","coat of arms","emblem",
    "svg","clipart","pictogram","symbol","typography","font",
}

_STOPWORDS = {
    "a","an","the","and","or","of","to","in","on","for","with","without","by","from","at","as",
    "is","are","was","were","be","been","being",
    "this","that","these","those",
    "into","over","under","between","within","during","after","before","about",
    "your","their","our","its","it's","his","her","them","they","we","you",
}

def _is_bad_filename(name: str) -> bool:
    n = (name or "").lower()
    return any(t in n for t in _BAD_TITLE_TERMS)

def _terms(s: str) -> List[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-z][a-z\-]{2,}", s)
    toks = [t for t in toks if t not in _STOPWORDS]
    # de-duplicate while preserving order
    out = []
    for t in toks:
        if t not in out:
            out.append(t)
    return out

def _title_score(title: str, query_terms: List[str]) -> int:
    tl = (title or "").lower()
    score = 0
    for t in query_terms:
        if t in tl:
            score += 6
    # prefer photographic terms (rough heuristic)
    if any(x in tl for x in ["photo","photograph","jpg","jpeg","png"]):
        score += 1
    return score

def wikimedia_find_image_url(query: str, limit: int = 25) -> Optional[Dict[str, Any]]:
    """Return dict with url/mime/size/title; or None.
    Chooses the most relevant candidate by title-term overlap (not random).
    """
    q = (query or "").strip()
    if not q:
        return None

    # A safer, broader query. Wikimedia search supports basic operators.
    # Exclude obvious non-photo/graphics terms to reduce text-y images.
    neg = " -logo -icon -diagram -chart -banner -poster -sign -infographic -svg"
    gsr = f"{q}{neg}"

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": gsr,
        "gsrnamespace": 6,  # File:
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": 1600,
    }
    try:
        r = requests.get(WIKIMEDIA_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    pages = ((data or {}).get("query") or {}).get("pages") or {}
    candidates = []
    q_terms = _terms(q)

    for _, p in pages.items():
        title = p.get("title") or ""
        if _is_bad_filename(title):
            continue
        infos = p.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = (info.get("mime") or "").lower()
        if not mime.startswith("image/") or "svg" in mime:
            continue
        w = int(info.get("width") or 0)
        h = int(info.get("height") or 0)
        url = info.get("url")
        if not url or w < 450 or h < 300:
            continue
        score = _title_score(title, q_terms)
        candidates.append({"url": url, "mime": mime, "title": title, "score": score, "w": w, "h": h})

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    return {"url": best["url"], "mime": best["mime"], "title": best["title"]}


def _freepik_headers() -> Dict[str, str]:
    """
    Freepik expects the API key in `x-freepik-api-key` and optionally an Accept-Language header.
    (Language defaults to "en-US" if you don't send one.)
    """
    h = {"User-Agent": USER_AGENT, "Accept-Language": FREEPIK_LANG}
    if FREEPIK_API_KEY:
        h["x-freepik-api-key"] = FREEPIK_API_KEY
    return h


def _freepik_build_params(
    *,
    term: str,
    page: int,
    limit: int,
    order: str,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Any]]:
    """
    Freepik docs define `filters` as a query-object. In practice, APIs typically accept this as:
      filters[content_type][]=photo&filters[license][]=freemium ...
    We'll encode it this way (and fall back to JSON-string encoding if the API rejects it).
    """
    params: List[Tuple[str, Any]] = [("term", term), ("page", page), ("limit", limit), ("order", order)]

    if not filters:
        return params

    for key, val in filters.items():
        if val is None:
            continue

        # allow either list[str] (recommended) or dict[str,bool] (from some wrappers)
        if isinstance(val, dict):
            selected = [k for k, enabled in val.items() if enabled]
            for v in selected:
                params.append((f"filters[{key}][]", v))
            continue

        if isinstance(val, (list, tuple, set)):
            for v in val:
                params.append((f"filters[{key}][]", v))
            continue

        params.append((f"filters[{key}]", val))

    return params


def _tokenise(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def freepik_find_image_url(query: str, limit: int = 25, prefer_vectors: bool = False) -> Optional[Dict[str, Any]]:
    """
    Search Freepik resources and return a best-match candidate dict:
      {id, title, page_url, preview_url, author_name, license_url, score}

    Uses:
      - GET /v1/resources (search) to get candidates
      - Heuristics (downloads/likes + text match) to approximate "website-like" ranking

    When prefer_vectors=True, vectors are searched first for a cleaner professional look.
    """
    if not FREEPIK_API_KEY:
        return None

    q = (query or "").strip()
    if not q:
        return None

    url = f"{FREEPIK_API_BASE}/resources"

    # Website search generally feels "better" because it can surface popular assets;
    # the public API only offers `relevance` and `recent` ordering, so we pull a bigger pool
    # and do our own lightweight popularity + textual scoring.
    per_page = min(max(25, limit), 80)
    max_pages = 3  # keep it small to avoid rate limits

    # When prefer_vectors is set, search vectors first for cleaner professional imagery.
    if prefer_vectors:
        filter_sets: List[Optional[Dict[str, Any]]] = [
            {"content_type": ["vector"]},
            {"content_type": ["photo"]},
            {"content_type": ["photo", "vector", "psd"]},
        ]
    else:
        # Start by preferring photos (common for course content), then relax if needed.
        filter_sets: List[Optional[Dict[str, Any]]] = [
            {"content_type": ["photo"]},
            {"content_type": ["vector"]},
            {"content_type": ["photo", "vector", "psd"]},
        ]

    q_tokens = set(_tokenise(q))

    def score_item(item: Dict[str, Any]) -> float:
        title = item.get("title") or ""
        title_tokens = set(_tokenise(title))

        # popularity proxies from API list response
        stats = item.get("stats") or {}
        downloads = int(stats.get("downloads") or 0)
        likes = int(stats.get("likes") or 0)

        overlap = len(q_tokens & title_tokens)
        exact_phrase = 2.0 if q.lower() in title.lower() else 0.0

        is_new = bool((item.get("meta") or {}).get("is_new"))
        tl = title.lower()
        # Penalise "childish"/cartoon-ish assets and text-heavy graphic styles.
        bad_terms = ["cartoon","kids","kid","child","children","cute","kawaii","doodle","hand drawn","hand-drawn","comic","coloring","colouring",
                     "sticker","emoji","clipart","icon","logo","typography","text",
                     "baby","toddler","nursery","kindergarten","preschool","childcare","daycare",
                     "toy","toys","teddy","plush","crayon","finger paint"]
        bad = sum(1 for t in bad_terms if t in tl)
        # Boost professional/business imagery
        good_terms = ["professional","business","corporate","office","workplace","training","modern","flat","minimal","clean"]
        good = sum(1 for t in good_terms if t in tl)
        # Prefer vectors when prefer_vectors is set
        content_type = (item.get("type") or "").lower()
        vector_bonus = 4.0 if (prefer_vectors and content_type == "vector") else 0.0
        # final score (tune as you like)
        return (
            overlap * 5.0
            + exact_phrase * 3.0
            + math.log1p(downloads) * 2.0
            + math.log1p(likes) * 1.0
            + (1.0 if is_new else 0.0)
            + good * 3.0
            + vector_bonus
            - (bad * 10.0)
        )

    best: Optional[Dict[str, Any]] = None
    best_score: float = -1e9

    for filters in filter_sets:
        # gather candidates across pages
        items: List[Dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            params = _freepik_build_params(term=q, page=page, limit=per_page, order="relevance", filters=filters)
            try:
                r = requests.get(url, headers=_freepik_headers(), params=params, timeout=20)
                # if the server doesn't like deepObject params, retry with JSON encoding
                if r.status_code in (400, 422) and filters:
                    r = requests.get(
                        url,
                        headers=_freepik_headers(),
                        params={"term": q, "page": page, "limit": per_page, "order": "relevance", "filters": json.dumps(filters)},
                        timeout=20,
                    )
                if r.status_code != 200:
                    continue
                data = r.json() or {}
                items.extend(data.get("data") or [])
                meta = data.get("meta") or {}
                last_page = int(meta.get("last_page") or page)
                if page >= last_page:
                    break
            except Exception:
                continue

        if not items:
            continue

        for item in items:
            if (item.get("image") or {}).get("source", {}).get("url") is None:
                continue
            s = score_item(item)
            if s > best_score:
                best_score = s
                best = item

        if best:
            break

    if not best:
        return None

    img = best.get("image") or {}
    src = (img.get("source") or {}).get("url")
    author = best.get("author") or {}
    licenses = best.get("licenses") or []

    return {
        "id": best.get("id"),
        "title": best.get("title"),
        "page_url": best.get("url"),
        "preview_url": src,
        "author_name": author.get("name"),
        "license_url": (licenses[0].get("url") if licenses else None),
        "score": round(best_score, 2),
    }


def freepik_download_signed_url(resource_id: int, image_size: str = None) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with at least:
      {signed_url, filename, url}

    Uses GET /v1/resources/{resource-id}/download.
    For photos you can request image_size: small|medium|large|original or "1000px".."2000px".
    """
    if not FREEPIK_API_KEY or not resource_id:
        return None

    url = f"{FREEPIK_API_BASE}/resources/{int(resource_id)}/download"
    size = (image_size or FREEPIK_DEFAULT_IMAGE_SIZE or "").strip() or None

    params: Dict[str, Any] = {}
    if size:
        params["image_size"] = size

    try:
        r = requests.get(url, headers=_freepik_headers(), params=params, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json() or {}
        return data.get("data") or None
    except Exception:
        return None



def openai_generate_image_to_path(images_dir: str, query: str, stem: str) -> Optional[Dict[str, Any]]:
    """
    Generate an image using the OpenAI Images API and save into the H5P images folder.

    Controlled by env var USE_AI_IMAGES=1. Always returns a PNG (image/png) on success.
    On any error, returns None and the caller can fall back to Freepik/Wikimedia.
    """
    if str(USE_AI_IMAGES).strip() != "1":
        return None

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return None

    q = (query or "").strip()
    if not q:
        return None

    # Prompt tuned for professional (non-childish), text-free visuals.
    prompt = (
        "Create a modern, professional, clean vector-style illustration suitable for workplace training. "
        "No text, no logos, no watermarks, no brand names. "
        f"Concept: {q}."
    )

    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "size": OPENAI_IMAGE_SIZE,
        "response_format": "b64_json",
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            return None
        data = r.json() or {}
        arr = data.get("data") or []
        if not arr or not isinstance(arr, list):
            return None
        b64 = (arr[0] or {}).get("b64_json")
        if not b64:
            return None
        img_bytes = base64.b64decode(b64)
        os.makedirs(images_dir, exist_ok=True)
        fname = f"{safe_filename(stem)}.png"
        abs_path = os.path.join(images_dir, fname)
        with open(abs_path, "wb") as f:
            f.write(img_bytes)
        return {
            "path": f"images/{fname}",
            "mime": "image/png",
            "credit": {"provider": "openai", "model": OPENAI_IMAGE_MODEL},
        }
    except Exception:
        return None


def openai_tts_to_file(
    text: str,
    out_path: str,
    speed: float = 1.0,
    voice: str = None,
    model: str = None,
) -> bool:
    """Generate speech audio via OpenAI TTS API and save to out_path as MP3.

    Args:
        text:     The text to speak.
        out_path: Absolute path where the .mp3 will be written.
        speed:    Playback speed (0.25–4.0). Use <1.0 for slow.
        voice:    TTS voice name (default from env OPENAI_TTS_VOICE).
        model:    TTS model (default from env OPENAI_TTS_MODEL).

    Returns True on success, False on any failure.
    """
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return False

    text = (text or "").strip()
    if not text:
        return False

    url = "https://api.openai.com/v1/audio/speech"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model or OPENAI_TTS_MODEL,
        "input": text,
        "voice": voice or OPENAI_TTS_VOICE,
        "response_format": "mp3",
        "speed": max(0.25, min(4.0, speed)),
    }

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                sleep_s = min(30.0, (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                if attempt == max_attempts:
                    return False
                time.sleep(sleep_s)
                continue
            if resp.status_code in (500, 502, 503, 504):
                sleep_s = min(20.0, (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                if attempt == max_attempts:
                    return False
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception:
            if attempt == max_attempts:
                return False
            time.sleep(min(10.0, (2 ** (attempt - 1))))
    return False


_PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3G7qkAAAAASUVORK5CYII="
)

def _write_placeholder_png(path: str) -> None:
    import base64
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(_PLACEHOLDER_PNG_B64))

def download_image_to_h5p(images_dir: str, query: str, stem: str, prefer_vectors: bool = False) -> Optional[Dict[str, Any]]:
    """Download an image for query into content/images and return dict with rel path + mime + optional credit.

    Preference order:
      1) Freepik (if FREEPIK_API_KEY is set)
      2) Wikimedia Commons
    """
    os.makedirs(images_dir, exist_ok=True)
    q = (query or "").strip()
    if not q:
        return None

    # --- AI generation (optional) ---
    ai = openai_generate_image_to_path(images_dir, q, stem=f"{stem}_ai")
    if ai:
        return ai

    # --- Freepik (preferred) ---
    if FREEPIK_API_KEY:
        cand = freepik_find_image_url(q, prefer_vectors=prefer_vectors)
        if cand and cand.get("id"):
            dl_meta = freepik_download_signed_url(int(cand["id"]), image_size="large")
            signed = ""
            if isinstance(dl_meta, dict):
                signed = dl_meta.get("signed_url") or dl_meta.get("url") or ""
            if signed:
                try:
                    rr = requests.get(signed, headers={"User-Agent": USER_AGENT}, timeout=60)
                    rr.raise_for_status()
                    mime = (rr.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                    if not mime.startswith("image/"):
                        mime = "image/jpeg" if signed.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    ext = ".jpg" if ("jpeg" in mime or "jpg" in mime) else ".png"
                    fname = f"{safe_filename(stem)}{ext}"
                    abs_path = os.path.join(images_dir, fname)
                    with open(abs_path, "wb") as f:
                        f.write(rr.content)
                    return {
                        "path": f"images/{fname}",
                        "mime": mime,
                        "credit": {
                            "provider": "freepik",
                            "source": cand.get("page_url") or "",
                            "title": cand.get("title") or "",
                            "author": cand.get("author_name") or "",
                            "license_url": cand.get("license_url") or "",
                        },
                    }
                except Exception:
                    pass  # fall through to Wikimedia

    # --- Wikimedia fallback ---
    found = wikimedia_find_image_url(q)
    if not found:
        return None
    url = found["url"]
    mime = found["mime"]
    ext = ".jpg" if ("jpeg" in mime or "jpg" in mime) else ".png"
    fname = f"{safe_filename(stem)}{ext}"
    abs_path = os.path.join(images_dir, fname)
    try:
        rr = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        rr.raise_for_status()
        with open(abs_path, "wb") as f:
            f.write(rr.content)
    except Exception:
        return None
    return {"path": f"images/{fname}", "mime": mime, "credit": {"provider": "wikimedia", "source": url}}

def download_image_to_h5p_multi(images_dir: str, queries: List[str], stem: str, prefer_vectors: bool = False) -> Optional[Dict[str, Any]]:
    """Try multiple queries until an image is found."""
    seen = set()
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        dl = download_image_to_h5p(images_dir, q, stem=f"{stem}_{q[:40]}", prefer_vectors=prefer_vectors)
        if dl:
            return dl
    return None

def ensure_image(images_dir: str, queries: List[str], stem: str, fallback_query: str = "inclusive classroom", prefer_vectors: bool = False) -> Dict[str, Any]:
    """Guarantee an image payload. Uses Freepik if configured, then Wikimedia; if it fails, writes a tiny placeholder PNG."""
    dl = download_image_to_h5p_multi(images_dir, queries + [fallback_query], stem=stem, prefer_vectors=prefer_vectors)
    if dl:
        return dl
    # absolute last resort
    fname = f"{safe_filename(stem)}_placeholder.png"
    abs_path = os.path.join(images_dir, fname)
    _write_placeholder_png(abs_path)
    return {"path": f"images/{fname}", "mime": "image/png", "credit": {"provider": "placeholder"}}

def extract_keywords(text: str, max_terms: int = 4) -> List[str]:
    """Lightweight keyword extraction for better image searches."""
    toks = _terms(text)
    # prefer longer / more specific terms
    toks.sort(key=lambda t: (-len(t), t))
    return toks[:max_terms]





# =========================
# SIMPLE WORKFLOW APP
# =========================
# 1) Upload PDF(s)
# 2) Course name
# 3) Click "Suggest H5P types"
# 4) Choose ONE type
# 5) Click "Generate H5P"
#
# Fixes:
# - Prevents repeated API calls due to Streamlit reruns
# - Caches PDF extraction + suggestions
# - Retries on 429 / transient errors with backoff
# - Shows friendly errors (no traceback)
#
# Requirements:
# - Set env var: LLM_API_KEY
# - Provide H5P templates in ./templates as blank .h5p exports
#   named exactly as the H5P type label, e.g. templates/Interactive Book.h5p
#
# True/False:
# - H5P "True/False Question" is single question. To generate 5+ in ONE file,
#   this app creates a "Quiz (Question Set)" containing multiple True/False items.
#   Therefore you MUST have templates/Quiz (Question Set).h5p
# =========================


# "Best" H5P types to suggest (keeps recommendations high-signal).
# The generator can still build ANY type that exists as a .h5p template in ./templates.
BEST_H5P_TYPES = [
    "Cornell Notes",
    "Course Presentation",
    "Dialog Cards",
    "Dictation",
    "Drag the Words",
    "Essay",
    "Fill in the Blanks",
    "Interactive Book",
    "Mark the Words",
    "Multiple Choice",
    "Page",
    "Quiz",
    "Single Choice",
    "Summary",
]

# Question-count limits for selected activity types
# - For a single PDF upload: 4–8 questions
# - For multiple PDFs:      4–12 questions
LIMITED_Q_TYPES = {
    "Dictation",
    "Drag the Words",
    "Fill in the Blanks",
    "Mark the Words",
    "Multiple Choice",
    "Quiz",
    "Single Choice",
}
LIMITED_Q_MIN = 4
LIMITED_Q_MAX_SINGLE_PDF = 8
LIMITED_Q_MAX_MULTI_PDF = 12
# Text-driven generators we implement directly (others use the generic patcher)
BUILTIN_TEXT_TYPES = {
    "Drag the Words": {"textfield_keys": ["textField", "text", "questionText", "content"], "mode": "dragtext"},
    "Fill in the Blanks": {"textfield_keys": ["textField", "text", "questionText", "content"], "mode": "blanks"},
    "Mark the Words": {"textfield_keys": ["textField", "text", "questionText", "content"], "mode": "markwords"},
}
@dataclass
class ContentChunk:
    source_file: str
    locator: str
    text: str


def safe_filename(name: str, max_len: int = 90) -> str:
    name = re.sub(r"[^\w\s\-\.]", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] if len(name) > max_len else name


def file_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def discover_templates(templates_dir: str = "templates") -> Dict[str, str]:
    """
    Returns: label -> filepath
    label is filename without extension.
    """
    out: Dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(templates_dir, "*.h5p"))):
        label = os.path.splitext(os.path.basename(p))[0]
        out[label] = p
    return out

def _norm_label(s: str) -> str:
    """Normalise labels for fuzzy matching user-entered types to template filenames."""
    s = (s or "").strip().lower()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def resolve_template_label(user_input: str, templates_map: Dict[str, str]) -> Tuple[Optional[str], List[str]]:
    """
    Attempt to map user-entered activity type to an existing template label.

    Returns:
      (matched_label or None, suggestions list)
    """
    import difflib

    raw = (user_input or "").strip()
    if not raw:
        return None, []

    # Exact match
    if raw in templates_map:
        return raw, []

    norm_map = {_norm_label(k): k for k in templates_map.keys()}
    n = _norm_label(raw)

    if n in norm_map:
        return norm_map[n], []

    # Extra heuristics for common typos/spacing
    n2 = n.replace("h5p ", "").replace(" h5p", "").strip()
    if n2 in norm_map:
        return norm_map[n2], []

    close = difflib.get_close_matches(n, list(norm_map.keys()), n=5, cutoff=0.72)
    suggestions = [norm_map[c] for c in close]
    return (suggestions[0] if suggestions else None), suggestions



def unzip_h5p(h5p_path: str, out_dir: str) -> None:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(h5p_path, "r") as z:
        z.extractall(out_dir)


def zip_dir_to_file(in_dir: str, out_path: str) -> None:
    if os.path.exists(out_path):
        os.remove(out_path)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(in_dir):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, in_dir)
                z.write(full, rel)


def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_chunks_from_bytes(filename: str, pdf_bytes: bytes, max_pages: int = 300) -> List[ContentChunk]:
    # Write to temp file because PdfReader expects file path reliably
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        chunks: List[ContentChunk] = []
        total = min(len(reader.pages), max_pages)
        for i in range(total):
            text = _clean_text(reader.pages[i].extract_text() or "")
            if text:
                chunks.append(ContentChunk(source_file=filename, locator=f"PDF p.{i+1}/{total}", text=text))
        return chunks
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ============================================================
# HEADING EXTRACTION + SMART FREEPIK QUERY BUILDING
# (Dynamic: uses course name + PDF headings + item content; no static nudges)
# ============================================================

_HEADING_MIN_LEN = 8
_HEADING_MAX_LEN = 90

def _looks_like_heading(line: str) -> bool:
    s = (line or "").strip()
    if len(s) < _HEADING_MIN_LEN or len(s) > _HEADING_MAX_LEN:
        return False
    if s.endswith((".", ":", ";")):
        return False
    if re.search(r"(www\.|http|@)", s.lower()):
        return False

    # Numbered headings like "1.2 The SEND Code of Practice"
    if re.match(r"^\d+(\.\d+)*\s+.+", s):
        return True

    # ALL CAPS headings (reasonable length)
    if s.isupper() and len(s.split()) <= 12:
        return True

    # Title Case-ish: many words start with capitals
    words = s.split()
    if 2 <= len(words) <= 12:
        cap = sum(1 for w in words if re.match(r"^[A-Z][a-z]", w))
        if cap >= max(2, int(0.6 * len(words))):
            return True

    return False


def extract_pdf_headings_from_bytes(filename: str, pdf_bytes: bytes, max_pages: int = 300) -> List[str]:
    """Best-effort extraction of headings/titles from PDF text.
    Uses lightweight heuristics (numbered headings, caps, title-case lines).
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    headings: List[str] = []
    try:
        reader = PdfReader(tmp_path)
        total = min(len(reader.pages), max_pages)
        for i in range(total):
            raw = reader.pages[i].extract_text() or ""
            raw = raw.replace("\r", "\n")
            for ln in raw.split("\n"):
                ln2 = re.sub(r"\s+", " ", (ln or "").strip())
                if not ln2:
                    continue
                if _looks_like_heading(ln2) and ln2 not in headings:
                    headings.append(ln2)
        return headings
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _overlap_terms(a: str, b: str) -> int:
    return len(set(_terms(a)) & set(_terms(b)))


def choose_best_heading(context: str, headings: List[str]) -> Optional[str]:
    if not headings:
        return None
    scored = [(h, _overlap_terms(context, h)) for h in headings]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, score = scored[0]
    return best if score > 0 else None


def _expand_course_terms(course: str) -> List[str]:
    """Expand only what the user actually typed.
    - Tokenises the course name
    - Adds expansions for common abbreviations ONLY if they appear in the course string
      (e.g., ICT -> information technology / computer science).
    """
    course = (course or "").strip()
    base = _terms(course)

    expansions: List[str] = []
    c = course.lower()

    if re.search(r"\bict\b", c):
        expansions += ["information technology", "computer science", "computing"]
    if re.search(r"\bit\b", c):
        expansions += ["information technology", "computing"]
    if re.search(r"\bgdpr\b", c):
        expansions += ["data protection", "privacy"]
    if re.search(r"\behcp\b", c):
        expansions += ["education health care plan"]
    if re.search(r"\bsend\b", c):
        expansions += ["special educational needs", "inclusive education"]

    out: List[str] = []
    for t in base + expansions:
        t = (t or "").strip()
        if not t:
            continue
        if t not in out:
            out.append(t)
    return out


def build_image_queries(
    course: str,
    pdf_headings: List[str],
    context_text: str,
    pdf_keywords: Optional[List[str]] = None,
    llm_image_query: str = "",
    max_queries: int = 10,
) -> List[str]:
    """Build Freepik/Wikimedia queries using ONLY:
      - course name
      - headings found in the PDF
      - the current generated item (question/answer/bullets/body)

    No static domain nudges.
    """
    course = (course or "").strip()
    ctx = (context_text or "").strip()
    llm_q = (llm_image_query or "").strip()
    pdf_keywords = pdf_keywords or []

    course_terms = _expand_course_terms(course)
    course_kw = " ".join(course_terms[:4]).strip()

    ctx_terms = extract_keywords(ctx, 6)
    ctx_kw = " ".join(ctx_terms[:4]).strip()
    pdf_kw = " ".join([str(k) for k in (pdf_keywords or [])[:4]]).strip()

    best_heading = choose_best_heading(ctx, pdf_headings or [])

    queries: List[str] = []

    # Anchor any LLM query with course keywords to reduce irrelevant imagery
    if llm_q and course_kw:
        queries.append(f"{llm_q} {course_kw}")
    if llm_q:
        queries.append(llm_q)

    # Course + best PDF heading (strongest)
    if course and best_heading:
        queries.append(f"{course} {best_heading}")
    if course_kw and best_heading:
        queries.append(f"{course_kw} {best_heading}")

    # Heading + content keywords
    if best_heading and ctx_kw:
        queries.append(f"{best_heading} {ctx_kw}")
    if best_heading:
        queries.append(best_heading)

    # Course + content keywords
    if course_kw and ctx_kw:
        queries.append(f"{course_kw} {ctx_kw}")
    if course and ctx_kw:
        queries.append(f"{course} {ctx_kw}")

    # Add globally-relevant PDF keywords to keep results on-topic
    if pdf_kw and ctx_kw:
        queries.append(f"{ctx_kw} {pdf_kw}")
    if course_kw and pdf_kw:
        queries.append(f"{course_kw} {pdf_kw}")
    if pdf_kw:
        queries.append(pdf_kw)

    # Additional fallback headings (top 2 by overlap with course terms)
    if pdf_headings:
        scored = [(h, _overlap_terms(h, " ".join(course_terms))) for h in pdf_headings]
        scored.sort(key=lambda x: x[1], reverse=True)
        for h, _ in scored[:2]:
            if h and h not in queries:
                queries.append(h)

    # De-dup + normalise
    out: List[str] = []
    seen = set()
    for q in queries:
        q = re.sub(r"\s+", " ", (q or "").strip())
        if not q:
            continue
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(q)

    return out[:max_queries]


def build_fallback_query(course: str, pdf_headings: List[str]) -> str:
    course_terms = _expand_course_terms(course)
    if course_terms:
        return f"{course_terms[0]} education"
    if pdf_headings:
        return f"{pdf_headings[0]} education"
    return "education training"




def _find_first_library_list(d: Any) -> Optional[List]:
    """Find a list that appears to contain H5P 'library' items."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "content" and isinstance(v, list) and (not v or (isinstance(v[0], dict) and "library" in v[0])):
                return v
            found = _find_first_library_list(v)
            if found is not None:
                return found
    elif isinstance(d, list):
        for v in d:
            found = _find_first_library_list(v)
            if found is not None:
                return found
    return None


def call_llm_dialog_cards(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    system = ("You are a strict content extractor. You ONLY use text that appears verbatim in SOURCE. "
              "Do not add, infer, embellish, or rephrase ANY facts. Every word in your answer must be traceable "
              "to the SOURCE text. Return JSON only.")
    src_txt = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Create {n} Dialog Cards from SOURCE only for course: {course}

Return JSON:
{{
  "title":"string",
  "description":"string",
  "cards":[
    {{
      "front":"string",
      "back":"string",
      "image_query":"string",
      "evidence":{{"source_file":"string","locator":"PDF p.X/Y","quote":"short exact quote"}}
    }}
  ]
}}

Rules (STRICT — violations cause rejection):
- FRONT must be a single clear question sentence ending with '?'.
- BACK must be a 3–8 word answer phrase copied EXACTLY from the QUOTE (verbatim substring, letter-for-letter match).
- BACK MUST appear word-for-word inside evidence.quote — no rephrasing, no reordering, no synonyms.
- FRONT must be answerable directly from the QUOTE and reuse key terms from it (no new facts).
- The card must be directly supported by the QUOTE.
- evidence.quote must be copied character-for-character from SOURCE — do NOT paraphrase or summarise.
- Double-check: after writing each card, verify that BACK is a contiguous substring of evidence.quote.
- image_query must be 2–6 words describing the professional TOPIC/CONCEPT of the card (e.g. "workplace safety equipment", "first aid training", "health hygiene standards"). Use specific domain terminology from the PDF, NOT generic terms like "child care" or "education". No brands, no logos, no text overlays, no people's names.

SOURCE:
{src_txt}
""".strip()
    return call_openai_chat_json(system, user)


_DIALOG_QWORDS = {"what","which","who","whom","whose","when","where","why","how","define","definition","describe","identify","name"}

def _dialog_front_terms(front: str) -> List[str]:
    t = _terms(front)
    return [x for x in t if x not in _DIALOG_QWORDS]

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _source_text_for_validation(chunks: List[ContentChunk]) -> str:
    # Normalise whitespace to make quote checks robust to PDF extraction quirks
    return _norm_ws("\n\n".join([c.text for c in chunks if c.text]))

def validate_dialog_card(card: Dict[str, Any], source_text: str) -> Tuple[bool, str]:
    """Hard validation to keep Dialog Cards 100% grounded in PDF text."""
    front = (card.get("front") or card.get("text") or "").strip()
    back = (card.get("back") or card.get("answer") or "").strip()
    ev = card.get("evidence") or {}
    quote = (ev.get("quote") or "").strip()

    if not front or not back or not quote:
        return False, "missing front/back/quote"
    if not front.endswith("?"):
        return False, "front not a question"
    # Quote must be substantial enough to be meaningful evidence
    if len(quote.split()) < 5:
        return False, "quote too short to be meaningful evidence"
    # 3–8 words
    bw = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", back)
    if len(bw) < 3 or len(bw) > 8:
        return False, "back word-count out of range"
    # Quote must exist in extracted PDF text
    qn = _norm_ws(quote)
    if qn not in source_text:
        # Also try case-insensitive match for robustness
        if qn.lower() not in source_text.lower():
            return False, "quote not found in PDF text"
    # Back must appear inside quote (case-insensitive, word-boundary)
    if re.search(r"\b" + re.escape(back) + r"\b", qn, re.IGNORECASE) is None:
        # Also try without word boundaries for hyphenated/compound terms
        if back.lower() not in qn.lower():
            return False, "back not found in quote"
    # Front must be about the quote (term overlap check)
    f_terms = set(_dialog_front_terms(front))
    q_terms = set(_terms(qn))
    if f_terms:
        overlap = len(f_terms & q_terms)
        if overlap < max(2, int(0.5 * min(len(f_terms), len(q_terms) or 1))):
            return False, "front not aligned to quote"
    return True, "ok"

def generate_dialog_cards_strict(
    chunks: List[ContentChunk],
    desired_n: int,
    course_context: str,
    max_attempts: int = 4,
) -> Dict[str, Any]:
    """Generate Dialog Cards with hard post-validation + re-tries until we have desired_n."""
    desired_n = int(max(3, min(5, desired_n)))
    source_text = _source_text_for_validation(chunks)

    title = f"Dialog Cards - {course_context}"
    description = ""
    valid_cards: List[Dict[str, Any]] = []
    used_quotes = set()

    for attempt in range(max_attempts):
        remaining = desired_n - len(valid_cards)
        if remaining <= 0:
            break

        # Oversample slightly to survive validation rejections
        request_n = min(8, remaining + 3)
        gen = call_llm_dialog_cards(chunks, request_n, course_context) or {}

        if not description:
            description = (gen.get("description") or "").strip()
        if gen.get("title"):
            title = gen.get("title")

        for c in (gen.get("cards") or []):
            ev = c.get("evidence") or {}
            quote = (ev.get("quote") or "").strip()
            qn = _norm_ws(quote)
            if not qn or qn in used_quotes:
                continue
            ok, _reason = validate_dialog_card(c, source_text)
            if not ok:
                continue
            used_quotes.add(qn)
            valid_cards.append(c)
            if len(valid_cards) >= desired_n:
                break

    if len(valid_cards) < desired_n:
        # Fail loudly rather than producing inaccurate cards
        raise RuntimeError(
            f"Could only validate {len(valid_cards)}/{desired_n} Dialog Cards against the PDF text. "
            "Try reducing the number of cards or uploading a more text-based PDF."
        )

    return {"title": title, "description": description, "cards": valid_cards}


def update_dialog_cards_template(
    work_dir: str,
    title: str,
    description: str,
    cards: List[Dict[str, Any]],
    course: str = "",
    pdf_headings: Optional[List[str]] = None,
    pdf_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Populate Dialog Cards and attach an illustrative image per card.

    Template-tolerant:
    - Locates the cards list by scoring all lists in content/content.json.
    - Preserves the template's per-card schema by cloning a sample card object.
    """
    pdf_headings = pdf_headings or []
    pdf_keywords = pdf_keywords or []

    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    def _iter_lists(obj: Any, path: str = "") -> List[Tuple[str, List]]:
        found: List[Tuple[str, List]] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else k
                if isinstance(v, list):
                    found.append((p, v))
                found.extend(_iter_lists(v, p))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(_iter_lists(v, f"{path}[{i}]"))
        return found

    def _score_cards_list(key_path: str, lst: List) -> int:
        score = 0
        kp = key_path.lower()
        last = kp.split('.')[-1]
        if "card" in last:
            score += 12
        if last in {"cards", "card"}:
            score += 10
        if "dialog" in kp:
            score += 3
        if not isinstance(lst, list):
            return -10
        if len(lst) == 0:
            score += 2
        if len(lst) > 0 and isinstance(lst[0], dict):
            keys = {k.lower() for k in lst[0].keys()}
            if {"text", "answer"} <= keys:
                score += 30
            if {"front", "back"} <= keys:
                score += 30
            if "image" in keys:
                score += 6
            if "subcontentid" in keys:
                score += 3
        return score

    candidates = _iter_lists(content)
    if not candidates:
        raise KeyError("Dialog Cards template content/content.json does not contain any lists.")

    key_path, cards_list_ref = max(candidates, key=lambda kv: _score_cards_list(kv[0], kv[1]))
    if _score_cards_list(key_path, cards_list_ref) < 12:
        top = sorted(((p, _score_cards_list(p, l)) for p, l in candidates), key=lambda x: x[1], reverse=True)[:15]
        raise KeyError(
            "Could not locate the Dialog Cards list in template content/content.json. "
            "Top list candidates (path -> score): " + ", ".join([f"{p} -> {s}" for p, s in top])
        )

    sample_card: Optional[Dict[str, Any]] = None
    if isinstance(cards_list_ref, list) and cards_list_ref and isinstance(cards_list_ref[0], dict):
        sample_card = cards_list_ref[0]

    def _pick_key(sample: Optional[Dict[str, Any]], options: List[str], default: str) -> str:
        if not sample:
            return default
        lower_map = {k.lower(): k for k in sample.keys()}
        for opt in options:
            if opt.lower() in lower_map:
                return lower_map[opt.lower()]
        return default

    front_key = _pick_key(sample_card, ["text", "front", "question", "prompt"], "text")
    back_key = _pick_key(sample_card, ["answer", "back", "solution"], "answer")
    image_key = _pick_key(sample_card, ["image", "picture", "illustration", "media"], "image")

    def _clean_short_answer(s: str, max_words: int = 8) -> str:
        s = (s or "").strip()
        s = re.sub(r"[\s\.,;:!\?]+$", "", s)
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", s)
        if not words:
            return s
        words = words[:max_words]
        while words and words[-1].lower() in {"or", "and"}:
            words = words[:-1]
        return " ".join(words)

    def _set_image_fields(obj: Any, rel_path: str, mime: str, credit: Optional[Dict[str, Any]] = None) -> Any:
        c = credit or {}
        copyright_obj = {
            "license": c.get("license_url", c.get("license", "")) or "",
            "source": c.get("source", "") or "",
            "title": c.get("title", "") or "",
            "author": c.get("author", "") or "",
        }
        copyright_obj = {k: v for k, v in copyright_obj.items() if v}

        if obj is None:
            base = {"path": rel_path, "mime": mime}
            if copyright_obj:
                base["copyright"] = copyright_obj
            return base

        if isinstance(obj, dict):
            out = copy.deepcopy(obj)
            if "path" in out and isinstance(out["path"], str):
                out["path"] = rel_path
            if "mime" in out and isinstance(out.get("mime"), str):
                out["mime"] = mime
            for k, v in list(out.items()):
                if isinstance(v, (dict, list)):
                    out[k] = _set_image_fields(v, rel_path, mime, credit)
            if "path" not in out:
                out["path"] = rel_path
            if "mime" not in out:
                out["mime"] = mime
            if copyright_obj and "copyright" not in out:
                out["copyright"] = copyright_obj
            return out

        if isinstance(obj, list):
            return [_set_image_fields(v, rel_path, mime, credit) for v in obj]

        base = {"path": rel_path, "mime": mime}
        if copyright_obj:
            base["copyright"] = copyright_obj
        return base

    images_dir = os.path.join(work_dir, "content", "images")
    os.makedirs(images_dir, exist_ok=True)

    new_cards: List[Dict[str, Any]] = []
    qa_items: List[Dict[str, Any]] = []

    for i, c in enumerate(cards, start=1):
        front = (c.get("front") or c.get("text") or "").strip()
        back = _clean_short_answer(c.get("back") or c.get("answer") or "")
        img_q = (c.get("image_query") or "").strip()
        ev = c.get("evidence") or {}
        src_file = (ev.get("source_file") or "").strip()
        locator = (ev.get("locator") or "").strip()
        quote = (ev.get("quote") or "").strip()

        if not front or not back:
            continue

        context_for_img = f"{front} {back} {quote}".strip()
        queries = build_image_queries(course=course, pdf_headings=pdf_headings, pdf_keywords=pdf_keywords, context_text=context_for_img, llm_image_query=img_q)
        fallback = build_fallback_query(course, pdf_headings)
        dl = ensure_image(images_dir, queries=queries, stem=f"dialog_{i}", fallback_query=fallback, prefer_vectors=True)

        card_obj = copy.deepcopy(sample_card) if sample_card else {}
        card_obj[front_key] = front
        card_obj[back_key] = back

        existing_img = card_obj.get(image_key)
        card_obj[image_key] = _set_image_fields(existing_img, dl["path"], dl["mime"], dl.get("credit"))

        new_cards.append(card_obj)

        qa_items.append({
            "label": "Dialog Card",
            "content": f"Front: {front}\nBack: {back}",
            "expected": back,
            "evidence": {"source_file": src_file, "locator": locator, "quote": quote},
        })

    if not new_cards:
        raise ValueError("No dialog cards were generated (empty output after validation).")

    cards_list_ref[:] = new_cards
    deep_find_set_first(content, ["title"], title)
    deep_find_set_first(content, ["description", "introduction", "taskDescription"], description)
    _save_json(work_dir, "content/content.json", content)
    return qa_items

def call_llm_multichoice_questions(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    system = "Create Multiple Choice questions strictly grounded in source text. Return JSON only."
    src_txt = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Create {n} multiple choice questions based only on the SOURCE text for course: {course}

Rules:
- Keep answers/options concise (typically 1–6 words where applicable).
- Each question must be answerable from the SOURCE.
- Provide 3-5 options per question.
- Exactly one option is correct.
- Evidence must quote the exact relevant sentence(s) from the SOURCE.

JSON:
{{
  "title":"string",
  "description":"string",
  "items":[
    {{
      "question":"string",
      "options":["string","string","string"],
      "correctIndex":0,
      "evidence":{{"source_file":"string","locator":"PDF p.X/Y","quote":"short exact quote"}}
    }}
  ]
}}

SOURCE:
{src_txt}
"""
    return call_openai_chat_json(system, user)


def build_question_set_multichoice(work_dir: str, title: str, description: str, mc_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    deep_find_set_first(
        content,
        ["introPage", "introduction", "taskDescription"],
        {"showIntroPage": True, "title": title, "introduction": description},
    )

    def find_questions_list(d: Any) -> Optional[List]:
        if isinstance(d, dict):
            if "questions" in d and isinstance(d["questions"], list):
                return d["questions"]
            for v in d.values():
                q = find_questions_list(v)
                if q is not None:
                    return q
        elif isinstance(d, list):
            for v in d:
                q = find_questions_list(v)
                if q is not None:
                    return q
        return None

    questions_container = find_questions_list(content)
    if questions_container is None:
        raise KeyError("Could not locate 'questions' list in Question Set template content.json")

    def _mc_question(question: str, options: List[str], correct_index: int) -> Dict[str, Any]:
        answers = [{"text": opt, "correct": (j == correct_index)} for j, opt in enumerate(options)]
        return {
            "library": "H5P.MultiChoice 1.16",
            "params": {
                "question": question,
                "answers": answers,
                "behaviour": {
                    "enableRetry": True,
                    "enableSolutionsButton": True,
                    "autoCheck": False,
                    "singlePoint": True,
                    "randomAnswers": False,
                },
            },
            "metadata": {"title": "Multiple Choice", "license": "U"},
        }

    new_questions: List[Dict[str, Any]] = []
    qa: List[Dict[str, Any]] = []

    for i, it in enumerate(mc_items, start=1):
        q = (it.get("question") or "").strip()
        opts = it.get("options") or []
        if not q or not isinstance(opts, list) or len(opts) < 2:
            continue
        correct = int(it.get("correctIndex", 0))
        correct = max(0, min(correct, len(opts) - 1))
        new_questions.append(_mc_question(q, opts, correct))

        qa.append({
            "label": f"{i}) Multiple Choice",
            "content": q,
            "expected": opts[correct] if opts else "",
            "evidence": it.get("evidence", {}) or {},
        })

    if not new_questions:
        raise ValueError("No Multiple Choice items were generated.")

    questions_container[:] = new_questions
    _save_json(work_dir, "content/content.json", content)
    return qa


def call_llm_page_content(chunks: List[ContentChunk], n_sections: int, course: str) -> Dict[str, Any]:
    system = "Create an H5P Page layout with sections grounded strictly in SOURCE. Return JSON only."
    src_txt = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Create a Page activity for course: {course}

Return JSON:
{{
  "title":"string",
  "sections":[
    {{
      "heading":"string",
      "body_html":"string",
      "image_query":"string",
      "evidence":{{"source_file":"string","locator":"PDF p.X/Y","quote":"short exact quote"}}
    }}
  ]
}}

Rules:
- 3 to {n_sections} sections.
- body_html must be simple HTML (p, ul, li, b).
- image_query must be 2–6 words for a clear, text-free illustrative image (no logos/brands, no source names).
- Evidence quote must be copied exactly from SOURCE and match the section content.

SOURCE:
{src_txt}
"""
    return call_openai_chat_json(system, user)


def update_page_template_with_images(
    work_dir: str,
    title: str,
    sections: List[Dict[str, Any]],
    course: str = "",
    pdf_headings: Optional[List[str]] = None,
    pdf_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Populate an H5P Page with Image + Text blocks."""
    pdf_headings = pdf_headings or []
    pdf_keywords = pdf_keywords or []

    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    lib_list = _find_first_library_list(content)
    if lib_list is None:
        raise KeyError("Could not find Page 'content' list to populate.")

    images_dir = os.path.join(work_dir, "content", "images")
    os.makedirs(images_dir, exist_ok=True)

    new_blocks: List[Dict[str, Any]] = []
    qa_items: List[Dict[str, Any]] = []

    def h5p_image(img_rel: str, mime: str, alt: str = "", caption: str = "") -> Dict[str, Any]:
        return {
            "library": "H5P.Image 1.1",
            "params": {
                "title": caption,
                "alt": alt,
                "file": {
                    "path": img_rel,
                    "mime": mime,
                    "copyright": {"license": "U"},
                },
            },
            "metadata": {"title": "Image", "license": "U"},
        }

    def h5p_adv_text(html: str) -> Dict[str, Any]:
        return {
            "library": "H5P.AdvancedText 1.1",
            "params": {"text": html},
            "metadata": {"title": "Text", "license": "U"},
        }

    for i, sec in enumerate(sections, start=1):
        heading = (sec.get("heading") or "").strip()
        body = (sec.get("body_html") or "").strip()
        img_q = (sec.get("image_query") or "").strip()
        ev = sec.get("evidence") or {}
        src_file = (ev.get("source_file") or "").strip()
        locator = (ev.get("locator") or "").strip()
        quote = (ev.get("quote") or "").strip()

        if not body:
            continue

        context_for_img = f"{heading} {re.sub('<[^<]+?>','', body)}".strip()
        queries = build_image_queries(course=course, pdf_headings=pdf_headings, pdf_keywords=pdf_keywords, context_text=context_for_img, llm_image_query=img_q)
        fallback = build_fallback_query(course, pdf_headings)
        dl = ensure_image(images_dir, queries=queries, stem=f"page_{i}", fallback_query=fallback)

        new_blocks.append(h5p_image(dl["path"], dl["mime"], alt=heading, caption=heading))

        html = f"<h3>{heading}</h3>\n{body}" if heading else body
        new_blocks.append(h5p_adv_text(html))

        qa_items.append({
            "label": f"Page section {i}",
            "content": f"{heading}\n{re.sub('<[^<]+?>','',body)[:500]}",
            "expected": "",
            "evidence": {"source_file": src_file, "locator": locator, "quote": quote},
        })

    if not new_blocks:
        raise ValueError("No Page content was generated.")

    lib_list[:] = new_blocks
    _save_json(work_dir, "content/content.json", content)
    return qa_items

def h5p_set_image_fields(obj: Any, rel_path: str, mime: str) -> Any:
    """Template-tolerant setter for H5P file objects (commonly used for images).
    It attempts to preserve the existing schema while updating path/mime recursively.
    """
    if obj is None:
        return {"path": rel_path, "mime": mime, "copyright": {"license": "U"}}
    if isinstance(obj, dict):
        out = copy.deepcopy(obj)
        if "path" in out and isinstance(out["path"], str):
            out["path"] = rel_path
        if "mime" in out and isinstance(out["mime"], str):
            out["mime"] = mime
        for k, v in list(out.items()):
            if isinstance(v, (dict, list)):
                out[k] = h5p_set_image_fields(v, rel_path, mime)
        if "path" not in out:
            out["path"] = rel_path
        if "mime" not in out:
            out["mime"] = mime
        if "copyright" not in out:
            out["copyright"] = {"license": "U"}
        return out
    if isinstance(obj, list):
        return [h5p_set_image_fields(v, rel_path, mime) for v in obj]
    return {"path": rel_path, "mime": mime, "copyright": {"license": "U"}}


def call_llm_course_presentation(chunks: List[ContentChunk], n_slides: int, course: str) -> Dict[str, Any]:
    system = (
        "You are a strict content extractor. You ONLY use text that exists word-for-word in the SOURCE. "
        "You NEVER add, infer, or rephrase. Every bullet point must be a direct key fact from the SOURCE — "
        "short, plain English, maximum 12 words each. Return valid JSON only."
    )
    src_txt = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Extract content from the SOURCE text below and organise it into EXACTLY {n_slides} presentation slides
for the course: {course}

Return JSON:
{{
  "title":"string — use the course/unit title from SOURCE",
  "description":"string",
  "slides":[
    {{
      "heading":"string — a section heading or topic name found in SOURCE",
      "bullets":["short key point (max 12 words)","short key point","short key point"],
      "image_keywords":"string — 2-5 descriptive nouns from the slide content for photo search",
      "evidence":{{"source_file":"string","locator":"Page X","quote":"exact sentence copied from SOURCE"}}
    }}
  ]
}}

STRICT RULES:
1. EXACTLY {n_slides} slides — no more, no less.
2. Each "heading" MUST be a topic or section title that appears in the SOURCE.
3. Each bullet MUST be a factual statement directly stated in the SOURCE.
   - Use 3-5 SHORT bullets per slide (≤12 words each).
   - Pick only the most important facts; do NOT overload slides with text.
   - Copy key phrases from SOURCE. Do NOT rephrase or add new information.
4. "bullets" must be a JSON array of plain strings — never nested objects.
5. evidence.quote MUST be an exact sentence copied verbatim from SOURCE.
6. image_keywords: pick 2-5 descriptive nouns from the slide's own content
   (e.g. "workplace safety training equipment"). No file names, brand names, or generic terms.
7. Cover different sections/topics of the SOURCE across slides — do not repeat the same content.
8. If SOURCE does not have enough distinct topics, go deeper into subtopics.

SOURCE:
{src_txt}
""".strip()
    return call_openai_chat_json(system, user)


def call_llm_cp_activity_questions(
    chunks: List[ContentChunk],
    activity_type: str,
    n_questions: int,
    course: str,
) -> Dict[str, Any]:
    """Generate questions for a single activity type, strictly from PDF source text."""
    system = (
        "You are a strict quiz creator. Every question, sentence, and answer MUST come directly "
        "from the SOURCE text. You NEVER invent facts. Copy real sentences from SOURCE and turn "
        "them into questions. Return valid JSON only."
    )
    src_txt = join_chunks_for_prompt(chunks, max_chars=65000)

    at = activity_type.strip()

    if "true" in at.lower() or "false" in at.lower():
        schema = """[
    {{
      "statement": "a factual statement copied or closely derived from SOURCE",
      "correct_answer": "True or False",
      "evidence": {{"source_file":"string","locator":"Page X","quote":"the exact SOURCE sentence this is based on"}}
    }}
  ]"""
        type_rules = """- Take a real sentence from SOURCE and present it as a True/False statement.
- For \"True\" items: use the sentence as-is or minimally shortened.
- For \"False\" items: change one key fact (e.g. swap a number or term) so it becomes false.
- correct_answer must be exactly \"True\" or \"False\".
- Mix of True and False answers (not all the same)."""
    elif "fill" in at.lower() or "blank" in at.lower():
        schema = """[
    {{
      "sentence": "a real sentence from SOURCE with one important word replaced by ___",
      "answer": "the removed word",
      "evidence": {{"source_file":"string","locator":"Page X","quote":"the original complete sentence from SOURCE"}}
    }}
  ]"""
        type_rules = """- Find a real sentence in SOURCE.
- Remove ONE important keyword and replace it with ___ (three underscores).
- answer is the exact word you removed.
- The sentence (with the blank) must still be recognisable from SOURCE."""
    elif "drag" in at.lower() and "word" in at.lower():
        schema = """[
    {{
      "sentence": "a real sentence from SOURCE with one key word replaced by ___",
      "missing_word": "the correct word to drag in",
      "distractors": ["wrong1","wrong2"],
      "evidence": {{"source_file":"string","locator":"Page X","quote":"the original complete sentence from SOURCE"}}
    }}
  ]"""
        type_rules = """- Find a real sentence in SOURCE.
- Remove ONE important keyword and replace it with ___.
- missing_word is the correct word.
- distractors: 2 plausible but incorrect alternatives (also relevant to the SOURCE topic)."""
    elif "mark" in at.lower() and "word" in at.lower():
        schema = """[
    {{
      "paragraph": "1-2 sentences copied from SOURCE",
      "marked_words": ["word1","word2","word3"],
      "evidence": {{"source_file":"string","locator":"Page X","quote":"the exact SOURCE text copied"}}
    }}
  ]"""
        type_rules = """- Copy 1-2 real sentences from SOURCE as the paragraph.
- marked_words: 2-4 important key terms/words within that paragraph that the learner should identify."""
    else:
        schema = """[
    {{
      "statement": "a factual statement from SOURCE",
      "correct_answer": "True or False",
      "evidence": {{"source_file":"string","locator":"Page X","quote":"exact SOURCE sentence"}}
    }}
  ]"""
        type_rules = "- Fallback to True/False format. Same rules as True/False above."

    user = f"""
Create EXACTLY {n_questions} "{at}" questions based on the SOURCE text below.
Course: {course}

Return JSON:
{{
  "questions": {schema}
}}

STRICT RULES:
1. EXACTLY {n_questions} questions.
2. Every question MUST be based on a specific sentence or fact from the SOURCE text.
{type_rules}
3. evidence.quote MUST be the exact original sentence from SOURCE that the question is based on.
4. Do NOT invent any facts, numbers, or terms that are not in the SOURCE.
5. Spread questions across different parts of the SOURCE — do not cluster them from one section.

SOURCE:
{src_txt}
""".strip()
    return call_openai_chat_json(system, user)


def call_llm_interactive_book(chunks: List[ContentChunk], n_chapters: int, course: str) -> Dict[str, Any]:
    system = "Create an H5P Interactive Book grounded strictly in SOURCE. Return JSON only."
    src_txt = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Create an Interactive Book for course: {course}

Return JSON:
{{
  "title":"string",
  "description":"string",
  "chapters":[
    {{
      "chapter_title":"string",
      "sections":[
        {{
          "heading":"string",
          "body_html":"string",
          "image_query":"string",
          "evidence":{{"source_file":"string","locator":"PDF p.X/Y","quote":"short exact quote"}}
        }}
      ]
    }}
  ]
}}

Rules:
- Create 2 to {n_chapters} chapters.
- Each chapter must contain 2–4 sections.
- body_html must use simple HTML only (p, ul, li, b).
- image_query must be 2–6 words for a clear, text-free illustrative photo (no logos/brands, no source names).
- Evidence quote must be copied exactly from SOURCE and support the section content.

SOURCE:
{src_txt}
"""
    return call_openai_chat_json(system, user)


def _iter_library_blocks(obj: Any):
    """Yield dicts that look like H5P library blocks: {'library': str, 'params': dict}."""
    if isinstance(obj, dict):
        if isinstance(obj.get("library"), str) and isinstance(obj.get("params"), dict):
            yield obj
        for v in obj.values():
            yield from _iter_library_blocks(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_library_blocks(v)


def _best_list_by_score(root: Any, scorer) -> Optional[List]:
    """Return a reference to the best matching list in a JSON object based on scorer(path, list_obj)."""
    best = None
    best_score = -10_000

    def rec(obj: Any, path: str = ""):
        nonlocal best, best_score
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else k
                if isinstance(v, list):
                    s = scorer(p, v)
                    if s > best_score:
                        best_score = s
                        best = v
                rec(v, p)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                rec(v, f"{path}[{i}]")

    rec(root)
    return best


def _score_slides_list(path: str, lst: List) -> int:
    kp = (path or "").lower()
    last = kp.split(".")[-1]
    score = 0
    if last in {"slides", "slide"}:
        score += 30
    if "coursepresentation" in kp or "presentation" in kp:
        score += 8
    if "slides" in kp:
        score += 10
    if not isinstance(lst, list):
        return -10
    if len(lst) == 0:
        score += 2
    if len(lst) > 0 and isinstance(lst[0], dict):
        keys = {k.lower() for k in lst[0].keys()}
        if "elements" in keys:
            score += 25
        if {"title", "slidetitle"} & keys:
            score += 10
    return score


def _score_chapters_list(path: str, lst: List) -> int:
    kp = (path or "").lower()
    last = kp.split(".")[-1]
    score = 0
    if last in {"chapters", "chapter"}:
        score += 35
    if "interactivebook" in kp or "book" in kp:
        score += 8
    if "chapters" in kp:
        score += 10
    if not isinstance(lst, list):
        return -10
    if len(lst) == 0:
        score += 2
    if len(lst) > 0 and isinstance(lst[0], dict):
        keys = {k.lower() for k in lst[0].keys()}
        if {"content", "sections", "params"} & keys:
            score += 12
        if {"title", "chapter_title"} & keys:
            score += 10
    return score


def _html_bullets(heading: str, bullets: List[str], max_bullets: int = 4) -> str:
    h = (heading or "").strip()
    clean = [(b or '').strip() for b in (bullets or []) if (b or '').strip()]
    clean = clean[:max_bullets]  # cap to avoid overloaded slides
    li = "".join([f"<li>{b}</li>" for b in clean])
    if not li:
        li = "<li>—</li>"
    if h:
        return f"<h2>{h}</h2><ul>{li}</ul>"
    return f"<ul>{li}</ul>"


def _build_cp_activity_slide_elements(activity_type: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build H5P element(s) for ALL activity questions on a SINGLE Course Presentation slide.

    IMPORTANT: Course Presentation elements use an "action" wrapper:
    { "x":..., "y":..., "action": { "library":"...", "params":{...}, "subContentId":"...", "metadata":{...} } }

    - Fill in the Blanks: one H5P.Blanks element with multiple question lines.
    - Drag the Words: one H5P.DragText element with multi-line textField.
    - Mark the Words: one H5P.MarkTheWords element with combined paragraphs.
    - True/False: one H5P.TrueFalse per question stacked vertically.
    """
    at = (activity_type or "").strip().lower()
    if not at or not questions:
        return []

    elements: List[Dict[str, Any]] = []

    def _make_element(x, y, w, h, library, params, content_type_label):
        return {
            "x": x, "y": y, "width": w, "height": h,
            "displayAsButton": False,
            "buttonSize": "big",
            "backgroundOpacity": 0,
            "action": {
                "library": library,
                "params": params,
                "subContentId": str(uuid.uuid4()),
                "metadata": {
                    "contentType": content_type_label,
                    "license": "U",
                    "title": "Untitled",
                },
            },
        }

    if "fill" in at or "blank" in at:
        q_lines = []
        for q in questions:
            sentence = (q.get("sentence") or "").strip()
            answer = (q.get("answer") or "").strip()
            if not sentence or not answer:
                continue
            text = sentence
            if f"*{answer}*" not in text:
                text = text.replace("___", f"*{answer}*", 1)
            if f"*{answer}*" not in text:
                text = text.replace(answer, f"*{answer}*", 1)
            if f"*{answer}*" not in text:
                text += f" *{answer}*"
            q_lines.append(f"<p>{text}</p>")
        if q_lines:
            elements.append(_make_element(0, 0, 100, 100, "H5P.Blanks 1.14", {
                "questions": q_lines,
                "overallFeedback": [{"from": 0, "to": 100}],
                "showSolutions": "Show solution",
                "tryAgain": "Retry",
                "behaviour": {"enableRetry": True, "enableSolutionsButton": True,
                              "caseSensitive": False, "autoCheck": False, "acceptSpellingErrors": False},
            }, "Fill in the Blanks"))

    elif "drag" in at and "word" in at:
        lines = []
        for q in questions:
            sentence = (q.get("sentence") or "").strip()
            missing = (q.get("missing_word") or "").strip()
            if not sentence or not missing:
                continue
            text = sentence
            if f"*{missing}*" not in text:
                text = text.replace("___", f"*{missing}*", 1)
            if f"*{missing}*" not in text:
                text = text.replace(missing, f"*{missing}*", 1)
            if f"*{missing}*" not in text:
                text += f" *{missing}*"
            lines.append(text)
        if lines:
            elements.append(_make_element(0, 0, 100, 100, "H5P.DragText 1.10", {
                "textField": "\n".join(lines),
                "overallFeedback": [{"from": 0, "to": 100}],
                "behaviour": {"enableRetry": True, "enableSolutionsButton": True, "instantFeedback": False},
                "taskDescription": "<p>Drag the words into the correct blanks.</p>",
            }, "Drag the Words"))

    elif "mark" in at and "word" in at:
        combined_text = ""
        for q in questions:
            paragraph = (q.get("paragraph") or "").strip()
            marked = q.get("marked_words") or []
            if not paragraph or not marked:
                continue
            text = paragraph
            for w in marked:
                w = w.strip()
                if w and f"*{w}*" not in text:
                    text = text.replace(w, f"*{w}*", 1)
            combined_text += text + "\n\n"
        if combined_text.strip():
            elements.append(_make_element(0, 0, 100, 100, "H5P.MarkTheWords 1.11", {
                "textField": combined_text.strip(),
                "overallFeedback": [{"from": 0, "to": 100}],
                "behaviour": {"enableRetry": True, "enableSolutionsButton": True},
                "taskDescription": "<p>Select the correct words.</p>",
                "checkAnswerButton": "Check",
                "tryAgainButton": "Retry",
                "showSolutionButton": "Show solution",
            }, "Mark the Words"))

    elif "true" in at or "false" in at:
        n = max(1, len(questions))
        per_h = max(10, 90 // n)
        for qi, q in enumerate(questions):
            statement = (q.get("statement") or "").strip()
            correct = (q.get("correct_answer") or "").strip().lower()
            if not statement:
                continue
            elements.append(_make_element(0, qi * per_h, 100, per_h, "H5P.TrueFalse 1.8", {
                "question": f"<p>{statement}</p>",
                "correct": "true" if correct in ("true", "yes", "1") else "false",
                "behaviour": {"enableRetry": True, "enableSolutionsButton": True,
                              "confirmCheckDialog": False, "confirmRetryDialog": False},
                "l10n": {"trueText": "True", "falseText": "False"},
                "media": {"type": {}},
            }, "True/False Question"))

    return elements


# Libraries that need to be in h5p.json preloadedDependencies for each activity type
_CP_ACTIVITY_DEPENDENCIES = {
    "true/false": [
        {"machineName": "H5P.TrueFalse", "majorVersion": 1, "minorVersion": 8},
    ],
    "fill in the blanks": [
        {"machineName": "H5P.Blanks", "majorVersion": 1, "minorVersion": 14},
    ],
    "drag the words": [
        {"machineName": "H5P.DragText", "majorVersion": 1, "minorVersion": 10},
    ],
    "mark the words": [
        {"machineName": "H5P.MarkTheWords", "majorVersion": 1, "minorVersion": 11},
    ],
}


def _ensure_cp_activity_dependencies(work_dir: str, activity_type: str) -> None:
    """Add the activity library to h5p.json preloadedDependencies if missing."""
    at_lower = activity_type.strip().lower()
    deps_to_add = _CP_ACTIVITY_DEPENDENCIES.get(at_lower, [])
    if not deps_to_add:
        return

    h5p_path = os.path.join(work_dir, "h5p.json")
    meta = json.loads(open(h5p_path, "r", encoding="utf-8").read())
    existing = meta.get("preloadedDependencies") or []

    existing_names = {d.get("machineName") for d in existing}
    for dep in deps_to_add:
        if dep["machineName"] not in existing_names:
            existing.append(dep)

    meta["preloadedDependencies"] = existing
    with open(h5p_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def update_course_presentation_template_with_images(
    work_dir: str,
    title: str,
    description: str,
    slides: List[Dict[str, Any]],
    course: str = "",
    pdf_headings: Optional[List[str]] = None,
    pdf_keywords: Optional[List[str]] = None,
    activity_groups: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Populate a Course Presentation template.
    Creates N content slides + 1 activity slide (last slide with all questions).

    activity_groups: {activity_type_str: [question_dicts]} — supports multiple types on one slide.
    """
    pdf_headings = pdf_headings or []
    pdf_keywords = pdf_keywords or []
    activity_groups = activity_groups or {}

    # Activity library prefixes to strip from content slides
    _ACTIVITY_LIBS = ("H5P.DragText", "H5P.Blanks", "H5P.TrueFalse", "H5P.MarkTheWords",
                      "H5P.MultiChoice", "H5P.SingleChoiceSet")

    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    slides_ref = _best_list_by_score(content, _score_slides_list)
    if slides_ref is None:
        raise KeyError(
            "Could not locate a 'slides' list in the Course Presentation template. "
            "Export a blank Course Presentation with at least 1 slide and re-add it to ./templates."
        )

    sample_slide = slides_ref[0] if slides_ref and isinstance(slides_ref[0], dict) else None

    images_dir = os.path.join(work_dir, "content", "images")
    os.makedirs(images_dir, exist_ok=True)

    qa_items: List[Dict[str, Any]] = []

    has_activity = bool(activity_groups and any(activity_groups.values()))
    total_needed = len(slides) + (1 if has_activity else 0)
    while len(slides_ref) < total_needed:
        slides_ref.append(copy.deepcopy(sample_slide) if sample_slide else {})
    while len(slides_ref) > total_needed:
        slides_ref.pop()

    # --- Content slides ---
    for i, gen in enumerate(slides, start=1):
        if i > len(slides_ref):
            break

        slide_obj = slides_ref[i - 1]
        heading = (gen.get("heading") or "").strip()
        raw_bullets = gen.get("bullets") or []
        # Normalise: LLM occasionally returns dicts instead of plain strings
        bullets = []
        for b in raw_bullets:
            if isinstance(b, str):
                bullets.append(b.strip())
            elif isinstance(b, dict):
                # Try common keys
                text = b.get("text") or b.get("point") or b.get("content") or b.get("bullet") or ""
                if text:
                    bullets.append(str(text).strip())
        img_kw = (gen.get("image_keywords") or gen.get("image_query") or "").strip()
        ev = gen.get("evidence") or {}

        deep_find_set_first(slide_obj, ["slideTitle", "title", "heading"], heading)

        html = _html_bullets(heading, bullets)

        # --- Strip interactive activity elements from this content slide ---
        if "elements" in slide_obj and isinstance(slide_obj["elements"], list):
            def _is_activity_element(el):
                if not isinstance(el, dict):
                    return False
                lib = str(el.get("library") or "")
                if any(lib.startswith(prefix) for prefix in _ACTIVITY_LIBS):
                    return True
                act = el.get("action")
                if isinstance(act, dict):
                    alib = str(act.get("library") or "")
                    if any(alib.startswith(prefix) for prefix in _ACTIVITY_LIBS):
                        return True
                return False

            slide_obj["elements"] = [el for el in slide_obj["elements"] if not _is_activity_element(el)]

        # --- Update AdvancedText content (or create one if missing) ---
        adv_blocks = [b for b in _iter_library_blocks(slide_obj) if str(b.get("library", "")).startswith("H5P.AdvancedText")]
        if adv_blocks:
            adv_blocks[0].setdefault("params", {})
            adv_blocks[0]["params"]["text"] = html
        elif not deep_find_set_first(slide_obj, ["text", "html", "content", "questionText"], html):
            # No existing text element found — inject one so the slide is never blank
            if "elements" not in slide_obj or not isinstance(slide_obj.get("elements"), list):
                slide_obj["elements"] = []
            # Text occupies right 55 % when image is on slide 1; full width otherwise
            x_pos, width = (2, 55) if i == 1 else (2, 96)
            slide_obj["elements"].append({
                "x": x_pos, "y": 2, "width": width, "height": 90,
                "displayAsButton": False,
                "buttonSize": "big",
                "backgroundOpacity": 0,
                "action": {
                    "library": "H5P.AdvancedText 1.1",
                    "params": {"text": html},
                    "subContentId": str(uuid.uuid4()),
                    "metadata": {
                        "contentType": "Advanced Text",
                        "license": "U",
                        "title": "Untitled",
                    },
                },
            })

        # --- Image: only on the FIRST slide ---
        if i == 1:
            context_for_img = f"{heading} {' '.join([str(x) for x in bullets])}".strip()
            queries = build_image_queries(course=course, pdf_headings=pdf_headings, pdf_keywords=pdf_keywords, context_text=context_for_img, llm_image_query=img_kw)
            fallback = build_fallback_query(course, pdf_headings)
            dl = ensure_image(images_dir, queries=queries, stem=f"cp_slide_{i}", fallback_query=fallback)

            img_blocks = [b for b in _iter_library_blocks(slide_obj) if str(b.get("library", "")).startswith("H5P.Image")]
            if img_blocks:
                file_obj = (img_blocks[0].get("params") or {}).get("file")
                img_blocks[0].setdefault("params", {})
                img_blocks[0]["params"]["file"] = (
                    {"path": dl["path"], "mime": dl["mime"], "copyright": {"license": "U"}}
                    if not isinstance(file_obj, dict)
                    else h5p_set_image_fields(file_obj, dl["path"], dl["mime"])
                )
            else:
                found = deep_find_first_key(slide_obj, ["backgroundImage", "background"])
                if found and isinstance(found[1], dict):
                    k, v = found
                    slide_obj[k] = h5p_set_image_fields(v, dl["path"], dl["mime"])
                else:
                    # Fallback: add a new image element
                    if "elements" not in slide_obj or not isinstance(slide_obj["elements"], list):
                        slide_obj["elements"] = []
                    slide_obj["elements"].append({
                        "x": 60, "y": 5, "width": 36, "height": 90,
                        "displayAsButton": False,
                        "buttonSize": "big",
                        "backgroundOpacity": 0,
                        "action": {
                            "library": "H5P.Image 1.1",
                            "params": {
                                "file": {"path": dl["path"], "mime": dl["mime"], "copyright": {"license": "U"}},
                                "alt": heading or "slide image",
                            },
                            "subContentId": str(uuid.uuid4()),
                            "metadata": {"contentType": "Image", "license": "U", "title": "Untitled"},
                        },
                    })
        else:
            # Remove any image elements from non-first slides
            if "elements" in slide_obj and isinstance(slide_obj["elements"], list):
                def _is_image_element(el):
                    if not isinstance(el, dict):
                        return False
                    lib = str(el.get("library") or "")
                    if lib.startswith("H5P.Image"):
                        return True
                    act = el.get("action")
                    if isinstance(act, dict) and str(act.get("library") or "").startswith("H5P.Image"):
                        return True
                    return False
                slide_obj["elements"] = [el for el in slide_obj["elements"] if not _is_image_element(el)]
        qa_items.append({
            "label": f"Slide {i}",
            "content": f"{heading}\n" + "\n".join([f"- {b}" for b in bullets[:6]]),
            "expected": "",
            "evidence": ev,
        })

    # --- ONE activity slide at the end with ALL questions from all types ---
    if has_activity:
        act_slide_idx = len(slides)
        act_slide_num = act_slide_idx + 1
        slide_obj = slides_ref[act_slide_idx]

        type_names = " & ".join(activity_groups.keys())
        act_heading = f"Activity — {type_names}"
        deep_find_set_first(slide_obj, ["slideTitle", "title", "heading"], act_heading)

        # Build elements from ALL activity types and combine on one slide
        all_act_elements: List[Dict[str, Any]] = []
        for atype, questions in activity_groups.items():
            if questions:
                all_act_elements.extend(_build_cp_activity_slide_elements(atype, questions))
                _ensure_cp_activity_dependencies(work_dir, atype)

        slide_obj["elements"] = all_act_elements if all_act_elements else []

        # QA report entries
        for atype, questions in activity_groups.items():
            for qi, q_data in enumerate(questions):
                ev = q_data.get("evidence") or {}
                q_summary = q_data.get("statement") or q_data.get("sentence") or q_data.get("paragraph") or ""
                q_answer = q_data.get("correct_answer") or q_data.get("answer") or q_data.get("missing_word") or ""
                if isinstance(q_data.get("marked_words"), list):
                    q_answer = ", ".join(q_data["marked_words"])
                qa_items.append({
                    "label": f"Slide {act_slide_num} — {atype} Q{qi + 1}",
                    "content": f"{q_summary}\nAnswer: {q_answer}",
                    "expected": q_answer,
                    "evidence": ev,
                })

    deep_find_set_first(content, ["introduction", "description", "taskDescription"], description)
    _save_json(work_dir, "content/content.json", content)
    return qa_items

def update_interactive_book_template_with_images(
    work_dir: str,
    title: str,
    description: str,
    chapters: List[Dict[str, Any]],
    course: str = "",
    pdf_headings: Optional[List[str]] = None,
    pdf_keywords: Optional[List[str]] = None,
    activity_groups: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Populate an Interactive Book template while preserving its structure."""
    pdf_headings = pdf_headings or []
    pdf_keywords = pdf_keywords or []
    activity_groups = activity_groups or {}

    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    chapters_ref = _best_list_by_score(content, _score_chapters_list)
    if chapters_ref is None:
        raise KeyError(
            "Could not locate a 'chapters' list in the Interactive Book template. "
            "Export a blank Interactive Book with at least 1 chapter and re-add it to ./templates."
        )

    sample_chapter = chapters_ref[0] if chapters_ref and isinstance(chapters_ref[0], dict) else None

    images_dir = os.path.join(work_dir, "content", "images")
    os.makedirs(images_dir, exist_ok=True)

    qa_items: List[Dict[str, Any]] = []

    has_activity = bool(activity_groups and any(activity_groups.values()))
    target_chapters = max(1, len(chapters)) + (1 if has_activity else 0)
    while len(chapters_ref) < target_chapters:
        chapters_ref.append(copy.deepcopy(sample_chapter) if sample_chapter else {})
    while len(chapters_ref) > target_chapters:
        chapters_ref.pop()

    for ci, ch in enumerate(chapters, start=1):
        if ci > len(chapters_ref):
            break

        ch_obj = chapters_ref[ci - 1]
        ch_title = (ch.get("chapter_title") or ch.get("title") or f"Chapter {ci}").strip()
        deep_find_set_first(ch_obj, ["title", "chapterTitle", "chapter_title", "heading"], ch_title)

        sections = ch.get("sections") or []
        parts: List[str] = []
        for sec in sections:
            h = (sec.get("heading") or "").strip()
            body = (sec.get("body_html") or "").strip()
            if h:
                parts.append(f"<h3>{h}</h3>")
            if body:
                parts.append(body)
        chapter_html = "\n".join(parts).strip() or f"<p>{ch_title}</p>"

        adv_blocks = [b for b in _iter_library_blocks(ch_obj) if str(b.get("library", "")).startswith("H5P.AdvancedText")]
        if adv_blocks:
            adv_blocks[0].setdefault("params", {})
            adv_blocks[0]["params"]["text"] = chapter_html
        else:
            deep_find_set_first(ch_obj, ["text", "html", "content", "introduction"], chapter_html)

        first_sec = sections[0] if sections else {}
        img_q = (first_sec.get("image_query") or "").strip()

        context_for_img = re.sub("<[^<]+?>", " ", chapter_html)
        queries = build_image_queries(course=course, pdf_headings=pdf_headings, pdf_keywords=pdf_keywords, context_text=context_for_img, llm_image_query=img_q)
        fallback = build_fallback_query(course, pdf_headings)
        dl = ensure_image(images_dir, queries=queries, stem=f"ib_ch_{ci}", fallback_query=fallback)

        img_blocks = [b for b in _iter_library_blocks(ch_obj) if str(b.get("library", "")).startswith("H5P.Image")]
        if img_blocks:
            file_obj = (img_blocks[0].get("params") or {}).get("file")
            img_blocks[0].setdefault("params", {})
            img_blocks[0]["params"]["file"] = (
                {"path": dl["path"], "mime": dl["mime"], "copyright": {"license": "U"}}
                if not isinstance(file_obj, dict)
                else h5p_set_image_fields(file_obj, dl["path"], dl["mime"])
            )
        else:
            found = deep_find_first_key(ch_obj, ["coverImage", "image", "backgroundImage"])
            if found and isinstance(found[1], dict):
                k, v = found
                ch_obj[k] = h5p_set_image_fields(v, dl["path"], dl["mime"])

        for si, sec in enumerate(sections, start=1):
            ev = sec.get("evidence") or {}
            qa_items.append({
                "label": f"Chapter {ci} — Section {si}",
                "content": f"{sec.get('heading','')}\n{re.sub('<[^<]+?>','', sec.get('body_html',''))[:450]}",
                "expected": "",
                "evidence": ev,
            })

    # --- ONE activity chapter at the end with ALL questions from all types ---
    if has_activity:
        act_ch_idx = len(chapters)
        act_ch_num = act_ch_idx + 1
        ch_obj = chapters_ref[act_ch_idx]

        type_names = " & ".join(activity_groups.keys())
        act_heading = f"Activity — {type_names}"
        deep_find_set_first(ch_obj, ["title", "chapterTitle", "chapter_title", "heading"], act_heading)

        # Build activity HTML content for the chapter
        activity_html_parts: List[str] = []
        activity_html_parts.append(f"<h2>{act_heading}</h2>")

        all_act_elements: List[Dict[str, Any]] = []
        for atype, questions in activity_groups.items():
            if questions:
                all_act_elements.extend(_build_cp_activity_slide_elements(atype, questions))
                _ensure_cp_activity_dependencies(work_dir, atype)

        # Try to inject activity elements into the chapter's content structure
        # Interactive Book chapters may use AdvancedText or a content/params structure
        adv_blocks = [b for b in _iter_library_blocks(ch_obj) if str(b.get("library", "")).startswith("H5P.AdvancedText")]
        if adv_blocks:
            adv_blocks[0].setdefault("params", {})
            adv_blocks[0]["params"]["text"] = f"<h2>{act_heading}</h2><p>Complete the activities below.</p>"
        else:
            deep_find_set_first(ch_obj, ["text", "html", "content", "introduction"], f"<h2>{act_heading}</h2><p>Complete the activities below.</p>")

        # Inject the activity elements into the chapter's content
        # Interactive Book chapters can hold H5P sub-content via a 'content' list or 'params.content'
        if "content" in ch_obj and isinstance(ch_obj["content"], list):
            ch_obj["content"] = all_act_elements if all_act_elements else ch_obj["content"]
        elif "params" in ch_obj and isinstance(ch_obj.get("params"), dict):
            if "content" in ch_obj["params"] and isinstance(ch_obj["params"]["content"], list):
                ch_obj["params"]["content"] = all_act_elements if all_act_elements else ch_obj["params"]["content"]
            else:
                ch_obj["params"]["content"] = all_act_elements
        else:
            ch_obj["content"] = all_act_elements

        # Remove any image blocks from the activity chapter
        img_blocks = [b for b in _iter_library_blocks(ch_obj) if str(b.get("library", "")).startswith("H5P.Image")]
        for ib in img_blocks:
            ib["params"] = {}

        # QA report entries
        for atype, questions in activity_groups.items():
            for qi, q_data in enumerate(questions):
                ev = q_data.get("evidence") or {}
                q_summary = q_data.get("statement") or q_data.get("sentence") or q_data.get("paragraph") or ""
                q_answer = q_data.get("correct_answer") or q_data.get("answer") or q_data.get("missing_word") or ""
                if isinstance(q_data.get("marked_words"), list):
                    q_answer = ", ".join(q_data["marked_words"])
                qa_items.append({
                    "label": f"Page {act_ch_num} — {atype} Q{qi + 1}",
                    "content": f"{q_summary}\nAnswer: {q_answer}",
                    "expected": q_answer,
                    "evidence": ev,
                })

    deep_find_set_first(content, ["description", "introduction", "taskDescription"], description)
    _save_json(work_dir, "content/content.json", content)
    return qa_items

def choose_representative_chunks(chunks: List[ContentChunk], max_pages: int = 18) -> List[ContentChunk]:
    """Reduce prompt size by sampling pages across PDFs to reduce token/min rate limits."""
    if not chunks:
        return []
    if len(chunks) <= max_pages:
        return chunks
    picked = []
    picked.extend(chunks[:5])
    remaining = max_pages - len(picked)
    if remaining <= 0:
        return picked[:max_pages]
    step = max(1, (len(chunks) - 5) // remaining)
    i = 5
    while len(picked) < max_pages and i < len(chunks):
        picked.append(chunks[i])
        i += step
    return picked[:max_pages]

def join_chunks_for_prompt(chunks: List[ContentChunk], max_chars: int = 22000) -> str:
    """Keep prompts short."""
    parts = [f"[{c.source_file} - {c.locator}]\n{c.text}" for c in chunks if c.text]
    return ("\n\n".join(parts))[:max_chars]


def deep_find_set_first(d: Any, key_candidates: List[str], new_value: Any) -> bool:
    if isinstance(d, dict):
        for k in key_candidates:
            if k in d:
                d[k] = new_value
                return True
        for v in d.values():
            if deep_find_set_first(v, key_candidates, new_value):
                return True
    elif isinstance(d, list):
        for v in d:
            if deep_find_set_first(v, key_candidates, new_value):
                return True
    return False


def deep_find_first_key(d: Any, key_candidates: List[str]) -> Optional[Tuple[str, Any]]:
    if isinstance(d, dict):
        for k in key_candidates:
            if k in d:
                return (k, d[k])
        for v in d.values():
            found = deep_find_first_key(v, key_candidates)
            if found:
                return found
    elif isinstance(d, list):
        for v in d:
            found = deep_find_first_key(v, key_candidates)
            if found:
                return found
    return None


def random_subcontent_id() -> str:
    return str(uuid.uuid4())


# ----------------------------
# OpenAI call with retry/backoff
# ----------------------------
def _parse_openai_error(resp: requests.Response) -> Tuple[str, str]:
    try:
        j = resp.json()
        err = (j or {}).get("error", {}) or {}
        code = err.get("code") or err.get("type") or "error"
        msg = err.get("message") or resp.text
        return str(code), str(msg)
    except Exception:
        return "error", resp.text

def call_openai_chat_json(system: str, user: str, model: str = "gpt-4.1-mini", temperature: float = 0.2) -> Dict[str, Any]:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set environment variable LLM_API_KEY.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
    }

    max_attempts = 7
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=240)

            if resp.status_code == 429:
                code, msg = _parse_openai_error(resp)
                if "insufficient_quota" in code or "quota" in msg.lower():
                    raise RuntimeError("OpenAI API quota/credits exhausted for this key. Add credits or use a different key.")
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(1.0, float(retry_after))
                    except Exception:
                        sleep_s = 5.0
                else:
                    sleep_s = min(40.0, (2 ** (attempt - 1))) + random.uniform(0, 0.8)
                if attempt == max_attempts:
                    raise RuntimeError("OpenAI API rate limit reached. Try again shortly, or reduce concurrency/requests.")
                time.sleep(sleep_s)
                continue

            if resp.status_code in (500, 502, 503, 504):
                sleep_s = min(30.0, (2 ** (attempt - 1))) + random.uniform(0, 0.8)
                if attempt == max_attempts:
                    code, msg = _parse_openai_error(resp)
                    raise RuntimeError(f"Temporary server error ({resp.status_code}). {msg[:200]}")
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)

        except requests.exceptions.RequestException as e:
            if attempt == max_attempts:
                raise RuntimeError(f"API request failed after retries. {str(e)}") from e
            sleep_s = min(30.0, (2 ** (attempt - 1))) + random.uniform(0, 0.8)
            time.sleep(sleep_s)

    raise RuntimeError("Unexpected: request loop ended without returning.")


# ----------------------------
# Suggestion + generators (data)
# ----------------------------
def llm_suggest_activities(chunks: List[ContentChunk], course_name: str, unit_name: str = "", qual_spec_text: str = "") -> Dict[str, Any]:
    system = "You are an instructional designer specialising in H5P. Return valid JSON only."
    src = join_chunks_for_prompt(chunks, max_chars=65000)

    unit_block = f"\nUnit name: {unit_name}\n" if unit_name else ""
    qual_block = f"\nQUALIFICATION SPECIFICATION:\n{qual_spec_text[:20000]}\n" if qual_spec_text.strip() else ""

    user = f"""
Recommend suitable H5P activity types for this course: {course_name}
{unit_block}
Consider these H5P types (exact labels):
{BEST_H5P_TYPES}

Return ONLY the best 8 recommendations.

Rules:
- Base suggestions on the source text and the qualification specification.
- Align suggestions to the unit name and learning outcomes where possible.
- Give a short practical reason.
- Give suggested_item_count (typical number of items/questions).
- Include one short exact quote with page reference.

JSON schema:
{{
  "recommendations": [
    {{
      "activity_type": "string",
      "score_0_to_5": 0,
      "why": "string",
      "suggested_item_count": 5,
      "evidence": {{"source_file":"string","locator":"Page X","quote":"short exact quote"}}
    }}
  ]
}}
{qual_block}
SOURCE TEXT:
{src}
""".strip()
    return call_openai_chat_json(system, user)


_BLANK_PATTERNS = [r"_{3,}", r"\[blank\]", r"\(\s*\)", r"\[\s*\]", r"……+", r"\.\.\.+"]


def _wrap_first_word_occurrence(text: str, word: str):
    pattern = r"\b" + re.escape(word) + r"\b"
    return re.subn(pattern, f"*{word}*", text, count=1)


def make_single_blank_markup(sentence: str, answer: str) -> str:
    s = (sentence or "").strip()
    a = (answer or "").strip()
    if not s or not a:
        raise ValueError("Sentence and answer cannot be empty.")
    new_s, n = _wrap_first_word_occurrence(s, a)
    if n:
        return new_s
    for bp in _BLANK_PATTERNS:
        if re.search(bp, s):
            return re.sub(bp, f"*{a}*", s, count=1)
    return f"{s} (*{a}*)"


def make_multiline_blocks(lines: List[str]) -> str:
    """Join multiple items into a single textField without numbering."""
    cleaned: List[str] = []
    for line in lines:
        line = (line or "").strip()
        if line:
            cleaned.append(line)
    return "\n\n".join(cleaned)

def make_dragtext_textfield(items: List[Dict[str, Any]]) -> str:
    return make_multiline_blocks([
        make_single_blank_markup(it.get("sentence", ""), it.get("missing_word", ""))
        for it in items
    ])

def make_blanks_textfield(items: List[Dict[str, Any]]) -> str:
    return make_multiline_blocks([
        make_single_blank_markup(it.get("sentence", ""), it.get("answer", ""))
        for it in items
    ])

def make_mark_words_textfield(items: List[Dict[str, Any]]) -> str:
    paragraphs = []
    for it in items:
        p = (it.get("paragraph") or "").strip()
        words = it.get("marked_words") or []
        for w in words:
            w = (w or "").strip()
            if not w:
                continue
            p2, n = _wrap_first_word_occurrence(p, w)
            p = p2 if n else f"{p} (*{w}*)"
        paragraphs.append(p)
    return "\n\n".join(paragraphs)


def call_llm_drag_words(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    system = "Create H5P Drag the Words strictly grounded in source text. Return JSON only."
    src = join_chunks_for_prompt(chunks)
    user = f"""
Create Drag the Words with {n} items.

Return JSON:
{{
 "title":"string",
 "description":"string",
 "overall_feedback":[
   {{"from":0,"to":40,"feedback":"string"}},
   {{"from":41,"to":80,"feedback":"string"}},
   {{"from":81,"to":100,"feedback":"string"}}
 ],
 "items":[
   {{
     "sentence":"string",
     "missing_word":"string",
     "distractors":["string","string"],
     "evidence":{{"source_file":"string","locator":"PDF p.X/Y","quote":"short exact quote"}}
   }}
 ]
}}

Rules:
- Keep answers/options concise (typically 1–6 words where applicable).
- sentence must include a blank marker like "____" where the missing word belongs (do NOT include the missing word in the sentence).
- missing_word must be 1–2 words.
- distractors should be plausible but incorrect 1–2 word options (2–4 per item).
- Everything must be directly supported by SOURCE.

Course: {course}
SOURCE:
{src}
""".strip()
    return call_openai_chat_json(system, user)

def call_llm_fill_blanks(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    system = "Create H5P Fill in the Blanks strictly grounded in source text. Return JSON only."
    src = join_chunks_for_prompt(chunks)
    user = f"""
Create Fill in the Blanks with {n} items.

JSON:
{{
 "title":"string","description":"string",
 "overall_feedback":[
   {{"from":0,"to":40,"feedback":"string"}},
   {{"from":41,"to":80,"feedback":"string"}},
   {{"from":81,"to":100,"feedback":"string"}}
 ],
 "items":[
   {{"sentence":"string","answer":"string","evidence":{{"source_file":"string","locator":"Page X","quote":"string"}}}}
 ]
}}

Course: {course}
Source:
{src}
""".strip()
    return call_openai_chat_json(system, user)


def call_llm_mark_words(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    system = "Create H5P Mark the Words strictly grounded in source text. Return JSON only."
    src = join_chunks_for_prompt(chunks)
    user = f"""
Create Mark the Words with {n} paragraphs. Each paragraph must include 3-6 marked_words that appear in the paragraph exactly.

JSON:
{{
 "title":"string","description":"string",
 "items":[
   {{"paragraph":"string","marked_words":["string"],"evidence":{{"source_file":"string","locator":"Page X","quote":"string"}}}}
 ]
}}

Course: {course}
Source:
{src}
""".strip()
    return call_openai_chat_json(system, user)


def call_llm_summary(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    system = "Create H5P Summary strictly grounded in source text. Return JSON only."
    src = join_chunks_for_prompt(chunks)
    user = f"""
Create an H5P Summary activity with EXACTLY {n} statement groups for course: {course}

Each group has ONE correct statement and 2 incorrect (but plausible) statements.
The correct statement must be a true fact from the SOURCE text.
The incorrect statements must sound plausible but be factually wrong based on the SOURCE.

Return JSON:
{{
  "title":"string",
  "description":"string",
  "groups":[
    {{
      "correct_statement":"a true statement taken directly from SOURCE",
      "incorrect_statements":["plausible but wrong statement 1","plausible but wrong statement 2"],
      "tip":"optional short hint",
      "evidence":{{"source_file":"string","locator":"Page X","quote":"exact quote from SOURCE supporting the correct statement"}}
    }}
  ]
}}

Rules:
1. EXACTLY {n} groups.
2. Every correct_statement MUST be grounded in a specific fact from SOURCE.
3. Incorrect statements should be related to the same topic but contain a wrong detail (e.g. swapped term, wrong number, reversed cause/effect).
4. Spread groups across different parts of the SOURCE — do not cluster from one section.
5. Each statement should be a complete, clear sentence.

SOURCE:
{src}
""".strip()
    return call_openai_chat_json(system, user)


# ----------------------------
# Essay: LLM generator + template builder
# ----------------------------
def call_llm_essay(chunks: List[ContentChunk], course: str) -> Dict[str, Any]:
    """Generate exactly 1 Essay question with keyword-based marking, strictly grounded in PDF source text.

    H5P Essay checks the learner's written response for specific keywords.
    Each keyword can have alternative acceptable forms (synonyms, abbreviations).
    The answer should be concise — not overly long — following the H5P Essay template format.
    All content must be 100% accurate and taken directly from the attached PDF(s).
    """
    system = (
        "You are a strict content extractor for essay-type questions. "
        "You ONLY use facts that appear VERBATIM in the SOURCE text. "
        "Every question, sample solution, keyword, and piece of information must be "
        "directly and exactly supported by the SOURCE — do NOT add, infer, rephrase, "
        "or embellish any facts. 100% accuracy is required. "
        "Return JSON only."
    )
    src = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Create exactly 1 Essay question for course: {course}

The essay question should:
- Ask the learner to explain, describe, or discuss a specific concept from the SOURCE
- Be a clear, focused question or instruction that the learner can respond to
- Have a concise model/sample answer (30–80 words MAX) drawn ONLY from the SOURCE text
- The sample answer must be SHORT and to-the-point — do NOT write a long paragraph
- Have 4–6 keywords that the system will check for in the learner's response
- Each keyword should have 1–2 alternative forms (synonyms, abbreviations)

CRITICAL ACCURACY RULES:
- Every single fact, term, number, and statement in the question AND sample answer
  MUST appear word-for-word in the SOURCE text.
- Do NOT add any information not found in the SOURCE.
- Do NOT rephrase or paraphrase SOURCE content — use the exact words from the PDF.
- The sample answer should be a concise summary using ONLY exact phrases from the SOURCE.
- If you cannot create a 100% accurate question from the SOURCE, return fewer keywords
  rather than inventing content.

Return JSON:
{{
  "title": "string (descriptive activity title from SOURCE content)",
  "description": "string (brief overall instruction, e.g. 'Read the question and write your answer below.')",
  "essays": [
    {{
      "taskDescription": "<p>The essay question/instruction. Be specific about what to include.</p>",
      "sampleSolution": "A concise model answer (30–80 words) using ONLY exact facts from SOURCE.",
      "keywords": [
        {{
          "keyword": "the primary keyword or key phrase from SOURCE",
          "alternatives": ["alt form 1"],
          "points": 1,
          "occurrences": 1
        }}
      ],
      "minimumLength": 30,
      "maximumLength": 300,
      "evidence": {{
        "source_file": "string",
        "locator": "Page X",
        "quote": "exact quote from SOURCE supporting this question and answer"
      }}
    }}
  ]
}}

Rules:
1. EXACTLY 1 essay question — no more.
2. taskDescription must clearly state what the learner should write about.
3. sampleSolution must be SHORT (30–80 words) and contain ONLY verbatim facts from SOURCE.
4. Keywords must be important terms that appear EXACTLY in both the SOURCE and sampleSolution.
5. Each keyword's alternatives should be reasonable variations (e.g., "GDPR" / "General Data Protection Regulation").
6. Points per keyword: 1 for common terms, 2 for critical/specific terms.
7. Keep taskDescription in simple HTML (<p> tags only).
8. Do NOT exceed 300 words for the maximum answer length.
9. The evidence quote must be copied character-for-character from SOURCE.

SOURCE:
{src}
""".strip()
    return call_openai_chat_json(system, user)


def _strip_html(s: str) -> str:
    """Remove HTML tags for plain-text display."""
    return re.sub(r"<[^>]+>", "", (s or "")).strip()


def _build_essay_params(essay: Dict[str, Any]) -> Dict[str, Any]:
    """Build the params dict for a single H5P.Essay content block."""
    task_desc = (essay.get("taskDescription") or "").strip()
    if not task_desc.startswith("<"):
        task_desc = f"<p>{task_desc}</p>"

    sample_solution = (essay.get("sampleSolution") or "").strip()
    min_len = int(essay.get("minimumLength", 30) or 30)
    max_len = int(essay.get("maximumLength", 300) or 300)

    # Build keywords array
    keywords = []
    for kw in (essay.get("keywords") or []):
        keyword_text = (kw.get("keyword") or "").strip()
        if not keyword_text:
            continue

        alternatives = []
        for alt in (kw.get("alternatives") or []):
            alt = (alt or "").strip()
            if alt and alt.lower() != keyword_text.lower():
                alternatives.append(alt)

        keywords.append({
            "keyword": keyword_text,
            "alternatives": alternatives,
            "options": {
                "points": int(kw.get("points", 1) or 1),
                "occurrences": int(kw.get("occurrences", 1) or 1),
                "caseSensitive": False,
                "forgiveMistakes": True,
            },
        })

    # Build overall feedback bands
    overall_feedback = [
        {"from": 0, "to": 25, "feedback": "You've made a start. Try to include more key concepts from the material."},
        {"from": 26, "to": 50, "feedback": "Good effort. Review the material and try to cover more of the key points."},
        {"from": 51, "to": 75, "feedback": "Well done! You've covered many important points. Check if you missed anything."},
        {"from": 76, "to": 100, "feedback": "Excellent! You've demonstrated a thorough understanding of the topic."},
    ]

    return {
        "taskDescription": task_desc,
        "solution": {
            "introduction": "Sample solution:",
            "sample": sample_solution,
        },
        "keywords": keywords,
        "overallFeedback": overall_feedback,
        "behaviour": {
            "minimumLength": min_len,
            "maximumLength": max_len,
            "inputFieldSize": "10",
            "enableRetry": True,
            "ignoreScoring": False,
            "pointsHost": 1,
        },
        "placeholderText": "Enter your answer here...",
        "checkAnswer": "Check",
        "tryAgain": "Retry",
        "showSolution": "Show solution",
    }


def _populate_essay_content(content: Dict[str, Any], essay: Dict[str, Any], description: str) -> None:
    """Populate a single-essay H5P content.json with the essay data."""
    params = _build_essay_params(essay)

    # Set task description
    deep_find_set_first(content, ["taskDescription"], params["taskDescription"])

    # Set solution
    if "solution" in content and isinstance(content.get("solution"), dict):
        content["solution"]["sample"] = params["solution"]["sample"]
        content["solution"]["introduction"] = params["solution"]["introduction"]
    else:
        if not deep_find_set_first(content, ["solution"], params["solution"]):
            content["solution"] = params["solution"]

    # Set keywords
    if not deep_find_set_first(content, ["keywords"], params["keywords"]):
        content["keywords"] = params["keywords"]

    # Set overall feedback
    deep_find_set_first(content, ["overallFeedback"], params["overallFeedback"])

    # Set behaviour
    if "behaviour" in content and isinstance(content["behaviour"], dict):
        content["behaviour"].update(params["behaviour"])
    else:
        if not deep_find_set_first(content, ["behaviour"], params["behaviour"]):
            content["behaviour"] = params["behaviour"]

    # Set placeholder text
    deep_find_set_first(content, ["placeholderText"], params.get("placeholderText", ""))


def update_essay_template(
    work_dir: str,
    title: str,
    description: str,
    essays: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Populate an H5P Essay template with generated question and keyword-based grading.

    Always populates a single essay (the first one) directly into the template.

    Args:
        work_dir:    Extracted H5P template directory.
        title:       Activity title.
        description: Overall task description.
        essays:      List of essay dicts from call_llm_essay() (always 1 essay).

    Returns:
        qa_items list for the QA evidence report.
    """
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    qa_items: List[Dict[str, Any]] = []

    if essays:
        _populate_essay_content(content, essays[0], description)

    # Set description (do NOT overwrite taskDescription — that holds the actual question)
    deep_find_set_first(content, ["introduction", "description"], description)

    _save_json(work_dir, "content/content.json", content)

    for i, e in enumerate(essays, start=1):
        ev = e.get("evidence") or {}
        kw_list = ", ".join(k.get("keyword", "") for k in (e.get("keywords") or []))
        qa_items.append({
            "label": f"Essay Q{i}",
            "content": f"Q: {_strip_html(e.get('taskDescription', ''))}\nSample: {e.get('sampleSolution', '')[:300]}",
            "expected": kw_list,
            "evidence": ev,
        })

    return qa_items


def call_llm_truefalse_statements(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    system = "Create True/False statements strictly grounded in source text. Return JSON only."
    src = join_chunks_for_prompt(chunks)
    user = f"""
Create {n} True/False statements grounded in the source.

JSON:
{{
 "title":"string","description":"string",
 "items":[
   {{"statement":"string","correctAnswer":true,"evidence":{{"source_file":"string","locator":"Page X","quote":"string"}}}}
 ]
}}

Course: {course}
Source:
{src}
""".strip()
    return call_openai_chat_json(system, user)


# ----------------------------
# Dictation: dedicated LLM + template builder with TTS audio
# ----------------------------
def call_llm_dictation(chunks: List[ContentChunk], n: int, course: str) -> Dict[str, Any]:
    """Generate short dictation sentences strictly grounded in the PDF source text."""
    system = (
        "You are a strict content extractor for dictation exercises. "
        "You ONLY use facts that appear in the SOURCE. "
        "Every sentence must be directly supported by the source text. "
        "Return JSON only."
    )
    src = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Create {n} short dictation sentences for course: {course}

Each sentence should be a short phrase or sentence (5–15 words) that a learner will
listen to and then type. The sentences must test key vocabulary and concepts from
the source material.

Return JSON:
{{
  "title": "string (a descriptive activity title)",
  "description": "string (brief task instruction, e.g. 'Listen carefully and type what you hear.')",
  "sentences": [
    {{
      "text": "The short sentence the learner must type.",
      "evidence": {{
        "source_file": "string",
        "locator": "Page X",
        "quote": "exact quote from SOURCE that supports this sentence"
      }}
    }}
  ]
}}

Rules:
1. EXACTLY {n} sentences.
2. Each sentence must be 5–15 words — short enough to remember after hearing once.
3. Use proper punctuation and capitalisation.
4. Spread sentences across different topics/sections of the SOURCE.
5. Every sentence MUST be grounded in a specific fact from the SOURCE.
6. Avoid overly complex vocabulary unless it is a key term from the source.
7. Do not repeat the same concept across multiple sentences.

SOURCE:
{src}
""".strip()
    return call_openai_chat_json(system, user)


def update_dictation_template(
    work_dir: str,
    title: str,
    description: str,
    sentences: List[Dict[str, Any]],
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """Build a Dictation H5P by generating TTS audio for each sentence.

    For each sentence:
      - Generates normal-speed audio via OpenAI TTS → content/audios/<name>.mp3
      - Generates slow-speed audio via OpenAI TTS   → content/audios/<name>_slow.mp3
      - Wires both into the content.json sentence entry

    Args:
        work_dir:          Extracted template directory.
        title:             Activity title.
        description:       Task description shown to learner.
        sentences:         List of dicts with at least "text" key.
        progress_callback: Optional callable(fraction, text) for UI updates.

    Returns:
        qa_items list for the QA evidence report.
    """
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    audios_dir = os.path.join(work_dir, "content", "audios")
    os.makedirs(audios_dir, exist_ok=True)

    # ── Build new sentence entries ──────────────────────────────────────
    new_sentences = []
    qa_items = []
    total = len(sentences)

    for i, sent in enumerate(sentences):
        text = (sent.get("text") or "").strip()
        if not text:
            continue

        stem = safe_filename(f"sentence_{i+1}", max_len=60)
        normal_fname = f"{stem}.mp3"
        slow_fname = f"{stem}_slow.mp3"
        normal_path = os.path.join(audios_dir, normal_fname)
        slow_path = os.path.join(audios_dir, slow_fname)

        if progress_callback:
            pct = int(65 + (i / max(total, 1)) * 20)
            progress_callback(min(pct, 85), f"Generating audio {i+1}/{total}: {text[:50]}...")

        # Normal-speed audio
        normal_ok = openai_tts_to_file(text, normal_path, speed=OPENAI_TTS_NORMAL_SPEED)
        # Slow-speed audio
        slow_ok = openai_tts_to_file(text, slow_path, speed=OPENAI_TTS_SLOW_SPEED)

        entry: Dict[str, Any] = {"text": text}

        if normal_ok:
            entry["sample"] = [{
                "path": f"audios/{normal_fname}",
                "mime": "audio/mpeg",
                "copyright": {"license": "U"},
            }]
        else:
            entry["sample"] = []

        if slow_ok:
            entry["sampleSlow"] = [{
                "path": f"audios/{slow_fname}",
                "mime": "audio/mpeg",
                "copyright": {"license": "U"},
            }]
        else:
            entry["sampleSlow"] = []

        new_sentences.append(entry)

        ev = sent.get("evidence") or {}
        qa_items.append({
            "label": f"Sentence {i+1}",
            "content": text,
            "evidence": ev,
        })

    # ── Inject into content.json ────────────────────────────────────────
    content["sentences"] = new_sentences

    # Set description / taskDescription
    deep_find_set_first(content, ["taskDescription", "description", "introduction"], description)

    # ── Remove any stale template audio files not referenced by new content ──
    referenced_files = set()
    for s in new_sentences:
        for key in ("sample", "sampleSlow"):
            for aud in (s.get(key) or []):
                p = aud.get("path", "")
                if p:
                    referenced_files.add(os.path.basename(p))

    for fname in os.listdir(audios_dir):
        if fname not in referenced_files:
            try:
                os.remove(os.path.join(audios_dir, fname))
            except Exception:
                pass

    _save_json(work_dir, "content/content.json", content)
    return qa_items


# ----------------------------
# Generic patcher (for any type with a template)
# ----------------------------
def _load_json(work_dir: str, rel_path: str) -> Dict[str, Any]:
    with open(os.path.join(work_dir, rel_path), "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(work_dir: str, rel_path: str, obj: Dict[str, Any]) -> None:
    with open(os.path.join(work_dir, rel_path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def update_h5p_title(work_dir: str, title: str) -> None:
    meta = _load_json(work_dir, "h5p.json")
    meta["title"] = title
    _save_json(work_dir, "h5p.json", meta)


def call_llm_generic_patch(
    chunks: List[ContentChunk],
    course_name: str,
    activity_type: str,
    template_h5p_json: Dict[str, Any],
    template_content_json: Dict[str, Any],
    item_count: int,
) -> Dict[str, Any]:
    system = "You generate valid H5P content strictly grounded in the provided source and template schema. Return JSON only."
    src = join_chunks_for_prompt(chunks, max_chars=85000)

    template_str = json.dumps(template_content_json, ensure_ascii=False)
    if len(template_str) > 85000:
        def shape(x, depth=0, max_depth=5):
            if depth >= max_depth:
                return "..."
            if isinstance(x, dict):
                return {k: shape(v, depth+1, max_depth) for k, v in list(x.items())[:60]}
            if isinstance(x, list):
                return [shape(x[0], depth+1, max_depth)] if x else []
            return type(x).__name__
        template_view = shape(template_content_json)
    else:
        template_view = template_content_json

    user = f"""
Create an H5P activity for: {course_name}
Target type label: {activity_type}

Template meta (h5p.json):
{json.dumps(template_h5p_json, ensure_ascii=False, indent=2)[:25000]}

Template content schema (content/content.json):
{json.dumps(template_view, ensure_ascii=False, indent=2)[:55000]}

Task:
- Produce a COMPLETE patched content/content.json object compatible with this template.
- Create about {item_count} meaningful items appropriate to the type.
- Keep it simple.
- Do not invent facts; every item must be supported by the source.
- If the template uses subContentId (or similar), generate UUID-like values.

Output JSON:
{{
  "title":"string",
  "description":"string",
  "patched_content_json": {{ ... }},
  "qa_items":[
    {{"label":"string","content":"string","evidence":{{"source_file":"string","locator":"Page X","quote":"string"}}}}
  ]
}}

SOURCE TEXT:
{src}
""".strip()

    data = call_openai_chat_json(system, user)
    if not isinstance(data.get("patched_content_json"), dict):
        raise RuntimeError("Model did not return patched content.")
    if not isinstance(data.get("qa_items", []), list):
        data["qa_items"] = []
    data.setdefault("title", f"{activity_type} - {course_name}")
    data.setdefault("description", f"Auto-generated {activity_type}.")
    return data


# ----------------------------
# Template updaters
# ----------------------------
def update_text_based_template(work_dir: str, title: str, description: str, textfield: str, overall_feedback=None, textfield_keys=None) -> None:
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")
    deep_find_set_first(content, ["taskDescription", "introduction", "description", "instructions"], description)

    keys = textfield_keys or ["textField", "text", "questionText", "content"]
    if not deep_find_set_first(content, keys, textfield):
        found = deep_find_first_key(content, keys)
        raise KeyError(f"Template missing a text field. Nearest match: {found}")

    if overall_feedback is not None:
        deep_find_set_first(content, ["overallFeedback"], overall_feedback)

    _save_json(work_dir, "content/content.json", content)


def maybe_set_distractors(work_dir: str, distractors: List[str]) -> None:
    """If the template supports distractors, set them (H5P Drag the Words)."""
    if not distractors:
        return
    content = _load_json(work_dir, "content/content.json")
    uniq = []
    seen = set()
    for d in distractors:
        d = (d or "").strip()
        if not d:
            continue
        key = d.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    # Common keys used by DragText
    deep_find_set_first(content, ["distractors", "distractor"], "\n".join(uniq))
    _save_json(work_dir, "content/content.json", content)



def call_llm_cornell_notes(chunks: List[ContentChunk], course: str) -> Dict[str, Any]:
    """Generate Cornell Notes instructional text from PDF content."""
    system = "Create H5P Cornell Notes content strictly grounded in source text. Return JSON only."
    src = join_chunks_for_prompt(chunks, max_chars=30000)
    user = f"""
Create Cornell Notes content for course: {course}

Return JSON:
{{
  "title": "Cornell Notes - <topic from source>",
  "body": "2-3 sentence instruction telling learners what to watch and what to note down",
  "cue_placeholder": "short helpful prompt for the Cue column",
  "notes_placeholder": "short helpful prompt for the Notes column",
  "summary_placeholder": "short prompt for the Summary box"
}}

Course: {course}
Source:
{src}
""".strip()
    return call_openai_chat_json(system, user)


def _cornell_mime(url: str) -> str:
    u = (url or "").lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "video/YouTube"
    if "vimeo.com" in u:
        return "video/Vimeo"
    return "video/YouTube"


def _normalise_video_url(url: str) -> str:
    """Strip /video/ from Vimeo URLs so H5P recognises them.
    https://vimeo.com/video/123 → https://vimeo.com/123
    """
    import re as _re
    m = _re.search(r"vimeo\.com(?:/video)?/(\d+)", url or "")
    if m:
        return f"https://vimeo.com/{m.group(1)}"
    return url


def _deep_set_video_sources(node: Any, sources: list) -> bool:
    """Recursively walk node and replace EVERY 'sources' list that looks like
    H5P video sources (contains dicts with a 'path' key).
    Returns True if at least one replacement was made.
    """
    replaced = False
    if isinstance(node, dict):
        if "sources" in node and isinstance(node["sources"], list):
            # Check it looks like a video sources list
            current = node["sources"]
            if not current or (isinstance(current[0], dict) and "path" in current[0]):
                node["sources"] = sources
                replaced = True
        for v in node.values():
            if _deep_set_video_sources(v, sources):
                replaced = True
    elif isinstance(node, list):
        for item in node:
            if _deep_set_video_sources(item, sources):
                replaced = True
    return replaced


def update_cornell_notes_template(
    work_dir: str,
    title: str,
    video_url: str,
    gen_data: Optional[Dict[str, Any]] = None,
    poster_image_bytes: Optional[bytes] = None,
    poster_image_ext: str = "jpg",
) -> dict:
    """Completely rewrite the video source in a Cornell Notes H5P template.

    Strategy: rather than guessing the JSON shape, we:
      1. Read the real content.json
      2. Replace the video sub-content sources at EVERY location in the tree
      3. If nothing was replaced, force-write the top-level 'video' key
         using the full H5P.Video sub-content structure
    Returns the patched content dict (useful for debug display).
    """
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    if video_url:
        url = _normalise_video_url(video_url.strip())
        mime = _cornell_mime(url)

        new_sources = [{
            "path": url,
            "mime": mime,
            "copyright": {"license": "U"},
            "aspectRatio": "16:9",
        }]

        # ── Strategy A: deep-replace every 'sources' list in the tree ───────
        replaced = _deep_set_video_sources(content, new_sources)

        # ── Strategy B: handle plain-string "video" key ───────────────────
        if isinstance(content.get("video"), str):
            content["video"] = url
            replaced = True

        # ── Strategy C: handle "video" as flat list of source objects ────
        if isinstance(content.get("video"), list):
            lst = content["video"]
            if lst and isinstance(lst[0], dict) and "path" in lst[0]:
                lst[0]["path"] = url
                lst[0]["mime"] = mime
                replaced = True
            elif not lst:
                content["video"] = new_sources
                replaced = True

        # ── Strategy D: force-write full H5P.Video sub-content ──────────
        if not replaced:
            content["video"] = {
                "params": {
                    "visuals": {"fit": True, "controls": True},
                    "playback": {"autoplay": False, "loop": False},
                    "sources": new_sources,
                },
                "library": "H5P.Video 1.6",
                "metadata": {"contentType": "Video", "license": "U", "title": title},
                "subContentId": str(uuid.uuid4()),
            }

        # ── Poster image ─────────────────────────────────────────────────
        if poster_image_bytes:
            img_dir = os.path.join(work_dir, "content", "images")
            os.makedirs(img_dir, exist_ok=True)
            ext = (poster_image_ext or "jpg").lower()
            img_filename = f"poster.{ext}"
            with open(os.path.join(img_dir, img_filename), "wb") as f:
                f.write(poster_image_bytes)
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
            poster_obj = {
                "path": f"images/{img_filename}",
                "mime": mime_map.get(ext, "image/jpeg"),
                "copyright": {"license": "U"},
                "width": 1280, "height": 720,
            }
            vid = content.get("video")
            if isinstance(vid, dict):
                vid.setdefault("params", {}).setdefault("visuals", {})["posterImage"] = poster_obj

    # ── Body / cue / notes / summary text ─────────────────────────────────
    if gen_data:
        _txt_map = [
            ("body",             ["body", "taskDescription", "introduction", "description"]),
            ("cue_placeholder",  ["cue", "cuePlaceholder", "cueText"]),
            ("notes_placeholder",["notes", "notesPlaceholder", "notesText"]),
            ("summary_placeholder", ["summary", "summaryPlaceholder", "summaryText"]),
        ]
        for gen_key, content_keys in _txt_map:
            val = gen_data.get(gen_key)
            if val:
                for ck in content_keys:
                    if ck in content:
                        content[ck] = val
                        break

    _save_json(work_dir, "content/content.json", content)
    return content


def update_summary_template(work_dir: str, title: str, description: str, groups: List[Dict[str, Any]]) -> None:
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")
    deep_find_set_first(content, ["taskDescription", "introduction", "description", "instructions"], description)

    # H5P Summary expects each group's "summary" to be a list of strings.
    # The FIRST string is the correct answer; the rest are incorrect options.
    summary_objs = []
    for grp in groups:
        correct = (grp.get("correct_statement") or "").strip()
        incorrects = grp.get("incorrect_statements") or []
        # Ensure we have strings
        incorrects = [(s or "").strip() for s in incorrects if (s or "").strip()]
        if not correct:
            continue
        # Build the statement list: correct first, then incorrects
        statements = [correct] + incorrects
        summary_objs.append({
            "subContentId": random_subcontent_id(),
            "tip": (grp.get("tip") or "").strip(),
            "summary": statements,
        })

    if not summary_objs:
        raise ValueError("No valid summary groups were generated.")

    if not deep_find_set_first(content, ["summaries", "summary", "items"], summary_objs):
        found = deep_find_first_key(content, ["summaries", "summary", "items"])
        raise KeyError(f"Template missing summaries/items field. Nearest match: {found}")

    _save_json(work_dir, "content/content.json", content)


def build_question_set_truefalse(work_dir: str, title: str, description: str, tf_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    deep_find_set_first(content, ["introPage", "introduction", "taskDescription"], {
        "showIntroPage": True,
        "title": title,
        "introduction": description
    })

    def find_questions_ref(d: Any) -> Optional[List]:
        if isinstance(d, dict):
            if "questions" in d and isinstance(d["questions"], list):
                return d["questions"]
            for v in d.values():
                q = find_questions_ref(v)
                if q is not None:
                    return q
        elif isinstance(d, list):
            for v in d:
                q = find_questions_ref(v)
                if q is not None:
                    return q
        return None

    questions_container = find_questions_ref(content)
    if questions_container is None:
        raise KeyError("Question Set template missing a 'questions' array. Make a blank Question Set with one sample question, export it, and use as template.")

    new_questions = []
    for it in tf_items:
        new_questions.append({
            "library": "H5P.TrueFalse 1.8",
            "subContentId": random_subcontent_id(),
            "params": {
                "question": it.get("statement", ""),
                "correctAnswer": bool(it.get("correctAnswer", True)),
                "feedbackCorrect": {"text": "Correct."},
                "feedbackIncorrect": {"text": "Incorrect."},
                "behaviour": {"enableRetry": True, "enableSolutionsButton": True, "autoCheck": False},
                "l10n": {"checkAnswer": "Check", "showSolutionButton": "Show solution", "tryAgainButton": "Retry"},
            },
            "metadata": {"title": "True/False", "license": "U"}
        })

    questions_container[:] = new_questions
    _save_json(work_dir, "content/content.json", content)

    qa = []
    for i, it in enumerate(tf_items, start=1):
        qa.append({"label": f"Q{i}", "content": f"{it.get('statement','')} (answer: {it.get('correctAnswer')})", "evidence": it.get("evidence", {})})
    return qa


def write_qa_report_html(path: str, title: str, activity_type: str, qa_items: List[Dict[str, Any]]) -> None:
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def tokens(s: str) -> List[str]:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        t = [w for w in s.split() if len(w) >= 4]
        stop = {"this","that","with","from","into","your","have","will","must","should","also","such","than","then","when","where","which","what","over","under","between","within","about"}
        return [w for w in t if w not in stop]

    def content_supported(content: str, quote: str) -> bool:
        a = set(tokens(content))
        b = set(tokens(quote))
        if not a or not b:
            return False
        overlap = len(a & b)
        return overlap >= max(2, int(0.25 * min(len(a), len(b))))

    def expected_in_quote(expected: str, quote: str) -> bool:
        exp = (expected or "").strip()
        if not exp:
            return False
        return re.search(r"\b" + re.escape(exp) + r"\b", quote or "", re.IGNORECASE) is not None

    def item_status(it: Dict[str, Any]) -> str:
        ev = it.get("evidence", {}) or {}
        quote = ev.get("quote", "") or ""
        expected = (it.get("expected") or "").strip()
        content = (it.get("content") or "").strip()
        if expected:
            return "Match" if expected_in_quote(expected, quote) else "No match"
        return "Match" if content_supported(content, quote) else "Needs review"

    statuses = [item_status(it) for it in qa_items]
    total = len(statuses)
    match_count = sum(1 for s in statuses if s == "Match")
    no_match_count = sum(1 for s in statuses if s == "No match")
    review_count = sum(1 for s in statuses if s == "Needs review")
    overall = "Match" if (total > 0 and match_count == total) else "Not fully matched"

    rows = []
    for it, stt in zip(qa_items, statuses):
        ev = it.get("evidence", {}) or {}
        expected = (it.get("expected") or "").strip()
        quote = ev.get("quote", "") or ""
        rows.append(
            f"<div style='padding:12px;border:1px solid #e7e7e7;border-radius:10px;margin:10px 0;'>"
            f"<div style='font-weight:600'>{esc(it.get('label','Item'))}</div>"
            f"<div style='margin-top:6px'><b>Item:</b> {esc(it.get('content',''))}</div>"
            + (f"<div style='margin-top:6px'><b>Expected answer:</b> {esc(expected)}</div>" if expected else "")
            + f"<div style='margin-top:6px'><b>Source in PDF:</b> {esc(ev.get('source_file',''))} — {esc(ev.get('locator',''))}</div>"
            + f"<div style='margin-top:6px'><b>Relevant text (PDF):</b> <i>{esc(quote)}</i></div>"
            + f"<div style='margin-top:6px'><b>Status:</b> {esc(stt)}</div>"
            + f"</div>"
        )

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)} - QA</title></head>
<body style='font-family:Arial, sans-serif;max-width:960px;margin:24px auto;'>
<h2 style='margin-bottom:4px'>{esc(title)}</h2>
<div style='color:#666;margin-bottom:16px'>
  <div><b>Type:</b> {esc(activity_type)}</div>
</div>

<div style='padding:12px;border:1px solid #d7d7d7;border-radius:10px;background:#fafafa;margin:14px 0;'>
  <div style='font-weight:700'>Overall report</div>
  <div style='margin-top:6px'><b>Overall status:</b> {esc(overall)}</div>
  <div style='margin-top:6px'><b>Total items:</b> {total}</div>
  <div style='margin-top:6px'><b>Matches:</b> {match_count} &nbsp;&nbsp; <b>No match:</b> {no_match_count} &nbsp;&nbsp; <b>Needs review:</b> {review_count}</div>
</div>

<p><b>Evidence per item (source page references and supporting text):</b></p>
{''.join(rows) if rows else '<p>No QA items.</p>'}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# =========================

# UI (very simple)
# =========================
st.set_page_config(page_title="H5P Activity Generator", layout="centered")

st.markdown(
    """
    <style>
    /* Primary buttons */
    div.stButton > button, div.stDownloadButton > button, button[kind="primary"] {
        background-color: #4b70fb !important;
        border: 1px solid #4b70fb !important;
        color: white !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover, button[kind="primary"]:hover {
        filter: brightness(0.95);
        border: 1px solid #4b70fb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("## H5P Activity Generator")
st.caption("Fill in all required fields (*) → Get suggestions → Select one type → Generate H5P")

templates = discover_templates("templates")

# Keep state keys
st.session_state.setdefault("pdf_fingerprints", None)
st.session_state.setdefault("chunks_cache", None)
st.session_state.setdefault("pdf_bytes_map", None)
st.session_state.setdefault("pdf_headings_cache", None)
st.session_state.setdefault("pdf_keywords_cache", None)
st.session_state.setdefault("suggestions_cache_key", None)
st.session_state.setdefault("suggestions", None)
st.session_state.setdefault("busy", False)
st.session_state.setdefault("qual_spec_text", "")

# Persist latest outputs across Streamlit reruns (download buttons remain visible)
st.session_state.setdefault("last_h5p_bytes", None)
st.session_state.setdefault("last_h5p_name", None)
st.session_state.setdefault("last_qa_bytes", None)
st.session_state.setdefault("last_qa_name", None)

uploads = st.file_uploader("Upload PDF file(s) *", type=["pdf"], accept_multiple_files=True)
course_name = st.text_input("Course name *", placeholder="e.g., Level 5 Diploma in ...")
unit_name = st.text_input("Unit name *", placeholder="e.g., Unit 1: Personal Development")
qual_spec_file = st.file_uploader("Qualification specification file *", type=["pdf"], accept_multiple_files=False, key="qual_spec")

def compute_inputs_key(files: List[Any], course: str, unit: str = "", qual_file: Any = None) -> str:
    parts = [course.strip(), unit.strip()]
    for f in files:
        b = f.getvalue()
        parts.append(f.name)
        parts.append(file_sha256(b))
    if qual_file is not None:
        parts.append(qual_file.name)
        parts.append(file_sha256(qual_file.getvalue()))
    return hashlib.sha256(("|".join(parts)).encode("utf-8")).hexdigest()

def ensure_chunks(files: List[Any]) -> List[ContentChunk]:
    fps = [(f.name, file_sha256(f.getvalue())) for f in files]
    if (
        st.session_state.get("pdf_fingerprints") == fps
        and st.session_state.get("chunks_cache") is not None
        and st.session_state.get("pdf_bytes_map") is not None
    ):
        return st.session_state["chunks_cache"]

    chunks: List[ContentChunk] = []
    pdf_map: Dict[str, bytes] = {}
    headings: List[str] = []
    term_freq: Dict[str, int] = {}

    for f in files:
        b = f.getvalue()
        pdf_map[f.name] = b

        file_chunks = extract_pdf_chunks_from_bytes(f.name, b)
        chunks.extend(file_chunks)

        # Headings help image searches feel "on topic"
        headings.extend(extract_pdf_headings_from_bytes(f.name, b))

        # Keywords (frequency-based, across PDFs) for better Freepik queries
        for ch in file_chunks:
            for t in _terms(ch.text):
                term_freq[t] = term_freq.get(t, 0) + 1

    if not chunks:
        raise RuntimeError("No readable text found in the uploaded PDF(s). (If PDFs are scanned images, use OCR PDFs.)")

    # Keep a compact list of high-signal keywords (used for image searches)
    sorted_terms = sorted(term_freq.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    keywords = [t for t, _ in sorted_terms[:40]]

    st.session_state["pdf_fingerprints"] = fps
    st.session_state["chunks_cache"] = chunks
    st.session_state["pdf_bytes_map"] = pdf_map
    st.session_state["pdf_headings_cache"] = headings
    st.session_state["pdf_keywords_cache"] = keywords
    return chunks

colA, colB = st.columns(2)
with colA:
    suggest_clicked = st.button("Suggest H5P types", use_container_width=True, disabled=st.session_state["busy"])
with colB:
    clear_clicked = st.button("Clear", use_container_width=True, disabled=st.session_state["busy"])

# Placeholder for suggest progress bar (appears directly under the buttons)
suggest_progress_area = st.empty()

if clear_clicked:
    st.session_state["pdf_fingerprints"] = None
    st.session_state["chunks_cache"] = None
    st.session_state["suggestions_cache_key"] = None
    st.session_state["suggestions"] = None
    st.session_state["pdf_bytes_map"] = None
    st.session_state["pdf_headings_cache"] = None
    st.session_state["pdf_keywords_cache"] = None
    st.session_state["last_h5p_bytes"] = None
    st.session_state["last_h5p_name"] = None
    st.session_state["last_qa_bytes"] = None
    st.session_state["last_qa_name"] = None
    st.session_state["qual_spec_text"] = ""
    st.session_state["busy"] = False
    st.rerun()

if suggest_clicked:
    try:
        st.session_state["busy"] = True

        if not uploads:
            st.warning("Please upload at least one PDF.")
            st.stop()
        if not (course_name or "").strip():
            st.warning("Please enter the course name.")
            st.stop()
        if not (unit_name or "").strip():
            st.warning("Please enter the unit name.")
            st.stop()
        if qual_spec_file is None:
            st.warning("Please upload a qualification specification file.")
            st.stop()
        if not os.environ.get("LLM_API_KEY"):
            st.error("Missing API key. Set LLM_API_KEY.")
            st.stop()

        # Show progress bar under the button
        _suggest_bar = suggest_progress_area.progress(0, text="Validating inputs...")
        time.sleep(0.3)

        # Extract qualification spec text for LLM context
        _suggest_bar.progress(10, text="Reading qualification specification...")
        qual_spec_text = ""
        try:
            qual_spec_chunks = extract_pdf_chunks_from_bytes(qual_spec_file.name, qual_spec_file.getvalue())
            qual_spec_text = join_chunks_for_prompt(qual_spec_chunks, max_chars=30000)
        except Exception:
            st.warning("Could not extract text from qualification specification file. Proceeding without it.")
        st.session_state["qual_spec_text"] = qual_spec_text

        key = compute_inputs_key(uploads, course_name, unit_name, qual_spec_file)
        if st.session_state["suggestions_cache_key"] == key and st.session_state["suggestions"] is not None:
            _suggest_bar.progress(100, text="Done ✓")
            time.sleep(0.5)
            suggest_progress_area.empty()
        else:
            _suggest_bar.progress(25, text="Extracting PDF content...")
            chunks = ensure_chunks(uploads)
            chunks_small = choose_representative_chunks(chunks, max_pages=18)
            _suggest_bar.progress(40, text="Analysing content with AI...")
            s = llm_suggest_activities(chunks_small, course_name.strip(), unit_name.strip(), qual_spec_text)
            _suggest_bar.progress(90, text="Finalising suggestions...")
            st.session_state["suggestions"] = s
            st.session_state["suggestions_cache_key"] = key
            _suggest_bar.progress(100, text="Done ✓")
            time.sleep(0.5)
            suggest_progress_area.empty()

    except Exception as e:
        suggest_progress_area.empty()
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg:
            st.error(msg)
        else:
            st.error(msg)
    finally:
        st.session_state["busy"] = False

        # Show suggestions and generation
if st.session_state["suggestions"]:
    recs = (st.session_state["suggestions"].get("recommendations") or [])
    allowed = set(BEST_H5P_TYPES)
    recs = [r for r in recs if isinstance(r, dict) and (r.get("activity_type") in allowed)]

    if not recs:
        st.info("No suggestions returned. Try again.")
        st.stop()

    st.markdown("---")
    st.markdown("### Choose one suggested type")

    other_label = "Other (choose from templates)"

    # Make a compact radio list with template availability (plus an "Other" option)
    options: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}

    for r in recs:
        typ = r.get("activity_type", "")
        score = int(r.get("score_0_to_5", 0) or 0)
        why = (r.get("why") or "").strip()
        suggested_n = int(r.get("suggested_item_count", 5) or 5)
        ev = r.get("evidence", {}) or {}

        template_ok = (typ in templates) if typ not in ("Quiz", "Multiple Choice") else ("Quiz" in templates)
        status = "" if template_ok else " (Missing template)"
        label = f"{typ}{status}"

        options.append(label)
        meta[label] = {
            "type": typ,
            "why": why,
            "n": suggested_n,
            "ev": ev,
            "template_ok": template_ok,
            "score": score,
        }

    options.append(other_label)
    meta[other_label] = {
        "type": "__OTHER__",
        "why": "",
        "n": 4,
        "ev": {},
        "template_ok": True,
        "score": 0,
    }

    choice = st.radio("Suggested types", options=options, index=0)
    chosen = meta[choice]

    # Resolve selected type (including Other)
    resolved_type = chosen["type"]

    if resolved_type == "__OTHER__":
        st.markdown("#### Other type")

        template_labels = sorted(list(templates.keys()))
        resolved_type = st.selectbox(
            "Pick an available template",
            options=template_labels,
            index=0
        )

        st.caption(f"Using template: **{resolved_type}**")

    # Apply resolved type back to chosen and re-check template availability
    chosen["type"] = resolved_type
    chosen["template_ok"] = (resolved_type in templates) if resolved_type not in ("Quiz", "Multiple Choice") else ("Quiz" in templates)

    st.markdown("---")
    st.markdown("### Generate H5P")

    cornell_video_url = ""
    cornell_poster_bytes: Optional[bytes] = None
    cornell_poster_ext: str = "jpg"

    if chosen["type"] == "Cornell Notes":
        st.markdown("#### Cornell Notes — Video URL")
        cornell_video_url = st.text_input(
            "Vimeo / YouTube video URL",
            placeholder="https://vimeo.com/123456789",
            help="Plain video page URL. Do NOT paste embed/iframe URLs.",
        )
        if cornell_video_url:
            import re as _re2
            _m = _re2.search(r"vimeo\.com(?:/video)?/(\d+)", cornell_video_url)
            if _m:
                _preview = f"https://vimeo.com/{_m.group(1)}"
            else:
                _preview = cornell_video_url
            if "player.vimeo.com" in cornell_video_url or "/embed/" in cornell_video_url:
                st.warning("\u26a0\ufe0f Embed URL detected — please use the plain page URL.")
            else:
                st.success(f"\u2705 Will use: `{_preview}`")
        else:
            st.info("\u2139\ufe0f No URL entered — template default video will be kept.")

        st.markdown("#### Poster Image (optional)")
        poster_upload = st.file_uploader(
            "Upload cover / poster image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Shown as the video thumbnail. Tip: export your PDF cover page as an image.",
        )
        if poster_upload is not None:
            cornell_poster_bytes = poster_upload.read()
            cornell_poster_ext = poster_upload.name.rsplit(".", 1)[-1].lower()
            st.image(cornell_poster_bytes, caption="Poster preview", use_container_width=True)

        n_items = 1
        cp_n_slides = None
        cp_activity_types = []
        cp_n_questions = 0
        ib_n_pages = None
        ib_activity_types = []
        ib_n_questions = 0

    # --- Course Presentation specific options ---
    elif chosen["type"] == "Course Presentation":
        # Slide-count limits: 1 PDF → max 8 total (7 content + 1 activity)
        #                     multiple PDFs → max 12 total (11 content + 1 activity)
        n_pdfs = len(uploads) if uploads else 1
        if n_pdfs == 1:
            max_content_slides = 7
            help_note = "Max 8 slides total (7 content + 1 activity) for a single PDF."
        else:
            max_content_slides = 11
            help_note = "Max 12 slides total (11 content + 1 activity) for multiple PDFs."

        default_slides = min(max_content_slides, max(3, chosen["n"]))

        cp_n_slides = st.number_input(
            "Number of content slides",
            min_value=3,
            max_value=max_content_slides,
            value=default_slides,
            step=1,
            help=help_note,
        )

        cp_activity_type = st.selectbox(
            "Activity type for the last slide",
            options=["Drag the Words", "Fill in the Blanks"],
            index=0,
            help="Select one activity type for the last slide.",
        )
        cp_activity_types = [cp_activity_type] if cp_activity_type else []

        cp_n_questions = st.number_input(
            "Number of questions (in the activity slide)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Total questions for the selected activity type on the last slide.",
        )

        st.caption(f"Total slides: **{int(cp_n_slides) + 1}** ({int(cp_n_slides)} content + 1 activity)")

        n_items = cp_n_slides  # n_items drives the rest of the pipeline
        ib_n_pages = None
        ib_activity_types = []
        ib_n_questions = 0

    # --- Interactive Book specific options ---
    elif chosen["type"] == "Interactive Book":
        # Page-count limits: 1 PDF → max 8 total (7 content + 1 activity)
        #                    multiple PDFs → max 12 total (11 content + 1 activity)
        n_pdfs = len(uploads) if uploads else 1
        if n_pdfs == 1:
            max_content_pages = 7
            help_note = "Max 8 pages total (7 content + 1 activity) for a single PDF."
        else:
            max_content_pages = 11
            help_note = "Max 12 pages total (11 content + 1 activity) for multiple PDFs."

        default_pages = min(max_content_pages, max(3, chosen["n"]))

        ib_n_pages = st.number_input(
            "Number of content pages",
            min_value=3,
            max_value=max_content_pages,
            value=default_pages,
            step=1,
            help=help_note,
        )

        ib_activity_type = st.selectbox(
            "Activity type for the last page",
            options=["Drag the Words", "Fill in the Blanks"],
            index=0,
            help="Select one activity type for the last page.",
        )
        ib_activity_types = [ib_activity_type] if ib_activity_type else []

        ib_n_questions = st.number_input(
            "Number of questions (in the activity page)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Total questions for the selected activity type on the last page.",
        )

        st.caption(f"Total pages: **{int(ib_n_pages) + 1}** ({int(ib_n_pages)} content + 1 activity)")

        n_items = ib_n_pages  # n_items drives the rest of the pipeline
        cp_n_slides = None
        cp_activity_types = []
        cp_n_questions = 0

    else:
        cp_n_slides = None
        cp_activity_types = []
        cp_n_questions = 0
        ib_n_pages = None
        ib_activity_types = []
        ib_n_questions = 0

        # ── ESSAY: always 1 question, no number input needed ──
        if chosen["type"] == "Essay":
            n_items = 1
            st.info("Essay generates a single question/instruction for the learner to respond to. All content is taken directly from the uploaded PDF(s).")

        # Enforce question limits for selected types
        elif chosen["type"] in LIMITED_Q_TYPES:
            n_pdfs = len(uploads) if uploads else 1
            max_q = LIMITED_Q_MAX_SINGLE_PDF if n_pdfs == 1 else LIMITED_Q_MAX_MULTI_PDF
            default_n = LIMITED_Q_MIN  # default display should start at 4
            n_items = st.number_input(
                "Number of items/questions",
                min_value=LIMITED_Q_MIN,
                max_value=int(max_q),
                value=int(default_n),
                step=1,
                key=f"n_items_limited_{chosen['type']}_{n_pdfs}",
                help=f"Allowed range: {LIMITED_Q_MIN}–{int(max_q)} ({n_pdfs} PDF{'s' if n_pdfs != 1 else ''}).",
            )
        else:
            default_n = max(5, chosen["n"]) if chosen["type"] in ("Quiz", "Multiple Choice") else max(3, chosen["n"])

            # Dialog Cards: keep a tight range (3–5) to avoid low-quality/duplicated cards
            if chosen["type"] == "Dialog Cards":
                default_cards = int(min(5, max(3, default_n)))
                n_items = st.number_input("Number of cards", min_value=3, max_value=5, value=default_cards, step=1)
            else:
                n_items = st.number_input("Number of items/questions", min_value=3, max_value=30, value=int(default_n), step=1)

    gen = st.button("Generate H5P file", type="primary", use_container_width=True, disabled=st.session_state["busy"])

    # Placeholder for generate progress bar (appears directly under the button)
    gen_progress_area = st.empty()

    if gen:
        try:
            st.session_state["busy"] = True

            if not chosen["template_ok"]:
                if chosen["type"] in ("Quiz", "Multiple Choice"):
                    st.error("Missing template: templates/Quiz.h5p (required for Question Set generation).")
                else:
                    st.error(f"Missing template: templates/{chosen['type']}.h5p")
                st.stop()

            _gen_bar = gen_progress_area.progress(0, text="Preparing content...")
            time.sleep(0.3)

            _gen_bar.progress(10, text="Extracting PDF content...")
            chunks = ensure_chunks(uploads)

            # Build enriched course context for LLM prompts
            _gen_bar.progress(15, text="Building course context...")
            _qs_text = st.session_state.get("qual_spec_text", "")
            _course_label = course_name.strip()
            _unit_label = (unit_name or "").strip()
            _context_parts = [_course_label]
            if _unit_label:
                _context_parts.append(f"Unit: {_unit_label}")
            if _qs_text:
                _context_parts.append(f"Qualification Specification excerpt:\n{_qs_text[:12000]}")
            enriched_course = "\n".join(_context_parts)

            with tempfile.TemporaryDirectory() as tmp:
                typ = chosen["type"]
                run_n = int(n_items)

                # Enforce question limits for selected types
                if typ in LIMITED_Q_TYPES:
                    n_pdfs = len(uploads) if uploads else 1
                    max_q = LIMITED_Q_MAX_SINGLE_PDF if n_pdfs == 1 else LIMITED_Q_MAX_MULTI_PDF
                    run_n = max(LIMITED_Q_MIN, min(run_n, int(max_q)))

                _gen_bar.progress(25, text=f"Generating {typ} content with AI...")

                if typ == "Quiz":
                    tf = call_llm_truefalse_statements(chunks, run_n, enriched_course)
                    _gen_bar.progress(65, text="AI content generated — building template...")

                    qs_dir = os.path.join(tmp, "_work_qs_tf")
                    unzip_h5p(templates["Quiz"], qs_dir)

                    title = tf.get("title", f"True/False Quiz - {course_name.strip()}")
                    desc = tf.get("description", "Answer the True/False questions.")
                    qa_items = build_question_set_truefalse(qs_dir, title, desc, tf.get("items", []))

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(qs_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, "Quiz (Question Set) — True/False", qa_items)

                elif typ == "Multiple Choice":
                    mc = call_llm_multichoice_questions(chunks, run_n, enriched_course)
                    _gen_bar.progress(65, text="AI content generated — building template...")

                    qs_dir = os.path.join(tmp, "_work_qs_mc")
                    unzip_h5p(templates["Quiz"], qs_dir)

                    title = mc.get("title", f"Multiple Choice Quiz - {course_name.strip()}")
                    desc = mc.get("description", "Answer the multiple choice questions.")
                    qa_items = build_question_set_multichoice(qs_dir, title, desc, mc.get("items", []))

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(qs_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, "Quiz (Question Set) — Multiple Choice", qa_items)

                elif typ == "Dialog Cards":
                    work_dir = os.path.join(tmp, "_work_dialog")
                    unzip_h5p(templates["Dialog Cards"], work_dir)

                    # Dialog Cards are validated strictly against the extracted PDF text.
                    # Keep the context minimal to avoid pulling in anything outside the PDFs.
                    dialog_context = f"{course_name.strip()}\nUnit: {(unit_name or '').strip()}"
                    gen_data = generate_dialog_cards_strict(chunks, run_n, dialog_context)
                    _gen_bar.progress(65, text="AI content generated & validated — building template...")
                    title = gen_data.get("title", f"Dialog Cards - {course_name.strip()}")
                    desc = gen_data.get("description", "")

                    qa_items = update_dialog_cards_template(
                        work_dir,
                        title,
                        desc,
                        gen_data.get("cards", []),
                        course=course_name.strip(),
                        pdf_headings=st.session_state.get("pdf_headings_cache") or [],
                        pdf_keywords=st.session_state.get("pdf_keywords_cache") or [],
                    )

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Dictation":
                    work_dir = os.path.join(tmp, "_work_dictation")
                    unzip_h5p(templates["Dictation"], work_dir)

                    gen_data = call_llm_dictation(chunks, run_n, enriched_course)
                    title = gen_data.get("title", f"Dictation - {course_name.strip()}")
                    desc = gen_data.get("description", "Listen carefully and type what you hear.")

                    qa_items = update_dictation_template(
                        work_dir,
                        title=title,
                        description=desc,
                        sentences=gen_data.get("sentences", []),
                        progress_callback=lambda pct, txt: _gen_bar.progress(pct, text=txt),
                    )

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Page":
                    work_dir = os.path.join(tmp, "_work_page")
                    unzip_h5p(templates["Page"], work_dir)

                    gen_data = call_llm_page_content(chunks, n_sections=min(6, max(3, run_n // 2)), course=enriched_course)
                    _gen_bar.progress(65, text="AI content generated — building template...")
                    title = gen_data.get("title", f"Page - {course_name.strip()}")
                    qa_items = update_page_template_with_images(
                        work_dir,
                        title,
                        gen_data.get("sections", []),
                        course=course_name.strip(),
                        pdf_headings=st.session_state.get("pdf_headings_cache") or [],
                        pdf_keywords=st.session_state.get("pdf_keywords_cache") or [],
                    )

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Course Presentation":
                    work_dir = os.path.join(tmp, "_work_course_presentation")
                    unzip_h5p(templates["Course Presentation"], work_dir)

                    # Step 1: Generate content slides
                    _gen_bar.progress(30, text="Generating content slides from PDFs...")
                    gen_data = call_llm_course_presentation(chunks, n_slides=run_n, course=enriched_course)

                    # Step 2: Generate activity questions for each selected type
                    act_groups: Dict[str, List[Dict[str, Any]]] = {}
                    if cp_activity_types and cp_n_questions and cp_n_questions > 0:
                        n_types = len(cp_activity_types)
                        base_per_type = int(cp_n_questions) // n_types
                        remainder = int(cp_n_questions) % n_types

                        for ti, atype in enumerate(cp_activity_types):
                            n_q = base_per_type + (1 if ti < remainder else 0)
                            if n_q < 1:
                                continue
                            _gen_bar.progress(
                                40 + (ti * 20 // n_types),
                                text=f"Generating {n_q} {atype} questions..."
                            )
                            q_data = call_llm_cp_activity_questions(
                                chunks,
                                activity_type=atype,
                                n_questions=n_q,
                                course=enriched_course,
                            )
                            act_groups[atype] = q_data.get("questions") or []

                    _gen_bar.progress(65, text="Building presentation template...")
                    title = gen_data.get("title", f"Course Presentation - {course_name.strip()}")
                    desc = gen_data.get("description", "")

                    qa_items = update_course_presentation_template_with_images(
                        work_dir,
                        title=title,
                        description=desc,
                        slides=gen_data.get("slides", []),
                        course=course_name.strip(),
                        pdf_headings=st.session_state.get("pdf_headings_cache") or [],
                        pdf_keywords=st.session_state.get("pdf_keywords_cache") or [],
                        activity_groups=act_groups,
                    )

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Interactive Book":
                    work_dir = os.path.join(tmp, "_work_interactive_book")
                    unzip_h5p(templates["Interactive Book"], work_dir)

                    # Step 1: Generate content chapters/pages
                    _gen_bar.progress(30, text="Generating content pages from PDFs...")
                    gen_data = call_llm_interactive_book(chunks, n_chapters=max(2, min(run_n, run_n)), course=enriched_course)

                    # Step 2: Generate activity questions for each selected type
                    ib_act_groups: Dict[str, List[Dict[str, Any]]] = {}
                    if ib_activity_types and ib_n_questions and ib_n_questions > 0:
                        n_types = len(ib_activity_types)
                        base_per_type = int(ib_n_questions) // n_types
                        remainder = int(ib_n_questions) % n_types

                        for ti, atype in enumerate(ib_activity_types):
                            n_q = base_per_type + (1 if ti < remainder else 0)
                            if n_q < 1:
                                continue
                            _gen_bar.progress(
                                40 + (ti * 20 // n_types),
                                text=f"Generating {n_q} {atype} questions..."
                            )
                            q_data = call_llm_cp_activity_questions(
                                chunks,
                                activity_type=atype,
                                n_questions=n_q,
                                course=enriched_course,
                            )
                            ib_act_groups[atype] = q_data.get("questions") or []

                    _gen_bar.progress(65, text="AI content generated — building template...")
                    title = gen_data.get("title", f"Interactive Book - {course_name.strip()}")
                    desc = gen_data.get("description", "")

                    qa_items = update_interactive_book_template_with_images(
                        work_dir,
                        title=title,
                        description=desc,
                        chapters=gen_data.get("chapters", []),
                        course=course_name.strip(),
                        pdf_headings=st.session_state.get("pdf_headings_cache") or [],
                        pdf_keywords=st.session_state.get("pdf_keywords_cache") or [],
                        activity_groups=ib_act_groups,
                    )

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ in BUILTIN_TEXT_TYPES:
                    meta_t = BUILTIN_TEXT_TYPES[typ]
                    work_dir = os.path.join(tmp, "_work_text")
                    unzip_h5p(templates[typ], work_dir)

                    if meta_t["mode"] == "dragtext":
                        gen_data = call_llm_drag_words(chunks, run_n, enriched_course)
                        _gen_bar.progress(65, text="AI content generated — building template...")
                        textfield = make_dragtext_textfield(gen_data["items"])
                        update_text_based_template(work_dir, gen_data["title"], gen_data["description"], textfield, gen_data.get("overall_feedback"), meta_t["textfield_keys"])
                        title = gen_data["title"]
                        all_dis = []
                        for it in gen_data.get("items", []):
                            all_dis.extend(it.get("distractors") or [])
                        maybe_set_distractors(work_dir, all_dis)
                        qa_items = [{"label": "Drag the Words", "content": it.get("sentence", ""), "expected": it.get("missing_word", ""), "evidence": it.get("evidence", {})}
                                    for it in gen_data.get("items", [])]

                    elif meta_t["mode"] == "blanks":
                        gen_data = call_llm_fill_blanks(chunks, run_n, enriched_course)
                        _gen_bar.progress(65, text="AI content generated — building template...")
                        textfield = make_blanks_textfield(gen_data["items"])
                        update_text_based_template(work_dir, gen_data["title"], gen_data["description"], textfield, gen_data.get("overall_feedback"), meta_t["textfield_keys"])
                        title = gen_data["title"]
                        qa_items = [{"label": f"Item {i+1}", "content": f"{it['sentence']} (answer: {it['answer']})", "evidence": it.get("evidence", {})}
                                    for i, it in enumerate(gen_data.get("items", []))]

                    else:  # markwords
                        gen_data = call_llm_mark_words(chunks, run_n, enriched_course)
                        _gen_bar.progress(65, text="AI content generated — building template...")
                        textfield = make_mark_words_textfield(gen_data["items"])
                        update_text_based_template(work_dir, gen_data["title"], gen_data["description"], textfield, None, meta_t["textfield_keys"])
                        title = gen_data["title"]
                        qa_items = [{"label": f"Item {i+1}", "content": f"{it['paragraph'][:160]}... (marked: {', '.join(it['marked_words'])})", "evidence": it.get("evidence", {})}
                                    for i, it in enumerate(gen_data.get("items", []))]

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Cornell Notes":
                    work_dir = os.path.join(tmp, "_work_cornell_notes")
                    unzip_h5p(templates["Cornell Notes"], work_dir)

                    # ── Read raw template content.json BEFORE patching ────────
                    raw_before = _load_json(work_dir, "content/content.json")

                    _gen_bar.progress(30, text="Generating Cornell Notes content with AI...")
                    cn_gen = call_llm_cornell_notes(chunks, enriched_course)

                    _gen_bar.progress(65, text="Injecting video URL into template...")
                    title = cn_gen.get("title") or f"Cornell Notes - {course_name.strip()}"
                    patched = update_cornell_notes_template(
                        work_dir,
                        title=title,
                        video_url=cornell_video_url,
                        gen_data=cn_gen,
                        poster_image_bytes=cornell_poster_bytes,
                        poster_image_ext=cornell_poster_ext,
                    )

                    # ── Debug expander: show template JSON so dev can verify ──
                    with st.expander("\U0001f50d Debug: Cornell Notes content.json (click to inspect)", expanded=False):
                        st.caption("**Before patching** (template original):")
                        st.json(raw_before)
                        st.caption("**After patching** (what goes into the .h5p):")
                        st.json(patched)

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    qa_items = [
                        {"label": "Video URL used", "content": _normalise_video_url(cornell_video_url) if cornell_video_url else "(template default)", "evidence": {}},
                        {"label": "Body", "content": cn_gen.get("body", ""), "evidence": {}},
                        {"label": "Cue placeholder", "content": cn_gen.get("cue_placeholder", ""), "evidence": {}},
                        {"label": "Notes placeholder", "content": cn_gen.get("notes_placeholder", ""), "evidence": {}},
                        {"label": "Summary placeholder", "content": cn_gen.get("summary_placeholder", ""), "evidence": {}},
                    ]
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Essay":
                    work_dir = os.path.join(tmp, "_work_essay")
                    unzip_h5p(templates["Essay"], work_dir)

                    # Always generate exactly 1 essay question
                    gen_data = call_llm_essay(chunks, enriched_course)
                    _gen_bar.progress(65, text="AI content generated — building template...")

                    title = gen_data.get("title", f"Essay - {course_name.strip()}")
                    desc = gen_data.get("description", "Read the question and write your answer below.")

                    qa_items = update_essay_template(
                        work_dir,
                        title=title,
                        description=desc,
                        essays=gen_data.get("essays", []),
                    )

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Summary":
                    work_dir = os.path.join(tmp, "_work_summary")
                    unzip_h5p(templates["Summary"], work_dir)
                    gen_data = call_llm_summary(chunks, run_n, enriched_course)
                    _gen_bar.progress(65, text="AI content generated — building template...")
                    update_summary_template(work_dir, gen_data["title"], gen_data["description"], gen_data.get("groups", []))
                    title = gen_data["title"]

                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    qa_items = []
                    for i, grp in enumerate(gen_data.get("groups", []), start=1):
                        correct = grp.get("correct_statement", "")
                        incorrects = grp.get("incorrect_statements", [])
                        qa_items.append({
                            "label": f"Group {i}",
                            "content": f"Correct: {correct}\nIncorrect: {'; '.join(incorrects)}",
                            "evidence": grp.get("evidence", {}),
                        })

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, qa_items)

                else:
                    work_dir = os.path.join(tmp, "_work_generic")
                    unzip_h5p(templates[typ], work_dir)

                    tpl_h5p = json.loads(open(os.path.join(work_dir, "h5p.json"), "r", encoding="utf-8").read())
                    tpl_content = json.loads(open(os.path.join(work_dir, "content", "content.json"), "r", encoding="utf-8").read())

                    gen_data = call_llm_generic_patch(
                        chunks=chunks,
                        course_name=enriched_course,
                        activity_type=typ,
                        template_h5p_json=tpl_h5p,
                        template_content_json=tpl_content,
                        item_count=run_n,
                    )
                    _gen_bar.progress(65, text="AI content generated — building template...")

                    update_h5p_title(work_dir, gen_data["title"])
                    _save_json(work_dir, "content/content.json", gen_data["patched_content_json"])

                    title = gen_data["title"]
                    out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                    zip_dir_to_file(work_dir, out_h5p)

                    out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                    write_qa_report_html(out_qa, title, typ, gen_data.get("qa_items", []))

                _gen_bar.progress(90, text="Packaging H5P file...")
                st.success("Done.")
                _gen_bar.progress(100, text="Done ✓")
                time.sleep(0.8)
                gen_progress_area.empty()

                # Persist outputs in session state so downloads remain available after reruns
                with open(out_h5p, "rb") as f:
                    st.session_state["last_h5p_bytes"] = f.read()
                with open(out_qa, "rb") as f:
                    st.session_state["last_qa_bytes"] = f.read()
                st.session_state["last_h5p_name"] = os.path.basename(out_h5p)
                st.session_state["last_qa_name"] = os.path.basename(out_qa)

        except Exception as e:
            gen_progress_area.empty()
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                st.error("Rate limit reached while generating. Please wait a minute and try again.")
            else:
                st.error(msg)
        finally:
            st.session_state["busy"] = False

# Persistent downloads (visible even after button click reruns)
if st.session_state.get("last_h5p_bytes"):
    st.markdown("---")
    st.subheader("Downloads")
    st.download_button(
        "Download H5P (.h5p)",
        data=st.session_state["last_h5p_bytes"],
        file_name=st.session_state.get("last_h5p_name") or "activity.h5p",
        use_container_width=True,
    )

if st.session_state.get("last_qa_bytes"):
    st.download_button(
        "Download QA evidence (.html)",
        data=st.session_state["last_qa_bytes"],
        file_name=st.session_state.get("last_qa_name") or "QA.html",
        use_container_width=True,
    )