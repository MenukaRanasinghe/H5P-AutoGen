
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
        # Hard reject completely irrelevant titles if we have terms and none match.
        if q_terms and score == 0:
            continue

        candidates.append({"url": url, "mime": mime, "width": w, "height": h, "title": title, "score": score})

    if not candidates:
        return None

    # Highest score first, then largest pixel area.
    candidates.sort(key=lambda x: (x["score"], x["width"] * x["height"]), reverse=True)
    return candidates[0]

_PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3G7qkAAAAASUVORK5CYII="
)

def _write_placeholder_png(path: str) -> None:
    import base64
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(_PLACEHOLDER_PNG_B64))

def download_image_to_h5p(images_dir: str, query: str, stem: str) -> Optional[Dict[str, str]]:
    """Download an image for query into content/images and return dict with rel path + mime."""
    os.makedirs(images_dir, exist_ok=True)
    found = wikimedia_find_image_url(query)
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
    return {"path": f"images/{fname}", "mime": mime}

def download_image_to_h5p_multi(images_dir: str, queries: List[str], stem: str) -> Optional[Dict[str, str]]:
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
        dl = download_image_to_h5p(images_dir, q, stem=f"{stem}_{q[:40]}")
        if dl:
            return dl
    return None

def ensure_image(images_dir: str, queries: List[str], stem: str, fallback_query: str = "healthcare worker") -> Dict[str, str]:
    """Guarantee an image payload. Uses Wikimedia search; if it fails, writes a tiny placeholder PNG."""
    dl = download_image_to_h5p_multi(images_dir, queries + [fallback_query], stem=stem)
    if dl:
        return dl
    # absolute last resort
    fname = f"{safe_filename(stem)}_placeholder.png"
    abs_path = os.path.join(images_dir, fname)
    _write_placeholder_png(abs_path)
    return {"path": f"images/{fname}", "mime": "image/png"}

def extract_keywords(text: str, max_terms: int = 4) -> List[str]:
    """Lightweight keyword extraction for better image searches."""
    toks = _terms(text)
    # prefer longer / more specific terms
    toks.sort(key=lambda t: (-len(t), t))
    return toks[:max_terms]


# Disk cache (avoids repeat API calls across refresh/restart)
CACHE_DIR = os.environ.get("H5P_CACHE_DIR", ".cache_h5p_app")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def cache_read_json(key: str):
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def cache_write_json(key: str, data):
    p = _cache_path(key)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



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


ALL_H5P_TYPES = [
    "Course Presentation",
    "Dialog Cards",
    "Drag the Words",
    "Fill in the Blanks",
    "Interactive Book",
    "Mark the Words",
    "Multiple Choice",
    "Page",
    "Quiz",
    "Single Choice",
    "Summary",
]

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
    system = "Create Dialog Cards grounded strictly in SOURCE. Return JSON only."
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

Rules:
- FRONT must be a short description (1–2 sentences) that clearly sets context.
- BACK must be the answer only: 1–2 words, no punctuation.
- The card must be directly supported by the QUOTE.
- image_query must be 2–6 words describing a suitable, text-free stock-style image (no brands, no logos, no text overlays, no source names).
- quote must be copied exactly from SOURCE.

SOURCE:
{src_txt}
""".strip()
    return call_openai_chat_json(system, user)

def update_dialog_cards_template(
    work_dir: str,
    title: str,
    description: str,
    cards: List[Dict[str, Any]],
    course: str = "",
) -> List[Dict[str, Any]]:
    """Populate Dialog Cards and attach an illustrative image per card.

    Template-tolerant:
    - Locates the cards list by scoring all lists in content/content.json.
    - Preserves the template's per-card schema by cloning a sample card object.
    """
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

    def _clean_short_answer(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[\s\.,;:!\?\-]+$", "", s)
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", s)
        if not words:
            return s
        return " ".join(words[:2])

    def _set_image_fields(obj: Any, rel_path: str, mime: str) -> Any:
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
                    out[k] = _set_image_fields(v, rel_path, mime)
            if "path" not in out:
                out["path"] = rel_path
            if "mime" not in out:
                out["mime"] = mime
            if "copyright" not in out:
                out["copyright"] = {"license": "U"}
            return out
        if isinstance(obj, list):
            return [_set_image_fields(v, rel_path, mime) for v in obj]
        return {"path": rel_path, "mime": mime, "copyright": {"license": "U"}}

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

        # Always attach an image. Prefer the LLM's image_query, then fall back to extracted keywords.
        kw_front = " ".join(extract_keywords(front, 4))
        queries = [img_q, back, kw_front, course]
        # Choose a sensible fallback query if course is empty
        fallback = "adult care" if "care" in (course or "").lower() else "workplace safety"
        dl = ensure_image(images_dir, queries=queries, stem=f"dialog_{i}", fallback_query=fallback)
        image_payload = {"path": dl["path"], "mime": dl["mime"]}

        card_obj = copy.deepcopy(sample_card) if sample_card else {}
        card_obj[front_key] = front
        card_obj[back_key] = back

        existing_img = card_obj.get(image_key)
        card_obj[image_key] = _set_image_fields(existing_img, image_payload["path"], image_payload["mime"])

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
) -> List[Dict[str, Any]]:
    """Populate an H5P Page with Image + Text blocks using section.image_query (Wikimedia Commons)."""
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

        kw = " ".join(extract_keywords(f"{heading} {re.sub('<[^<]+?>','',body)}", 4))
        queries = [img_q, heading, kw, course]
        fallback = "adult care" if "care" in (course or "").lower() else "workplace safety"
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
    system = "Create an H5P Course Presentation grounded strictly in SOURCE. Return JSON only."
    src_txt = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Create a Course Presentation for course: {course}

Return JSON:
{{
  "title":"string",
  "description":"string",
  "slides":[
    {{
      "heading":"string",
      "bullets":["string","string","string"],
      "image_query":"string",
      "evidence":{{"source_file":"string","locator":"PDF p.X/Y","quote":"short exact quote"}}
    }}
  ]
}}

Rules:
- Produce 5 to {n_slides} slides (keep slides concise).
- bullets must be 3–6 short bullet points (no numbering).
- image_query must be 2–6 words for a clear, text-free illustrative photo (no logos/brands, no source names).
- Evidence quote must be copied exactly from SOURCE and support the slide content.

SOURCE:
{src_txt}
"""
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


def _html_bullets(heading: str, bullets: List[str]) -> str:
    h = (heading or "").strip()
    li = "".join([f"<li>{(b or '').strip()}</li>" for b in (bullets or []) if (b or '').strip()])
    if not li:
        li = "<li>—</li>"
    if h:
        return f"<h2>{h}</h2><ul>{li}</ul>"
    return f"<ul>{li}</ul>"


def update_course_presentation_template_with_images(
    work_dir: str,
    title: str,
    description: str,
    slides: List[Dict[str, Any]],
    course: str = "",
) -> List[Dict[str, Any]]:
    """Populate a Course Presentation template while preserving slide layout.
    - Reuses existing slides/elements from the template whenever possible.
    - Sets text in the first AdvancedText element per slide.
    - Replaces the first Image element per slide (or background image if present).
    """
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    slides_ref = _best_list_by_score(content, _score_slides_list)
    if slides_ref is None:
        raise KeyError(
            "Could not locate a 'slides' list in the Course Presentation template. "
            "Export a blank Course Presentation with at least 1 slide and re-add it to ./templates."
        )

    sample_slide = None
    if slides_ref and isinstance(slides_ref[0], dict):
        sample_slide = slides_ref[0]

    images_dir = os.path.join(work_dir, "content", "images")
    os.makedirs(images_dir, exist_ok=True)

    qa_items: List[Dict[str, Any]] = []

    target_n = max(1, len(slides))
    while len(slides_ref) < target_n:
        slides_ref.append(copy.deepcopy(sample_slide) if sample_slide else {})

    for i, gen in enumerate(slides, start=1):
        if i > len(slides_ref):
            break
        slide_obj = slides_ref[i - 1]
        heading = (gen.get("heading") or "").strip()
        bullets = gen.get("bullets") or []
        img_q = (gen.get("image_query") or "").strip()
        ev = gen.get("evidence") or {}

        deep_find_set_first(slide_obj, ["slideTitle", "title", "heading"], heading)

        html = _html_bullets(heading, bullets)
        adv_blocks = [b for b in _iter_library_blocks(slide_obj) if str(b.get("library","")).startswith("H5P.AdvancedText")]
        if adv_blocks:
            adv_blocks[0].setdefault("params", {})
            adv_blocks[0]["params"]["text"] = html
        else:
            deep_find_set_first(slide_obj, ["text", "html", "content", "questionText"], html)

        kw = " ".join(extract_keywords(f"{heading} {' '.join([str(x) for x in bullets])}", 4))
        queries = [img_q, heading, kw, course]
        fallback = "adult care" if "care" in (course or "").lower() else "workplace training"
        dl = ensure_image(images_dir, queries=queries, stem=f"cp_slide_{i}", fallback_query=fallback)

        img_blocks = [b for b in _iter_library_blocks(slide_obj) if str(b.get("library","")).startswith("H5P.Image")]
        if img_blocks:
            file_obj = (img_blocks[0].get("params") or {}).get("file")
            img_blocks[0].setdefault("params", {})
            img_blocks[0]["params"]["file"] = {"path": dl["path"], "mime": dl["mime"], "copyright": {"license": "U"}} if not isinstance(file_obj, dict) else h5p_set_image_fields(file_obj, dl["path"], dl["mime"])
        else:
            found = deep_find_first_key(slide_obj, ["backgroundImage", "background"])
            if found and isinstance(found[1], dict):
                k, v = found
                slide_obj[k] = h5p_set_image_fields(v, dl["path"], dl["mime"])

        qa_items.append({
            "label": f"Slide {i}",
            "content": f"{heading}\n" + "\n".join([f"- {b}" for b in bullets[:6]]),
            "expected": "",
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
) -> List[Dict[str, Any]]:
    """Populate an Interactive Book template while preserving its structure."""
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")

    chapters_ref = _best_list_by_score(content, _score_chapters_list)
    if chapters_ref is None:
        raise KeyError(
            "Could not locate a 'chapters' list in the Interactive Book template. "
            "Export a blank Interactive Book with at least 1 chapter and re-add it to ./templates."
        )

    sample_chapter = None
    if chapters_ref and isinstance(chapters_ref[0], dict):
        sample_chapter = chapters_ref[0]

    images_dir = os.path.join(work_dir, "content", "images")
    os.makedirs(images_dir, exist_ok=True)

    qa_items: List[Dict[str, Any]] = []

    target_chapters = max(1, len(chapters))
    while len(chapters_ref) < target_chapters:
        chapters_ref.append(copy.deepcopy(sample_chapter) if sample_chapter else {})

    for ci, ch in enumerate(chapters, start=1):
        if ci > len(chapters_ref):
            break
        ch_obj = chapters_ref[ci - 1]
        ch_title = (ch.get("chapter_title") or ch.get("title") or f"Chapter {ci}").strip()
        deep_find_set_first(ch_obj, ["title", "chapterTitle", "chapter_title", "heading"], ch_title)

        sections = ch.get("sections") or []
        parts = []
        for si, sec in enumerate(sections, start=1):
            h = (sec.get("heading") or "").strip()
            body = (sec.get("body_html") or "").strip()
            if h:
                parts.append(f"<h3>{h}</h3>")
            if body:
                parts.append(body)
        chapter_html = "\n".join(parts).strip() or f"<p>{ch_title}</p>"

        adv_blocks = [b for b in _iter_library_blocks(ch_obj) if str(b.get("library","")).startswith("H5P.AdvancedText")]
        if adv_blocks:
            adv_blocks[0].setdefault("params", {})
            adv_blocks[0]["params"]["text"] = chapter_html
        else:
            deep_find_set_first(ch_obj, ["text", "html", "content", "introduction"], chapter_html)

        first_sec = sections[0] if sections else {}
        img_q = (first_sec.get("image_query") or "").strip()
        kw = " ".join(extract_keywords(re.sub('<[^<]+?>', ' ', chapter_html), 5))
        queries = [img_q, ch_title, kw, course]
        fallback = "adult care" if "care" in (course or "").lower() else "workplace training"
        dl = ensure_image(images_dir, queries=queries, stem=f"ib_ch_{ci}", fallback_query=fallback)

        img_blocks = [b for b in _iter_library_blocks(ch_obj) if str(b.get("library","")).startswith("H5P.Image")]
        if img_blocks:
            file_obj = (img_blocks[0].get("params") or {}).get("file")
            img_blocks[0].setdefault("params", {})
            img_blocks[0]["params"]["file"] = {"path": dl["path"], "mime": dl["mime"], "copyright": {"license": "U"}} if not isinstance(file_obj, dict) else h5p_set_image_fields(file_obj, dl["path"], dl["mime"])
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

def call_openai_chat_json(system: str, user: str, model: str = "gpt-5-mini", temperature: float = 0.2) -> Dict[str, Any]:
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
def llm_suggest_activities(chunks: List[ContentChunk], course_name: str) -> Dict[str, Any]:
    system = "You are an instructional designer specialising in H5P. Return valid JSON only."
    src = join_chunks_for_prompt(chunks, max_chars=65000)
    user = f"""
Recommend suitable H5P activity types for this course: {course_name}

Consider these H5P types (exact labels):
{ALL_H5P_TYPES}

Return ONLY the best 8 recommendations.

Rules:
- Base suggestions on the source text.
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
Create Summary with {n} statements (mix correct/incorrect ~60/40).

JSON:
{{
 "title":"string","description":"string",
 "items":[
   {{"statement":"string","is_correct":true,"evidence":{{"source_file":"string","locator":"Page X","quote":"string"}}}}
 ]
}}

Course: {course}
Source:
{src}
""".strip()
    return call_openai_chat_json(system, user)


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



def update_summary_template(work_dir: str, title: str, description: str, items: List[Dict[str, Any]]) -> None:
    update_h5p_title(work_dir, title)
    content = _load_json(work_dir, "content/content.json")
    deep_find_set_first(content, ["taskDescription", "introduction", "description", "instructions"], description)

    summary_objs = [{
        "subContentId": random_subcontent_id(),
        "tip": "",
        "summary": it.get("statement", ""),
        "correct": bool(it.get("is_correct")),
    } for it in items]

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

st.markdown("## H5P Activity Generator")
st.caption("Upload PDF(s) → Get suggestions → Select one type → Generate H5P")

templates = discover_templates("templates")

# Keep state keys
st.session_state.setdefault("pdf_fingerprints", None)
st.session_state.setdefault("chunks_cache", None)
st.session_state.setdefault("pdf_bytes_map", None)
st.session_state.setdefault("suggestions_cache_key", None)
st.session_state.setdefault("suggestions", None)
st.session_state.setdefault("busy", False)

# Persist latest outputs across Streamlit reruns (download buttons remain visible)
st.session_state.setdefault("last_h5p_bytes", None)
st.session_state.setdefault("last_h5p_name", None)
st.session_state.setdefault("last_qa_bytes", None)
st.session_state.setdefault("last_qa_name", None)

uploads = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)
course_name = st.text_input("Course name", placeholder="e.g., Level 5 Diploma in ...")

def compute_inputs_key(files: List[Any], course: str) -> str:
    # Use file hashes + course name to cache extraction + suggestions
    parts = [course.strip()]
    for f in files:
        b = f.getvalue()
        parts.append(f.name)
        parts.append(file_sha256(b))
    return hashlib.sha256(("|".join(parts)).encode("utf-8")).hexdigest()

def ensure_chunks(files: List[Any]) -> List[ContentChunk]:
    # Cache by fingerprints of uploaded PDFs
    fps = [(f.name, file_sha256(f.getvalue())) for f in files]
    if st.session_state["pdf_fingerprints"] == fps and st.session_state["chunks_cache"] is not None and st.session_state["pdf_bytes_map"] is not None:
        return st.session_state["chunks_cache"]

    chunks: List[ContentChunk] = []
    pdf_map: Dict[str, bytes] = {}
    for f in files:
        b = f.getvalue()
        pdf_map[f.name] = b
        chunks.extend(extract_pdf_chunks_from_bytes(f.name, b))
    if not chunks:
        raise RuntimeError("No readable text found in the uploaded PDF(s). (If PDFs are scanned images, use OCR PDFs.)")

    st.session_state["pdf_fingerprints"] = fps
    st.session_state["chunks_cache"] = chunks
    st.session_state["pdf_bytes_map"] = pdf_map
    return chunks

    chunks: List[ContentChunk] = []
    for f in files:
        chunks.extend(extract_pdf_chunks_from_bytes(f.name, f.getvalue()))
    if not chunks:
        raise RuntimeError("No readable text found in the uploaded PDF(s). (If PDFs are scanned images, use OCR PDFs.)")

    st.session_state["pdf_fingerprints"] = fps
    st.session_state["chunks_cache"] = chunks
    return chunks

colA, colB = st.columns(2)
with colA:
    suggest_clicked = st.button("Suggest H5P types", use_container_width=True, disabled=st.session_state["busy"])
with colB:
    clear_clicked = st.button("Clear", use_container_width=True, disabled=st.session_state["busy"])

if clear_clicked:
    st.session_state["pdf_fingerprints"] = None
    st.session_state["chunks_cache"] = None
    st.session_state["suggestions_cache_key"] = None
    st.session_state["suggestions"] = None
    st.session_state["pdf_bytes_map"] = None
    st.session_state["last_h5p_bytes"] = None
    st.session_state["last_h5p_name"] = None
    st.session_state["last_qa_bytes"] = None
    st.session_state["last_qa_name"] = None
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
        if not os.environ.get("LLM_API_KEY"):
            st.error("Missing API key. Set LLM_API_KEY.")
            st.stop()

        key = compute_inputs_key(uploads, course_name)
        if st.session_state["suggestions_cache_key"] == key and st.session_state["suggestions"] is not None:
            # Already computed for these inputs
            pass
        else:
            chunks = ensure_chunks(uploads)
            disk_key = f"suggest_{key}"
            cached = cache_read_json(disk_key)
            if cached is not None:
                st.session_state["suggestions"] = cached
                st.session_state["suggestions_cache_key"] = key
            else:
                chunks_small = choose_representative_chunks(chunks, max_pages=18)
                with st.spinner("Analysing PDFs and generating suggestions..."):
                    s = llm_suggest_activities(chunks_small, course_name.strip())
                st.session_state["suggestions"] = s
                st.session_state["suggestions_cache_key"] = key
                cache_write_json(disk_key, s)

    except Exception as e:
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
    # Constrain to the templates/types we actually support in this build
    allowed = set(ALL_H5P_TYPES)
    recs = [r for r in recs if isinstance(r, dict) and (r.get("activity_type") in allowed)]

    if not isinstance(recs, list) or not recs:
        st.info("No suggestions returned. Try again.")
        st.stop()

    st.markdown("---")
    st.markdown("### Choose one suggested type")

    # Make a compact radio list with template availability
    options = []
    meta = {}
    for r in recs:
        typ = r.get("activity_type", "")
        score = int(r.get("score_0_to_5", 0) or 0)
        why = (r.get("why") or "").strip()
        suggested_n = int(r.get("suggested_item_count", 5) or 5)
        ev = r.get("evidence", {}) or {}

        template_ok = (typ in templates) if typ not in ("Quiz","Multiple Choice") else ("Quiz" in templates)
        status = "OK" if template_ok else "Missing template"
        label = f"{typ} (score {score}/5) — {status}"
        options.append(label)
        meta[label] = {"type": typ, "why": why, "n": suggested_n, "ev": ev, "template_ok": template_ok}

    choice = st.radio("Suggested types", options=options, index=0)
    chosen = meta[choice]

    st.caption(chosen["why"] if chosen["why"] else "—")
    ev = chosen["ev"] or {}
    if ev:
        st.caption(f"Evidence: {ev.get('source_file','')} {ev.get('locator','')} — “{str(ev.get('quote',''))[:170]}”")

    st.markdown("---")
    st.markdown("### Generate H5P")
    default_n = max(5, chosen["n"]) if chosen["type"] in ("Quiz","Multiple Choice") else max(3, chosen["n"])
    n_items = st.number_input("Number of items/questions", min_value=3, max_value=30, value=int(default_n), step=1)

    gen = st.button("Generate H5P file", type="primary", use_container_width=True, disabled=st.session_state["busy"])

    if gen:
        try:
            st.session_state["busy"] = True

            if not chosen["template_ok"]:
                if chosen["type"] in ("Quiz","Multiple Choice"):
                    st.error("Missing template: templates/Quiz.h5p (required for Question Set generation).")
                else:
                    st.error(f"Missing template: templates/{'Quiz' if chosen['type'] in ('Quiz','Multiple Choice') else chosen['type']}.h5p")
                st.stop()

            chunks = ensure_chunks(uploads)

            with tempfile.TemporaryDirectory() as tmp:
                typ = chosen["type"]
                run_n = int(n_items)
                if typ == "Quiz":
                        tf = call_llm_truefalse_statements(chunks, run_n, course_name.strip())

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
                        mc = call_llm_multichoice_questions(chunks, run_n, course_name.strip())

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

                        gen_data = call_llm_dialog_cards(chunks, run_n, course_name.strip())
                        title = gen_data.get("title", f"Dialog Cards - {course_name.strip()}")
                        desc = gen_data.get("description", "")

                        qa_items = update_dialog_cards_template(work_dir, title, desc, gen_data.get("cards", []), course=course_name.strip())

                        out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                        zip_dir_to_file(work_dir, out_h5p)

                        out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                        write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Page":
                        work_dir = os.path.join(tmp, "_work_page")
                        unzip_h5p(templates["Page"], work_dir)

                        gen_data = call_llm_page_content(chunks, n_sections=min(6, max(3, run_n//2)), course=course_name.strip())
                        title = gen_data.get("title", f"Page - {course_name.strip()}")
                        qa_items = update_page_template_with_images(work_dir, title, gen_data.get("sections", []), course=course_name.strip())

                        out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                        zip_dir_to_file(work_dir, out_h5p)

                        out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                        write_qa_report_html(out_qa, title, typ, qa_items)


                elif typ == "Course Presentation":
                        work_dir = os.path.join(tmp, "_work_course_presentation")
                        unzip_h5p(templates["Course Presentation"], work_dir)

                        gen_data = call_llm_course_presentation(chunks, n_slides=run_n, course=course_name.strip())
                        title = gen_data.get("title", f"Course Presentation - {course_name.strip()}")
                        desc = gen_data.get("description", "")

                        qa_items = update_course_presentation_template_with_images(
                            work_dir,
                            title=title,
                            description=desc,
                            slides=gen_data.get("slides", []),
                            course=course_name.strip(),
                        )

                        out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                        zip_dir_to_file(work_dir, out_h5p)

                        out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                        write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Interactive Book":
                        work_dir = os.path.join(tmp, "_work_interactive_book")
                        unzip_h5p(templates["Interactive Book"], work_dir)

                        gen_data = call_llm_interactive_book(chunks, n_chapters=max(2, min(6, run_n // 2)), course=course_name.strip())
                        title = gen_data.get("title", f"Interactive Book - {course_name.strip()}")
                        desc = gen_data.get("description", "")

                        qa_items = update_interactive_book_template_with_images(
                            work_dir,
                            title=title,
                            description=desc,
                            chapters=gen_data.get("chapters", []),
                            course=course_name.strip(),
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
                            gen_data = call_llm_drag_words(chunks, run_n, course_name.strip())
                            textfield = make_dragtext_textfield(gen_data["items"])
                            update_text_based_template(work_dir, gen_data["title"], gen_data["description"], textfield, gen_data.get("overall_feedback"), meta_t["textfield_keys"])
                            title = gen_data["title"]
                            # Apply distractors if the template supports it
                            all_dis = []
                            for it in gen_data.get("items", []):
                                all_dis.extend(it.get("distractors") or [])
                            maybe_set_distractors(work_dir, all_dis)
                            qa_items = [{"label": "Drag the Words", "content": it.get("sentence",""), "expected": it.get("missing_word",""), "evidence": it.get("evidence", {})}
                                        for i, it in enumerate(gen_data["items"])]

                        elif meta_t["mode"] == "blanks":
                            gen_data = call_llm_fill_blanks(chunks, run_n, course_name.strip())
                            textfield = make_blanks_textfield(gen_data["items"])
                            update_text_based_template(work_dir, gen_data["title"], gen_data["description"], textfield, gen_data.get("overall_feedback"), meta_t["textfield_keys"])
                            title = gen_data["title"]
                            # Apply distractors if the template supports it
                            all_dis = []
                            for it in gen_data.get("items", []):
                                all_dis.extend(it.get("distractors") or [])
                            maybe_set_distractors(work_dir, all_dis)
                            qa_items = [{"label": f"Item {i+1}", "content": f"{it['sentence']} (answer: {it['answer']})", "evidence": it.get("evidence", {})}
                                        for i, it in enumerate(gen_data["items"])]

                        else:  # markwords
                            gen_data = call_llm_mark_words(chunks, run_n, course_name.strip())
                            textfield = make_mark_words_textfield(gen_data["items"])
                            update_text_based_template(work_dir, gen_data["title"], gen_data["description"], textfield, None, meta_t["textfield_keys"])
                            title = gen_data["title"]
                            qa_items = [{"label": f"Item {i+1}", "content": f"{it['paragraph'][:160]}... (marked: {', '.join(it['marked_words'])})", "evidence": it.get("evidence", {})}
                                        for i, it in enumerate(gen_data["items"])]

                        out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                        zip_dir_to_file(work_dir, out_h5p)

                        out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                        write_qa_report_html(out_qa, title, typ, qa_items)

                elif typ == "Summary":
                        work_dir = os.path.join(tmp, "_work_summary")
                        unzip_h5p(templates["Summary"], work_dir)
                        gen_data = call_llm_summary(chunks, run_n, course_name.strip())
                        update_summary_template(work_dir, gen_data["title"], gen_data["description"], gen_data["items"])
                        title = gen_data["title"]
                        out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                        zip_dir_to_file(work_dir, out_h5p)

                        qa_items = [{"label": f"Item {i+1}", "content": f"{it['statement']} (is_correct: {it['is_correct']})", "evidence": it.get("evidence", {})}
                                    for i, it in enumerate(gen_data["items"])]
                        out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                        write_qa_report_html(out_qa, title, typ, qa_items)

                else:
                        work_dir = os.path.join(tmp, "_work_generic")
                        unzip_h5p(templates[typ], work_dir)

                        tpl_h5p = json.loads(open(os.path.join(work_dir, "h5p.json"), "r", encoding="utf-8").read())
                        tpl_content = json.loads(open(os.path.join(work_dir, "content", "content.json"), "r", encoding="utf-8").read())

                        gen_data = call_llm_generic_patch(
                            chunks=chunks,
                            course_name=course_name.strip(),
                            activity_type=typ,
                            template_h5p_json=tpl_h5p,
                            template_content_json=tpl_content,
                            item_count=run_n,
                        )

                        update_h5p_title(work_dir, gen_data["title"])
                        _save_json(work_dir, "content/content.json", gen_data["patched_content_json"])

                        title = gen_data["title"]
                        out_h5p = os.path.join(tmp, f"{safe_filename(title)}.h5p")
                        zip_dir_to_file(work_dir, out_h5p)

                        out_qa = os.path.join(tmp, f"QA_{safe_filename(title)}.html")
                        write_qa_report_html(out_qa, title, typ, gen_data.get("qa_items", []))

                st.success("Done.")

                # Persist outputs in session state so downloads remain available after reruns
                with open(out_h5p, "rb") as f:
                    st.session_state["last_h5p_bytes"] = f.read()
                with open(out_qa, "rb") as f:
                    st.session_state["last_qa_bytes"] = f.read()
                st.session_state["last_h5p_name"] = os.path.basename(out_h5p)
                st.session_state["last_qa_name"] = os.path.basename(out_qa)

        except Exception as e:
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
