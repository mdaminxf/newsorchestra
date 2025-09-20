# app.py
"""
NewsOrchestra — Multi-agent AI Orchestrator for explainable fact-checking
(Updated to use Gemini as planner + synthesizer so the orchestrator behaves more like a single LLM agent)
Requirements:
- SERPAPI_KEY (optional but recommended)
- GEMINI_API_KEY (optional to enable Gemini planner/synthesizer)
- Optional HF models (summarizer, zero-shot, image models)
"""
import os
import re
import json
import logging
import traceback
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import urlparse
from io import BytesIO
import time

import requests
from bs4 import BeautifulSoup
from PIL import Image, ExifTags

import gradio as gr
from transformers import pipeline

# Optional Google Gemini (genai) client
try:
    from google import genai
except Exception:
    genai = None

# Optional OCR
try:
    import pytesseract
except Exception:
    pytesseract = None

# -----------------------------
# Logging / Config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("newsorchestra")

# Env-configurable model IDs
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SAFE_BROWSING_API_KEY = os.getenv("SAFE_BROWSING_API_KEY")
HF_SUMMARIZER = os.getenv("HF_SUMMARIZER", "sshleifer/distilbart-cnn-12-6")
HF_ZERO_SHOT = os.getenv("HF_ZERO_SHOT", "facebook/bart-large-mnli")
HF_IMAGE_CAPTION = os.getenv("HF_IMAGE_CAPTION", "nlpconnect/vit-gpt2-image-captioning")
HF_IMAGE_CLASSIFIER = os.getenv("HF_IMAGE_CLASSIFIER", "google/vit-base-patch16-224")
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-2.5-flash")

# Candidate labels for zero-shot classification (tweak as desired)
CANDIDATE_LABELS = ["True", "False", "Misleading", "Unclear", "Opinionated", "Unsupported"]

# Trusted-source mapping (used for evidence scoring)
SOURCE_TRUST = {
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "bbc.com": 0.93,
    "theguardian.com": 0.9,
    "nytimes.com": 0.9,
    "washingtonpost.com": 0.9,
    "aljazeera.com": 0.88,
    "cnn.com": 0.88,
    "msn.com": 0.8,
}

# -----------------------------
# Load HF pipelines (best-effort)
# -----------------------------
summarizer = None
zero_shot = None
img_caption = None
image_classifier = None

try:
    summarizer = pipeline("summarization", model=HF_SUMMARIZER, truncation=True)
    logger.info("Loaded summarizer pipeline")
except Exception as e:
    logger.warning("Could not load summarizer: %s", e)
    summarizer = None

try:
    zero_shot = pipeline("zero-shot-classification", model=HF_ZERO_SHOT)
    logger.info("Loaded zero-shot pipeline")
except Exception as e:
    logger.warning("Could not load zero-shot classifier: %s", e)
    zero_shot = None

try:
    # some HF builds call this "image-to-text" or "image-captioning"
    try:
        img_caption = pipeline("image-to-text", model=HF_IMAGE_CAPTION)
    except Exception:
        img_caption = pipeline("image-captioning", model=HF_IMAGE_CAPTION)
    logger.info("Loaded image caption pipeline")
except Exception as e:
    logger.warning("Image caption pipeline unavailable: %s", e)
    img_caption = None

# More robust image-classifier load with full exception trace
try:
    image_classifier = pipeline("image-classification", model=HF_IMAGE_CLASSIFIER)
    logger.info("Loaded image-classification pipeline: %s", HF_IMAGE_CLASSIFIER)
except Exception as e:
    logger.exception("Image-classifier unavailable at startup: %s", e)
    image_classifier = None

# -----------------------------
# Gemini (genai) client init (optional)
# -----------------------------
genai_client = None
if genai:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
        logger.info("Gemini (genai) client initialized")
    except Exception as e:
        logger.warning("Could not initialize genai client: %s", e)
        genai_client = None
else:
    logger.info("genai not installed / available; Gemini planner/synthesizer disabled")

# -----------------------------
# Utility helpers
# -----------------------------
def safe_truncate(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n\n[[TRUNCATED]]"

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    # remove HTML tags crudely
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _extract_json_from_text(text: str) -> Optional[str]:
    """Find first top-level JSON object in text and return it"""
    if not text:
        return None
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i+1]
    return None

# -----------------------------
# Misinfo heuristics (same as before, unchanged)
# -----------------------------
def compute_misinfo_risk(classifier_result, serpapi_result, image_check, verifier_result):
    score = 0.0
    explanations = []
    recommendations = []

    # 1. Classifier signal
    if classifier_result and isinstance(classifier_result, dict):
        labels = [str(l).lower() for l in classifier_result.get("labels", [])]
        if "opinionated" in labels:
            score += 0.2
            explanations.append("Classifier flagged the claim as opinionated.")
        if "misleading" in labels:
            score += 0.3
            explanations.append("Classifier suggested the claim might be misleading.")

    # 2. Verifier signal
    if verifier_result and isinstance(verifier_result, dict):
        if verifier_result.get("verdict") == "Misleading":
            score += 0.3
            explanations.append("Verifier judged the claim as misleading.")
        elif verifier_result.get("verdict") == "False":
            score += 0.5
            explanations.append("Verifier judged the claim as false.")

    # 3. Evidence consistency
    if serpapi_result and isinstance(serpapi_result, dict) and serpapi_result.get("available"):
        agg = serpapi_result.get("result", {}) if isinstance(serpapi_result.get("result"), dict) else serpapi_result.get("result")
        organic = agg.get("organic_results", []) if isinstance(agg, dict) else []
        if not organic:
            score += 0.2
            explanations.append("No supporting evidence found in web search.")
        else:
            explanations.append("Supporting evidence was found in web search.")

    # 4. Image mismatch
    if image_check and isinstance(image_check, dict):
        if image_check.get("claim_caption_mismatch", False):
            score += 0.2
            explanations.append("Attached image does not match the claim caption.")

    score = min(score, 1.0)
    if score < 0.3:
        level = "low"
    elif score < 0.6:
        level = "medium"
    else:
        level = "high"

    if level != "low":
        recommendations.extend([
            "Cross-check with multiple independent news outlets.",
            "Look for official government or NGO reports.",
            "Inspect metadata and reverse image search results if media is attached."
        ])

    return {"score": round(score, 3), "level": level, "explanation": "; ".join(explanations) if explanations else "No major risk signals detected.", "recommendations": recommendations}

# -----------------------------
# Fetch / image helpers (same as before)
# -----------------------------
def fetch_article_text_from_url(url: str) -> str:
    try:
        if not url.startswith("http"):
            url = "http://" + url
        r = requests.get(url, timeout=10, headers={"User-Agent": "newsorchestra-bot/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        article = "\n\n".join([p for p in paragraphs if len(p) > 30])
        if not article:
            desc = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", attrs={"property":"og:description"})
            if desc and desc.get("content"):
                article = desc.get("content")
        return article or (soup.get_text()[:5000] if soup else "")
    except Exception as e:
        logger.exception("fetch_article_text_from_url failed")
        return f"[Error fetching URL: {e}]"

def fetch_image_bytes(url: str, timeout: int = 12) -> Tuple[Optional[Image.Image], Optional[bytes], Optional[str]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Referer": urlparse(url).scheme + "://" + urlparse(url).hostname  # e.g. https://7news.com.au
    }
    try:
        r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        r.raise_for_status()
        b = r.content
        try:
            img = Image.open(BytesIO(b)).convert("RGB")
            return img, b, None
        except Exception as e:
            # Return bytes even if PIL fails to decode
            logger.warning("PIL open failed for %s: %s", url, e)
            return None, b, f"PIL open error: {e}"
    except Exception as e:
        logger.error("fetch_image_bytes failed for %s: %s", url, e)
        return None, None, str(e)


def extract_exif(img: Image.Image) -> dict:
    out = {}
    try:
        raw = img._getexif()
        if not raw:
            return {}
        for tag_id, val in raw.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            out[tag] = val
    except Exception:
        pass
    return out

def image_ocr_text(img: Image.Image) -> Optional[str]:
    if not pytesseract:
        return None
    try:
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return None

# -----------------------------
# SerpApi wrappers (same)
# -----------------------------
def serpapi_web_search(query: str, num: int = 6) -> dict:
    if not SERPAPI_KEY:
        return {"available": False, "note": "SERPAPI_KEY not set"}
    try:
        endpoint = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": query, "num": num, "api_key": SERPAPI_KEY}
        r = requests.get(endpoint, params=params, timeout=12, headers={"User-Agent": "newsorchestra-bot/1.0"})
        if r.status_code != 200:
            return {"available": True, "status_code": r.status_code, "raw": r.text}
        return {"available": True, "status_code": r.status_code, "result": r.json()}
    except Exception as e:
        logger.exception("serpapi_web_search error")
        return {"available": True, "error": str(e)}

def serpapi_reverse_image(image_url: Optional[str] = None, image_bytes: Optional[bytes] = None) -> dict:
    if not SERPAPI_KEY:
        return {"available": False, "note": "SERPAPI_KEY not set"}
    try:
        endpoint = "https://serpapi.com/search"
        if image_url:
            params = {"engine": "google_reverse_image", "image_url": image_url, "api_key": SERPAPI_KEY}
            r = requests.get(endpoint, params=params, timeout=20, headers={"User-Agent": "newsorchestra-bot/1.0"})
        else:
            data = {"engine": "google_reverse_image", "api_key": SERPAPI_KEY}
            files = {"encoded_image": ("image.png", image_bytes, "image/png")}
            r = requests.post(endpoint, data=data, files=files, timeout=30, headers={"User-Agent": "newsorchestra-bot/1.0"})
        if r.status_code != 200:
            return {"available": True, "status_code": r.status_code, "raw": r.text}
        return {"available": True, "status_code": r.status_code, "result": r.json()}
    except Exception as e:
        logger.exception("serpapi_reverse_image error")
        return {"available": True, "error": str(e)}

# -----------------------------
# HF wrappers (same)
# -----------------------------
def extract_claim_from_text(text: str) -> str:
    text = sanitize_text(text)
    if not text:
        return ""
    if len(text) < 300:
        return text.strip().split("\n")[0][:800]
    if summarizer:
        try:
            out = summarizer(text, max_length=100, min_length=20, do_sample=False)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return out[0].get("summary_text") or text.strip().split("\n")[0][:800]
            return str(out)
        except Exception:
            logger.exception("summarizer failed")
            return text.strip().split("\n")[0][:800]
    else:
        return text.strip().split("\n")[0][:800]

def hf_zero_shot(claim: str) -> Dict[str, Any]:
    if not zero_shot:
        return {"error": "zero-shot pipeline not available"}
    
    if not claim or not claim.strip():
        # No text claim provided → skip classification
        return {
            "sequence": "",
            "labels": CANDIDATE_LABELS,
            "scores": [0.0] * len(CANDIDATE_LABELS),
            "note": "No claim text provided; skipping zero-shot classification."
        }
    
    try:
        res = zero_shot(claim, candidate_labels=CANDIDATE_LABELS, multi_label=False)
        if isinstance(res, dict):
            return res
        return dict(res)
    except Exception as e:
        logger.exception("zero-shot failed")
        return {"error": str(e), "trace": traceback.format_exc()}


def hf_image_caption(img: Image.Image) -> Optional[str]:
    if not img_caption:
        return None
    try:
        out = img_caption(img)
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                return first.get("generated_text") or first.get("caption") or str(first)
            return str(first)
        return str(out)
    except Exception:
        logger.exception("image_captioning failed")
        return None

def hf_image_classify(img: Image.Image) -> List[dict]:
    """
    Classify an image using HF pipeline. Always returns a list.
    Accepts PIL.Image, bytes, or file-like. Provides verbose logging on failure.
    """
    results = []
    global image_classifier

    try:
        if img is None:
            logger.warning("hf_image_classify called with img=None")
            return results

        # If bytes were provided, try to open into PIL
        if isinstance(img, (bytes, bytearray)):
            try:
                img = Image.open(BytesIO(img)).convert("RGB")
                logger.info("hf_image_classify: opened image from bytes")
            except Exception as e:
                logger.exception("Could not open image from bytes: %s", e)
                return results

        # If not a PIL image, attempt to open
        if not isinstance(img, Image.Image):
            try:
                img = Image.open(img).convert("RGB")
            except Exception as e:
                logger.exception("hf_image_classify input is not a PIL.Image and could not be opened: %s", e)
                return results

        logger.info("hf_image_classify received image mode=%s size=%s", img.mode, img.size)

        # lazy-init the pipeline if it wasn't loaded at startup
        if image_classifier is None:
            try:
                logger.info("Image classifier pipeline not loaded, attempting lazy init with model: %s", HF_IMAGE_CLASSIFIER)
                from transformers import pipeline as _pipeline
                image_classifier = _pipeline("image-classification", model=HF_IMAGE_CLASSIFIER)
                logger.info("Lazy-loaded image-classification pipeline.")
            except Exception as e:
                logger.exception("Lazy init of image-classification pipeline failed: %s", e)
                return results

        # Call classifier — many HF pipelines accept PIL.Image directly
        try:
            out = image_classifier(img, top_k=3)
            if isinstance(out, list):
                for r in out:
                    if isinstance(r, dict):
                        results.append({"label": str(r.get("label", "unknown")), "score": float(r.get("score", 0))})
                    else:
                        results.append({"label": str(r), "score": None})
        except Exception as e:
            logger.exception("image_classifier call failed: %s", e)
            return results

    except Exception as e:
        logger.exception("hf_image_classify unexpected error: %s", e)
    return results

# -----------------------------
# Gemini planner + synthesizer wrappers (NEW)
# -----------------------------
def call_gemini_planner(claim: str, article_text: str, tools_available: List[str]) -> Dict[str, Any]:
    """
    Ask Gemini to produce a JSON plan instructing which tools to call and in what order.
    Returns a dict like:
    {"plan": ["evidence","classify","image","gemini","aggregate"], "notes": "..." }
    """
    if not genai_client:
        return {"error": "Gemini not configured", "plan": None}
    prompt = f"""
You are an expert orchestrator assistant. Given a short core claim and the article excerpt (if any),
return a JSON plan enumerating which tools to run (from this set: {tools_available}) and why.
Respond JSON ONLY with keys:
- plan: array of steps (each step one of {tools_available})
- rationale: short explanation for the chosen order (1-2 sentences)
Constraints: prefer to fetch web evidence first when available; always run classification; run image checks only if image provided.
Input claim:
{claim}
Article excerpt (if available):
{article_text}
Example output:
{{"plan": ["evidence","classify","image","gemini","aggregate"], "rationale":"..."}}
"""
    try:
        resp = genai_client.models.generate_content(model=GENAI_MODEL, contents=prompt)
        raw = getattr(resp, "text", None) or str(resp)
        maybe = _extract_json_from_text(raw)
        if maybe:
            try:
                parsed = json.loads(maybe)
                return {"raw": raw, "plan": parsed.get("plan"), "rationale": parsed.get("rationale"), "parsed": parsed}
            except Exception as e:
                logger.exception("Planner JSON parse failed")
                return {"raw": raw, "parse_error": str(e)}
        else:
            return {"raw": raw, "error": "No JSON plan found"}
    except Exception as e:
        logger.exception("call_gemini_planner failed")
        return {"error": str(e), "trace": traceback.format_exc()}

def call_gemini_synthesizer(claim: str, article_text: str, tools_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask Gemini to synthesize a final verdict + explanation based on structured tool outputs.
    Request structured JSON:
    {
      "verdict": "True|False|Misleading|Unclear|Other",
      "score": 1..10,
      "explanation": "...",
      "issues": [{"issue":"", "explanation":""}],
      "recommendations": ["..."]
    }
    """
    if not genai_client:
        return {"error": "Gemini not configured"}
    # keep structured data JSON-serializable and reasonably sized
    tools_snapshot = json.dumps(tools_outputs, indent=0)[:15000]
    prompt = f"""
You are an expert fact-checker and synthesizer. Based on the following structured tool outputs (JSON),
produce a JSON object with fields:
- verdict: one of True, False, Misleading, Unclear
- score: integer 1..10 (reliability/confidence)
- explanation: a concise paragraph explaining the decision and evidence
- issues: up to 5 flagged issues (array of {{'issue':'', 'explanation':''}})
- recommendations: actionable next steps (array of strings)
Tool outputs (JSON):
{tools_snapshot}
Claim:
{claim}
Article excerpt:
{article_text}
Return JSON ONLY.
"""
    try:
        resp = genai_client.models.generate_content(model=GENAI_MODEL, contents=prompt)
        raw = getattr(resp, "text", None) or str(resp)
        maybe = _extract_json_from_text(raw)
        if maybe:
            try:
                parsed = json.loads(maybe)
                return {"raw": raw, "parsed": parsed}
            except Exception as e:
                logger.exception("Synthesizer JSON parse failed")
                return {"raw": raw, "parse_error": str(e)}
        else:
            return {"raw": raw, "error": "No JSON found in synthesizer output"}
    except Exception as e:
        logger.exception("call_gemini_synthesizer failed")
        return {"error": str(e), "trace": traceback.format_exc()}

# -----------------------------
# Evidence aggregation & heuristics (same as before)
# -----------------------------
def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname.lower()
    except Exception:
        return (url or "").lower()

def score_source_trust(domain: str) -> float:
    if not domain:
        return 0.4
    for k, v in SOURCE_TRUST.items():
        if k in domain:
            return float(v)
    parts = domain.split(".")
    tld = parts[-1] if parts else ""
    if tld in ["xyz", "top", "biz", "club", "online"]:
        return 0.35
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) or len(domain) > 40:
        return 0.3
    return 0.6

def aggregate_search_results(serpapi_result: dict) -> Dict[str, Any]:
    if not serpapi_result or not serpapi_result.get("available") or not serpapi_result.get("result"):
        return {"evidence": [], "consensus": None}
    res = serpapi_result["result"]
    organic = res.get("organic_results", []) or []
    evidence = []
    snippets_combined = ""
    domains = {}
    for r in organic[:8]:
        title = r.get("title")
        link = r.get("link")
        snippet = r.get("snippet") or ""
        domain = _domain_from_url(link or r.get("displayed_link", ""))
        trust = score_source_trust(domain)
        evidence.append({"title": title, "link": link, "snippet": snippet, "domain": domain, "trust": round(trust, 2)})
        domains[domain] = domains.get(domain, 0) + 1
        snippets_combined += " " + (title or "") + " " + (snippet or "")
    lc = snippets_combined.lower()
    contradiction_signals = ["not", "no evidence", "did not", "does not", "false", "debunked", "disputed"]
    contrad = any(sig in lc for sig in contradiction_signals)
    top3 = evidence[:3]
    avg_trust = (sum([e["trust"] for e in top3]) / len(top3)) if top3 else 0.5
    consensus = {"contradicts_claim": bool(contrad), "top_trust_avg": round(avg_trust, 2), "top_domains": domains}
    return {"evidence": evidence, "consensus": consensus}

def assess_event_validity(serpapi_result: dict) -> str:
    if not serpapi_result:
        return "Unknown"

    try:
        agg = aggregate_search_results(serpapi_result) or {}
    except Exception as e:
        logger.error("aggregate_search_results failed", exc_info=True)
        return "Unknown"

    evidence = agg.get("evidence") or []
    consensus = agg.get("consensus") or {}
    top_domains = consensus.get("top_domains") or {}

    trusted_count = sum(
        1 for d in top_domains.keys()
        if any(k in d for k in SOURCE_TRUST)
    )

    if trusted_count >= 2:
        return "True"

    if len(evidence) >= 1 and consensus.get("top_trust_avg", 0) >= 0.65:
        return "True"

    if len(evidence) >= 1:
        if consensus.get("contradicts_claim"):
            return "Unclear"
        return "Unclear"

    return "False"


def assess_framing(hf_labels: dict, gemini_parsed: dict) -> str:
    try:
        if gemini_parsed and isinstance(gemini_parsed, dict):
            issues = gemini_parsed.get("issues", []) or []
            for it in issues:
                title = (it.get("issue") or "").lower()
                if any(k in title for k in ["selective", "loaded", "unchallenged", "bias", "omission"]):
                    return "Misleading"
        if isinstance(hf_labels, dict) and "labels" in hf_labels:
            top = (hf_labels["labels"][0] or "").lower()
            if "opinionated" in top:
                return "Opinionated"
        return "Balanced"
    except Exception:
        return "Balanced"

# -----------------------------
# Orchestrator agent (enhanced with Gemini planner + synthesizer)
# -----------------------------
class Orchestrator:
    def __init__(self):
        pass

    def plan_tools(self, claim: str, has_image: bool, run_gemini: bool, run_serpapi: bool) -> Dict[str, Any]:
        # If Gemini is available and user asked, ask it for a plan
        available_tools = ["evidence", "classify", "image", "gemini", "aggregate"]
        tools_present = [t for t in available_tools if (t != "evidence" or run_serpapi) and (t != "gemini" or genai_client)]
        if run_gemini and genai_client:
            try:
                planner = call_gemini_planner(claim, "", tools_present)
                if planner and planner.get("plan"):
                    return {"plan": planner.get("plan"), "rationale": planner.get("rationale") or planner.get("raw")}
                # fallback to heuristic below
            except Exception:
                logger.exception("Gemini planner failed; falling back to heuristic plan")
        # fallback deterministic plan
        plan = []

        # If no claim but an image exists, process image first so OCR/caption can become the claim
        if has_image and not claim:
            plan.extend(["image", "classify"])
        else:
            if run_serpapi and SERPAPI_KEY:
                plan.append("evidence")
            plan.append("classify")
            if has_image:
                plan.append("image")

        if run_gemini and genai_client:
            plan.append("gemini")

        plan.append("aggregate")
        return {"plan": plan, "rationale": "Fallback heuristic plan"}

    def run(self, *,
            claim_text: str,
            article_text: Optional[str] = None,
            url: Optional[str] = None,
            image_upload: Optional[Image.Image] = None,
            image_url: Optional[str] = None,
            run_gemini: bool = True,
            run_serpapi: bool = True,
            run_safe_browsing: bool = False) -> Dict[str, Any]:

        claim_text = sanitize_text(claim_text or "")
        article_text = sanitize_text(article_text or "")
        url = url or None

        has_image = bool(image_upload or image_url)
        planner_out = self.plan_tools(claim_text, has_image, run_gemini, run_serpapi)
        plan = planner_out.get("plan") or []
        logger.info("Orchestrator plan: %s (rationale: %s)", plan, planner_out.get("rationale"))

        serpapi_result = None
        classifier_result = None
        gemini_raw = None
        gemini_parsed = None
        image_check = None
        tool_outputs = {}

        for step in plan:
            try:
                if step == "evidence":
                    q = claim_text or article_text or url or ""
                    serpapi_result = serpapi_web_search(q, num=6) if run_serpapi else {"available": False}
                    tool_outputs["serpapi_result"] = serpapi_result
                elif step == "classify":
                    classifier_result = hf_zero_shot(claim_text)
                    tool_outputs["classifier"] = classifier_result

                elif step == "image":
                    image_check = {}
                    # Process uploaded image (gradio returns PIL when configured type="pil")
                    if image_upload:
                        logger.info("Processing uploaded image: type=%s", type(image_upload))
                        try:
                            img = image_upload if isinstance(image_upload, Image.Image) else Image.open(image_upload).convert("RGB")
                            logger.info("Loaded PIL image from upload: mode=%s size=%s", img.mode, img.size)

                            # Save debug copy to /tmp for inspection (best-effort)
                            try:
                                ts = int(time.time())
                                tmp_path = f"/tmp/newsorchestra_debug_{ts}.png"
                                img.save(tmp_path)
                                logger.info("Saved uploaded image debug copy to: %s", tmp_path)
                                image_check["debug_saved_path"] = tmp_path
                            except Exception as e:
                                logger.warning("Failed to save debug image: %s", e)

                            image_check["exif"] = extract_exif(img)
                            image_check["ocr_text"] = image_ocr_text(img) if pytesseract else None
                            image_check["caption"] = hf_image_caption(img)
                            image_check["classification"] = hf_image_classify(img)
                            if run_serpapi and SERPAPI_KEY:
                                try:
                                    buff = BytesIO()
                                    img.save(buff, format="PNG")
                                    image_bytes = buff.getvalue()
                                    image_check["reverse_search"] = serpapi_reverse_image(image_bytes=image_bytes)
                                    image_check["raw_bytes_len"] = len(image_bytes)
                                except Exception as e:
                                    image_check["reverse_search_error"] = str(e)
                        except Exception as e:
                            logger.exception("Failed to open/process image_upload: %s", e)
                            image_check["fetch_error"] = f"open_failed: {e}"

                    elif image_url:
                        logger.info("Processing image from URL: %s", image_url)
                        image_check["fetched_from_url"] = image_url
                        img, b, err = fetch_image_bytes(image_url)
                        if b:
                            image_check["raw_bytes_len"] = len(b)
                        if err:
                            image_check["fetch_error"] = err
                        if run_serpapi and SERPAPI_KEY:
                            try:
                                if b:
                                    image_check["reverse_search"] = serpapi_reverse_image(image_bytes=b)
                                else:
                                    image_check["reverse_search"] = serpapi_reverse_image(image_url=image_url)
                            except Exception as e:
                                image_check["reverse_search_error"] = str(e)
                        if img:
                            # Save debug copy
                            try:
                                ts = int(time.time())
                                tmp_path = f"/tmp/newsorchestra_debug_{ts}.png"
                                img.save(tmp_path)
                                logger.info("Saved fetched image debug copy to: %s", tmp_path)
                                image_check["debug_saved_path"] = tmp_path
                            except Exception as e:
                                logger.warning("Failed to save fetched debug image: %s", e)
                            image_check["exif"] = extract_exif(img)
                            image_check["ocr_text"] = image_ocr_text(img) if pytesseract else None
                            image_check["caption"] = hf_image_caption(img)
                            image_check["classification"] = hf_image_classify(img)

                    tool_outputs["image_check"] = image_check

                elif step == "gemini":
                    # Reserve gemini for synthesis later; but optionally call an intermediate verifier
                    gemini_raw = call_gemini_synthesizer(claim_text, article_text, tool_outputs)
                    try:
                        gemini_parsed = gemini_raw.get("parsed") if isinstance(gemini_raw, dict) else None
                    except Exception:
                        gemini_parsed = None
                    tool_outputs["gemini_verifier"] = gemini_parsed or {"raw": gemini_raw}
                elif step == "aggregate":
                    # nothing here — aggregation below
                    pass
                else:
                    logger.info("Unknown plan step: %s", step)
            except Exception:
                logger.exception("Error during step %s", step)

        # compute misinfo heuristics
        misinfo = compute_misinfo_risk(classifier_result, serpapi_result, image_check, gemini_parsed or {"overall": "likely_clean"})
        tool_outputs["misinfo"] = misinfo

        # evidence aggregation & metrics
        evidence_agg = aggregate_search_results(serpapi_result)
        tool_outputs["evidence_agg"] = evidence_agg

        # uses heuristics to assess event validity & framing
        event_validity = assess_event_validity(serpapi_result)
        framing_quality = assess_framing(classifier_result, gemini_parsed)

        # If genai is available, ask Gemini synthesizer to compose the final verdict using tool_outputs
        synthesizer_out = None
        if genai_client:
            try:
                synthesizer_out = call_gemini_synthesizer(claim_text, article_text, tool_outputs)
                gemini_synth_parsed = synthesizer_out.get("parsed") if isinstance(synthesizer_out, dict) else None
            except Exception:
                logger.exception("Gemini synthesizer failed")
                gemini_synth_parsed = None
            # prefer synthesizer parsed content if present
            final_verdict = None
            confidence = None
            explanation = None
            if synthesizer_out and synthesizer_out.get("parsed"):
                parsed = synthesizer_out.get("parsed")
                final_verdict = parsed.get("verdict")
                try:
                    s = int(parsed.get("score", 0))
                    confidence = (s / 10.0) if s <= 10 else None
                except Exception:
                    confidence = None
                explanation = parsed.get("explanation")
            else:
                # no synthesizer parsed -> fallback composition
                final_verdict = None
                if event_validity == "True" and framing_quality == "Misleading":
                    final_verdict = "True but Misleading"
                elif event_validity == "True" and framing_quality in ("Balanced", None):
                    final_verdict = "True"
                elif event_validity == "False":
                    final_verdict = "False"
                else:
                    gm = gemini_parsed.get("verdict") if gemini_parsed else None
                    final_verdict = gm or (classifier_result["labels"][0] if classifier_result and isinstance(classifier_result, dict) and "labels" in classifier_result else "Unclear")
                # confidence inverse to misinfo score
                try:
                    confidence = round(max(0.0, min(1.0, 1.0 - float(misinfo.get("score")))), 3)
                except Exception:
                    confidence = None
                explanation = "; ".join([f"{it.get('issue')}: {it.get('explanation')}" for it in (gemini_parsed.get("issues") if gemini_parsed and isinstance(gemini_parsed, dict) else [])]) if gemini_parsed else None
        else:
            # no genai -> heuristic composition
            if event_validity == "True" and framing_quality == "Misleading":
                final_verdict = "True but Misleading"
            elif event_validity == "True" and framing_quality in ("Balanced", None):
                final_verdict = "True"
            elif event_validity == "False":
                final_verdict = "False"
            else:
                final_verdict = (classifier_result["labels"][0] if classifier_result and isinstance(classifier_result, dict) and "labels" in classifier_result else "Unclear")
            try:
                confidence = round(max(0.0, min(1.0, 1.0 - float(misinfo.get("score")))), 3)
            except Exception:
                confidence = None
            explanation = "; ".join([f"{it.get('issue')}: {it.get('explanation')}" for it in (gemini_parsed.get("issues") if gemini_parsed and isinstance(gemini_parsed, dict) else [])]) if gemini_parsed else None

        merged = {
            "claim": claim_text,
            "article_text": safe_truncate(article_text, 4000) if article_text else None,
            "url": url,
            "classifier": classifier_result,
            "verifier": gemini_parsed or None,
            "gemini_raw": gemini_raw,
            "serpapi_result": serpapi_result,
            "evidence_agg": evidence_agg,
            "image_check": image_check,
            "misinfo_risk": misinfo,
            "event_validity": event_validity,
            "framing_quality": framing_quality,
            "final_verdict": final_verdict,
            "confidence": confidence,
            "explanation": explanation,
            "planner": planner_out,
            "synthesizer": synthesizer_out
        }
        return merged

# Instantiate orchestrator
ORCH = Orchestrator()

# -----------------------------
# Gradio UI wiring
# -----------------------------
title = "NewsOrchestra — Multi-agent AI Orchestrator for Explainable Fact-checking"
description = """Paste article text or a URL and optionally upload an image.  
This demo orchestrates multiple models (zero-shot classifier, SerpApi evidence fetch, optional Gemini planner/synthesizer) and returns a verdict and human-friendly explanation."""

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}\n\n{description}")
    with gr.Row():
        inp = gr.Textbox(lines=8, label="Article text or URL", placeholder="Paste full article text or a URL (http...)")
    with gr.Row():
        img_upload = gr.Image(type="pil", label="Upload image (optional)")
        img_url = gr.Textbox(lines=1, label="Or image URL (optional)", placeholder="https://...")
    with gr.Row():
        run_gemini_cb = gr.Checkbox(label="Run Gemini planner + synthesizer (if configured)", value=True)
        run_serp_cb = gr.Checkbox(label="Run SerpApi evidence fetch (recommended)", value=bool(SERPAPI_KEY))
    with gr.Row():
        analyze_btn = gr.Button("Analyze (run orchestration)")
    with gr.Row():
        merged_out = gr.JSON(label="Merged verification report (JSON)")
    with gr.Row():
        human_md = gr.Markdown(label="Friendly explanation & evidence")


    def on_click_analyze(text_or_url: str = "", image_upload: Optional[bytes] = None, image_url: str = "", run_gemini: bool = True, run_serpapi: bool = True):
        """
        Gradio handler. Receives:
          - text_or_url: textbox content (URL or article text)
          - image_upload: PIL.Image when gr.Image(type="pil") else bytes/file-like (best-effort)
          - image_url: string
          - run_gemini: bool
          - run_serpapi: bool
        Returns (merged_json, markdown_string)
        """
        try:
            def is_url_only(s: str) -> bool:
                if not s:
                    return False
                s = s.strip()
                return bool(re.match(r"^https?://\S+$", s))

            raw_input = (text_or_url or "").strip()
            mode_is_url = is_url_only(raw_input)

            article_text = None
            url = None
            claim = None

            logger.info("on_click_analyze called: text_len=%s image_upload=%s image_url=%s run_gemini=%s run_serpapi=%s",
                        len(raw_input), type(image_upload).__name__ if image_upload is not None else None, bool(image_url), run_gemini, run_serpapi)

            # URL-only input → fetch article and extract claim
            if mode_is_url:
                url = raw_input
                article_text = fetch_article_text_from_url(url)
                claim = extract_claim_from_text(article_text) or ""
            else:
                # treat input as article text (may be empty)
                article_text = raw_input or None
                if article_text:
                    claim = extract_claim_from_text(article_text) or article_text.split("\n", 1)[0][:800]
                else:
                    # No text provided: attempt to derive claim from image (OCR -> caption)
                    img_obj = None

                    # image_upload might already be a PIL.Image (gradio type="pil")
                    if image_upload:
                        try:
                            if isinstance(image_upload, Image.Image):
                                img_obj = image_upload
                            else:
                                # try bytes -> PIL
                                if isinstance(image_upload, (bytes, bytearray)):
                                    img_obj = Image.open(BytesIO(image_upload)).convert("RGB")
                                else:
                                    # try to open file-like
                                    try:
                                        img_obj = Image.open(image_upload).convert("RGB")
                                    except Exception:
                                        img_obj = None
                        except Exception as e:
                            logger.exception("Error opening image_upload: %s", e)
                            img_obj = None

                    # If still no PIL, try to fetch from image_url string
                    if img_obj is None and image_url:
                        img_obj, b, err = fetch_image_bytes(image_url)
                        if err:
                            logger.info("fetch_image_bytes returned error: %s", err)

                    # Attempt OCR first, then caption
                    if img_obj:
                        ocr_text = image_ocr_text(img_obj) if pytesseract else None
                        if ocr_text:
                            claim = ocr_text.strip()
                        else:
                            caption = hf_image_caption(img_obj)
                            if caption:
                                claim = caption.strip()

                    claim = claim or "No claim text provided"

            # Prefer to pass PIL.Image to ORCH.run when available
            # If image_upload param is bytes but we created img_obj above, pass img_obj
            image_to_pass = None
            if isinstance(image_upload, Image.Image):
                image_to_pass = image_upload
            else:
                # if we created img_obj from bytes/url above, prefer that
                try:
                    if 'img_obj' in locals() and isinstance(img_obj, Image.Image):
                        image_to_pass = img_obj
                    else:
                        image_to_pass = image_upload
                except Exception:
                    image_to_pass = image_upload

            merged = ORCH.run(
                claim_text=claim,
                article_text=article_text,
                url=url,
                image_upload=image_to_pass,
                image_url=image_url,
                run_gemini=run_gemini,
                run_serpapi=run_serpapi
            )

            # Build human-friendly markdown (same format as before)
            md_lines = []
            md_lines.append(f"### Claim\n\n> {merged.get('claim')}\n")
            md_lines.append(f"**Final verdict:** **{merged.get('final_verdict')}**  ")

            conf = merged.get('confidence')
            md_lines.append(f"**Confidence:** {(conf*100):.1f}%  " if conf is not None else "**Confidence:** —  ")
            md_lines.append(f"**Event validity:** {merged.get('event_validity')}  ")
            md_lines.append(f"**Framing quality:** {merged.get('framing_quality')}  ")
            md_lines.append("\n---\n")

            # Top evidence
            ea = merged.get("evidence_agg", {}).get("evidence", [])
            if ea:
                md_lines.append("### Top evidence (click to open)\n")
                for e in ea[:6]:
                    title = e.get('title') or e.get('snippet') or "Result"
                    link_display = "🔗" if e.get('link') else ""
                    link_url = e.get('link') or "#"
                    md_lines.append(f"- **[{e.get('domain')}]** [{link_display}] [{title}]({link_url}) — trust **{e.get('trust')}**  \n  > {e.get('snippet')}\n")
            else:
                md_lines.append("- No top search results found.\n")

            md_lines.append("\n---\n")

            if merged.get("explanation"):
                md_lines.append("### Why we reached this verdict\n")
                md_lines.append(merged.get("explanation") + "\n")

            # Planner rationale
            if merged.get("planner"):
                md_lines.append("\n---\n**Orchestrator planner rationale:**\n")
                md_lines.append("```\n" + json.dumps(merged.get("planner"), indent=2) + "\n```\n")

            # Synthesizer output
            if merged.get("synthesizer"):
                md_lines.append("\n---\n**Synthesizer output (Gemini):**\n")
                md_lines.append("```\n" + json.dumps(merged.get("synthesizer"), indent=2) + "\n```\n")

            md = "\n".join(md_lines)
            return merged, md

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Error in analyze handler: %s", e)
            return {"error": str(e), "trace": tb}, f"Error running analysis: {e}"

    analyze_btn.click(on_click_analyze, inputs=[inp, img_upload, img_url, run_gemini_cb, run_serp_cb],
                      outputs=[merged_out, human_md])

    gr.Markdown(
        "- Notes: set SERPAPI_KEY and GEMINI_API_KEY in environment for best results.\n"
        "- The demo is for research/educational purposes. Always human-review outputs before acting on them."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)