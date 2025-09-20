import os
import re
import json
import logging
import traceback
import time
import io
import socket
import ipaddress
from typing import Optional, Dict, Any, Tuple, List
from urllib.parse import urlparse
from functools import lru_cache
from collections import Counter

import requests
from bs4 import BeautifulSoup
import gradio as gr
from transformers import pipeline

# IMAGE libs
from PIL import Image, ImageChops, ImageStat, ExifTags
import imagehash

# --- GEMINI--
try:
    from google import genai
except Exception:
    raise SystemExit("gemini (genai) Python client not installed. Run: pip install genai")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit("GEMINI_API_KEY env var is required. Export it before running the app.")

# Initialize Gemini client
try:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    raise SystemExit(f"Failed to init genai client: {e}")

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# --- Transformers auxiliary ML ---
HF_ZERO_SHOT = os.getenv("HF_ZERO_SHOT", "facebook/bart-large-mnli")
try:
    zero_shot = pipeline("zero-shot-classification", model=HF_ZERO_SHOT)
except Exception as e:
    zero_shot = None
    logging.warning("Zero-shot unavailable: %s", e)

# config
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("newsorchestra_gemini")
SAFE_BROWSING_KEY = os.getenv("SAFE_BROWSING_KEY")
VIRUSTOTAL_KEY = os.getenv("VIRUSTOTAL_KEY")

CANDIDATE_LABELS = ["True", "False", "Misleading", "Unclear", "Opinionated", "Unsupported"]

SOURCE_TRUST = {
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "bbc.com": 0.93,
    "theguardian.com": 0.9,
    "nytimes.com": 0.9,
    "washingtonpost.com": 0.9,
}


# Helpers

def compute_modal_accuracy(verdicts: list, true_labels: list) -> float:
    
    if not verdicts or not true_labels or len(verdicts) != len(true_labels):
        return 0.0
    correct = 0
    for v_list, true in zip(verdicts, true_labels):
        if not v_list:
            continue
        mode = Counter(v_list).most_common(1)[0][0]
        if mode == true:
            correct += 1
    return correct / len(true_labels)

def google_safe_browsing_check(url: str, api_key: str) -> dict:
    try:
        endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
        body = {
            "client": {"clientId": "newsorchestra", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": [
                    "MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"
                ],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}]
            }
        }
        r = requests.post(endpoint, json=body, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "matches" in data:
            return {"safe": False, "matches": data["matches"]}
        return {"safe": True, "matches": []}
    except Exception as e:
        return {"safe": None, "error": str(e)}

def virustotal_url_check(url: str, api_key: str) -> dict:
    try:
        headers = {"x-apikey": api_key}
        import base64
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        vt_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        r = requests.get(vt_url, headers=headers, timeout=15)
        if r.status_code == 404:
            scan_r = requests.post("https://www.virustotal.com/api/v3/urls", headers=headers, data={"url": url}, timeout=15)
            scan_r.raise_for_status()
            return {"safe": None, "submitted": True}
        r.raise_for_status()
        data = r.json()
        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        safe = malicious == 0 and suspicious == 0
        return {"safe": safe, "malicious_votes": malicious, "suspicious_votes": suspicious}
    except Exception as e:
        return {"safe": None, "error": str(e)}

def sanitize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _extract_json_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        return m.group(1)
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

def _safe_parse_gemini_json(raw_text: str) -> Optional[dict]:
    jstr = _extract_json_from_text(raw_text)
    if not jstr:
        return None
    try:
        return json.loads(jstr)
    except Exception:
        return None

def _domain_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host.lower().lstrip("www.")
    except Exception:
        return ""

def _is_host_public(url: str) -> bool:
    
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not host:
            return False
        host = host.strip().lower()
        if host in ("localhost", "ip6-localhost", "::1"):
            return False
        # If host is an IP literal
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # resolve DNS once
            try:
                infos = socket.getaddrinfo(host, None)
                addr = infos[0][4][0]
                ip = ipaddress.ip_address(addr)
            except Exception:
                # If we can't resolve, treat as non-public to be conservative
                return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return False
        return True
    except Exception:
        return False

def phishing_checks(url: str) -> dict:
    if not url:
        return {}
    out = {"url": url, "safe_browsing": None, "virustotal": None}
    if SAFE_BROWSING_KEY:
        out["safe_browsing"] = google_safe_browsing_check(url, SAFE_BROWSING_KEY)
    if VIRUSTOTAL_KEY:
        out["virustotal"] = virustotal_url_check(url, VIRUSTOTAL_KEY)
    return out


@lru_cache(maxsize=256)
def serpapi_web_search(query: str, num: int = 6) -> dict:
    if not SERPAPI_KEY:
        return {"available": False, "note": "SERPAPI_KEY not set"}
    try:
        r = requests.get("https://serpapi.com/search.json",
                         params={"engine": "google", "q": query, "num": num, "api_key": SERPAPI_KEY},
                         timeout=12)
        r.raise_for_status()
        return {"available": True, "result": r.json()}
    except Exception as e:
        logger.exception("SerpApi search failed")
        return {"available": True, "error": str(e)}

@lru_cache(maxsize=256)
def serpapi_reverse_image(image_url: str, num: int = 6) -> dict:
    if not SERPAPI_KEY:
        return {"available": False, "note": "SERPAPI_KEY not set"}
    try:
        r = requests.get("https://serpapi.com/search.json",
                         params={"engine": "google", "q": image_url, "num": num, "api_key": SERPAPI_KEY},
                         timeout=12)
        r.raise_for_status()
        return {"available": True, "result": r.json()}
    except Exception as e:
        logger.exception("SerpApi reverse failed")
        return {"available": True, "error": str(e)}


# Image analysis helpers

MAX_BYTES = 6 * 1024 * 1024  # 6MB
ALLOWED_CONTENT_PREFIXES = ("image/",)

def download_image_bytes(url: str, timeout: int = 12) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        if not _is_host_public(url):
            logger.warning("Blocked image download for private/local host: %s", url)
            return None, None

        with requests.get(url, timeout=timeout, stream=True, headers={"User-Agent": "newsorchestra/1.0"}) as r:
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            if not any(ct.startswith(p) for p in ALLOWED_CONTENT_PREFIXES):
                logger.warning("Rejected non-image content-type: %s", ct)
                return None, ct
            buf = io.BytesIO()
            total = 0
            for chunk in r.iter_content(8192):
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    logger.warning("Image too large (%d bytes)", total)
                    return None, ct
                buf.write(chunk)
            return buf.getvalue(), ct
    except Exception as e:
        logger.warning("download_image_bytes failed: %s", e)
        return None, None

def extract_exif_from_bytes(img_bytes: bytes) -> dict:
    out = {"has_exif": False, "exif": {}, "has_gps": False}
    try:
        img = Image.open(io.BytesIO(img_bytes))
        exif_raw = getattr(img, "_getexif", lambda: None)()
        if not exif_raw:
            return out
        exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
        out["has_exif"] = True
        if "GPSInfo" in exif:
            out["has_gps"] = True
            exif.pop("GPSInfo", None)
        out["exif"] = exif
        return out
    except Exception:
        return out

def error_level_analysis_score(img_bytes: bytes, quality: int = 90) -> dict:
    out = {"available": False}
    try:
        orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        orig.save(buf, "JPEG", quality=quality)
        recompr = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        diff = ImageChops.difference(orig, recompr)
        stat = ImageStat.Stat(diff)
        mean_val = sum(stat.mean)/len(stat.mean)
        out.update({"available": True, "ela_score": round(float(mean_val), 3)})
        return out
    except Exception:
        return out

def compute_phash(img_bytes: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        ph = imagehash.phash(img)
        return {"available": True, "phash": str(ph)}
    except Exception:
        return {"available": False}

def analyze_image_url(image_url: str) -> dict:
    result = {"image_url": image_url, "fetched": False}
    b, ct = download_image_bytes(image_url)
    if not b:
        result["error"] = "download failed"
        return result
    result["fetched"] = True
    result["content_type"] = ct
    result["bytes_length"] = len(b)
    result["exif"] = extract_exif_from_bytes(b)
    result["ela"] = error_level_analysis_score(b)
    result["phash"] = compute_phash(b)
    result["serpapi_reverse"] = serpapi_reverse_image(image_url) if SERPAPI_KEY else {"available": False}
    return result


# Gemini functions

GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-2.5-flash")

def gemini_generate_claim_from_image(image_url: str) -> Optional[str]:
    try:
        img_bytes, _ = download_image_bytes(image_url)
        if not img_bytes:
            return None
        img = Image.open(io.BytesIO(img_bytes))

        prompt = (
            "You are a cautious fact-check assistant.\n"
            "Look at the image and, ONLY IF you can identify a plausible short factual claim about the main subject, "
            "return a JSON object **ONLY** inside triple backticks, with the exact keys: claim, rationale.\n\n"
            "Rules:\n"
            "- If you can propose a factual testable claim, set \"claim\" to a short sentence (<= 140 chars) starting with "
            "\"Auto-generated (unverified):\" and use cautious phrasing like 'appears to show' or 'is claimed to show'.\n"
            "- If you cannot identify a testable factual claim, set \"claim\": null and provide a short rationale.\n"
            "- DO NOT output any prose outside the fenced JSON block.\n\n"
            "Example output:\n"
            "```json\n"
            "{\"claim\": \"Auto-generated (unverified): The photo appears to show the mayor speaking at the flood site.\", "
            "\"rationale\": \"person at podium, banner text, context implies event\"}\n"
            "```\n"
        )
        resp = genai_client.models.generate_content(
            model=GENAI_MODEL,
            contents=[prompt, img]
        )
        raw = getattr(resp, "text", None) or str(resp)
        parsed = _safe_parse_gemini_json(raw)
        if parsed is not None:
            claim = parsed.get("claim")
            rationale = parsed.get("rationale", "")
            if claim:
                return sanitize_text(claim)[:400]
            if rationale:
                return f"Auto-generated (unverified): Image provided; no clear factual claim. Rationale: {sanitize_text(rationale)[:240]}"
            return None
    except Exception:
        logger.exception("Gemini multimodal claim gen failed")
    return None

def gemini_extract_claims_from_text(article_text: str, max_claims: int = 3) -> List[Dict[str, str]]:
    article_text = sanitize_text(article_text or "")
    if not article_text:
        return []
    prompt = (
        "You are a cautious fact-check assistant. From the following article text, extract up to "
        f"{max_claims} concise, testable factual claims that a fact-checker could verify. "
        "Return ONLY a single fenced JSON block (```json ... ```). The JSON object must have key `claims` "
        "which is a list of objects with `claim` (short sentence <=140 chars) and `context` (short context snippet).\n\n"
        "If the article contains no testable factual claims, return {\"claims\": []}.\n\n"
        "Article:\n"
        "```\n"
        f"{article_text[:12000]}\n"
        "```\n"
    )
    try:
        resp = genai_client.models.generate_content(model=GENAI_MODEL, contents=[prompt])
        raw = getattr(resp, "text", None) or str(resp)
        j = _extract_json_from_text(raw)
        if j:
            parsed = json.loads(j)
            claims = parsed.get("claims") or []
            out = []
            for c in claims[:max_claims]:
                claim_text = sanitize_text(c.get("claim", ""))[:800]
                context = sanitize_text(c.get("context", ""))[:400]
                if claim_text:
                    out.append({"claim": claim_text, "context": context})
            return out
    except Exception:
        logger.exception("Gemini extract claims failed")
    try:
        sents = re.split(r'(?<=[.!?])\s+', article_text)
        out = []
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) > 30:
                out.append({"claim": s_clean[:800], "context": s_clean[:400]})
            if len(out) >= max_claims:
                break
        return out
    except Exception:
        return []

def build_evidence_snippet(serpapi_web: dict, image_analysis: dict) -> str:
    out = ""
    try:
        if serpapi_web and serpapi_web.get("result"):
            organic = serpapi_web["result"].get("organic_results", []) or []
            pieces = []
            for r in organic[:8]:
                pieces.append(f"{r.get('title','')} :: {r.get('snippet','')} :: {r.get('link','')}")
            if pieces:
                out += "WEB EVIDENCE:\n" + "\n".join(pieces)
        if image_analysis and image_analysis.get("serpapi_reverse", {}).get("result"):
            rorg = image_analysis["serpapi_reverse"]["result"].get("organic_results", []) or []
            pieces = []
            for r in rorg[:6]:
                pieces.append(f"{r.get('title','')} :: {r.get('snippet','')} :: {r.get('link','')}")
            if pieces:
                out += "\nREVERSE IMAGE EVIDENCE:\n" + "\n".join(pieces)
    except Exception:
        logger.exception("Building evidence snippet failed")
    return out

def gemini_verify_claim(claim: str, serpapi_web: dict, image_analysis: dict) -> Dict[str, Any]:
    if not claim:
        return {"verdict": "Unclear", "overall": "No claim provided", "issues": [], "citations": []}
    evidence_snippet = build_evidence_snippet(serpapi_web, image_analysis)
    prompt = (
        "You are a cautious fact-checker. Evaluate the claim and available evidence.\n"
        "Return ONLY a single fenced JSON block with keys: verdict, overall, issues, citations.\n"
        "verdict must be one of: True, False, Mixed, Unsupported, Unclear, Misleading.\n"
        "citations should be a list of objects {source, snippet, url} if possible.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Evidence (may be empty):\n{evidence_snippet}\n\n"
        "Be concise. If you cannot reach a conclusion, use 'Unclear' or 'Unsupported'.\n"
    )
    try:
        contents = [prompt]
        if image_analysis and image_analysis.get("fetched"):
            try:
                img_bytes, _ = download_image_bytes(image_analysis["image_url"])
                if img_bytes:
                    contents.append(Image.open(io.BytesIO(img_bytes)))
            except Exception:
                logger.exception("Attaching image to Gemini verify failed")
        resp = genai_client.models.generate_content(model=GENAI_MODEL, contents=contents)
        raw = getattr(resp, "text", None) or str(resp)
        parsed = _safe_parse_gemini_json(raw)
        if parsed:
            return parsed
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Gemini verify produced unparsable output: %s", raw[:400])
            return {"verdict": "Unclear", "overall": raw[:400], "issues": ["unparsed"], "citations": []}
    except Exception:
        logger.exception("Gemini verify failed")
        return {"verdict": "Unclear", "overall": "Gemini failure", "issues": ["gemini_failure"], "citations": []}


# HF zero-shot

def hf_zero_shot_classify(claim: str) -> Dict[str, Any]:
    if not zero_shot or not claim:
        return {"error": "hf-unavailable"}
    try:
        return zero_shot(claim, candidate_labels=CANDIDATE_LABELS, multi_label=False)
    except Exception:
        logger.exception("HF zero-shot failed")
        return {"error": "hf-failed"}


# Aggregator & combiner

def aggregate_search_results(serpapi_result: dict) -> Dict[str, Any]:
    if not serpapi_result or not serpapi_result.get("available") or not serpapi_result.get("result"):
        return {"evidence": [], "consensus": {"contradicts_claim": False, "top_trust_avg": 0.5, "top_domains": {}}, "raw_snippets": ""}
    res = serpapi_result["result"]
    organic = res.get("organic_results", []) or []
    evidence = []
    domains = {}
    for r in organic[:12]:
        title = r.get("title") or ""
        snippet = r.get("snippet") or ""
        link = r.get("link") or r.get("displayed_link") or ""
        domain = _domain_from_url(link)
        trust = SOURCE_TRUST.get(domain, 0.6)
        evidence.append({"title": title, "snippet": snippet, "link": link, "domain": domain, "trust": round(trust, 2)})
        if domain:
            domains[domain] = domains.get(domain, 0) + 1
    top3 = evidence[:3]
    top_trust_avg = sum([e["trust"] for e in top3]) / len(top3) if top3 else 0.5
    return {"evidence": evidence,
            "consensus": {"contradicts_claim": False, "top_trust_avg": round(top_trust_avg, 2), "top_domains": domains},
            "raw_snippets": " ".join([e["title"] + " " + e["snippet"] for e in evidence])[:4000]}

def _map_gemini_verdict_to_score(v: str) -> float:
    if not v:
        return 0.0
    vv = v.lower()
    if vv == "true":
        return 1.0
    if vv == "false":
        return -1.0
    if vv in ("mixed", "misleading"):
        return -0.2
    if vv in ("unsupported", "unclear"):
        return 0.0
    return 0.0

def _map_hf_label_to_score(hf_result: dict) -> float:
    try:
        if not hf_result or "labels" not in hf_result:
            return 0.0
        top = hf_result["labels"][0].lower()
        if top == "true":
            return 0.6
        if top == "false":
            return -0.6
        if top == "unsupported":
            return -0.7
        if top == "misleading":
            return -0.4
        return 0.0
    except Exception:
        return 0.0

def combine_signals(gemini_verdict: dict, hf_result: dict, evidence_agg: dict) -> Dict[str, Any]:
    reasons = []
    g_ver = (gemini_verdict or {}).get("verdict", "Unclear")
    g_overall = (gemini_verdict or {}).get("overall", "")
    g_issues = (gemini_verdict or {}).get("issues", []) or []

    top_trust = evidence_agg.get("consensus", {}).get("top_trust_avg", 0.5)
    evidence_count = len(evidence_agg.get("evidence", []))

    g_score = _map_gemini_verdict_to_score(g_ver)
    hf_score = _map_hf_label_to_score(hf_result)
    trust_norm = (top_trust - 0.5) * 2.0

    # weights
    w_g = 0.55
    w_h = 0.2
    w_e = 0.25

    final_score = w_g * g_score + w_h * hf_score + w_e * trust_norm
    confidence = min(0.99, max(0.05, 0.4 + abs(final_score) * 0.6))

    if final_score >= 0.45:
        label = "True"
        reasons.append("Aggregated signals indicate likely truth")
    elif final_score <= -0.45:
        label = "False"
        reasons.append("Aggregated signals indicate likely falsehood")
    else:
        if evidence_count >= 2 and hf_score < 0 and final_score < 0.2:
            label = "Misleading"
            reasons.append("Evidence / classifier suggest partial inaccuracy or omission")
        else:
            label = "Unclear"
            reasons.append("Insufficient agreement between models and web evidence")

    if evidence_count >= 2 and top_trust >= 0.7:
        if label in ("Unclear", "Misleading"):
            reasons.append("Multiple high-trust outlets corroborate the core event")
            label = "True"
    if g_issues:
        reasons.extend(g_issues if isinstance(g_issues, list) else [str(g_issues)])
    if g_overall:
        reasons.append(f"Gemini note: {g_overall[:240]}")

    return {"final_verdict": label, "confidence": round(confidence, 3), "reasons": reasons, "final_score": round(final_score, 3)}


# Q/A formatting 

def _trust_score_pct_from_final_score(final_score: float) -> int:
    """Map final_score (-1..1) to 0..100; clamp."""
    try:
        pct = int((final_score + 1.0) * 50.0)
        pct = max(0, min(100, pct))
        return pct
    except Exception:
        return 50

def format_user_friendly_explanation(report_entry: dict) -> str:

    def _reason_to_text(r) -> str:
        try:
            if r is None:
                return ""
            if isinstance(r, str):
                return r.strip()
            if isinstance(r, dict):
                for key in ("reason", "message", "detail", "issue", "note"):
                    if key in r and r[key]:
                        return str(r[key])[:300]
                try:
                    return json.dumps(r, ensure_ascii=False)[:300]
                except Exception:
                    return str(r)[:300]
            return str(r)[:300]
        except Exception:
            return ""

    claim = report_entry.get("claim", "").strip() or "(no claim provided)"

    # Q1: Why — 
    reasons = report_entry.get("reasons", []) or []
    if isinstance(reasons, (str, dict)):
        reasons = [reasons]

    reason_texts = []
    for r in reasons[:3]:
        t = _reason_to_text(r)
        if t:
            reason_texts.append(t)
    if reason_texts:
        reasons_text = "; ".join(reason_texts)
    else:
        gem_notes = (report_entry.get("gemini_verdict") or {}).get("overall", "")
        reasons_text = gem_notes[:300] if gem_notes else "No strong model reasons were returned."

    q1 = f"Q1: Why did we reach this verdict?\nA: {reasons_text}"

    # Q2: How was it verified? — list up to 3 top sources and performed checks
    evidence_agg = report_entry.get("evidence_agg", {}) or {}
    evidence = evidence_agg.get("evidence", []) or []
    top_sources = []
    for e in evidence[:3]:
        domain = e.get("domain") or _domain_from_url(e.get("link") or "")
        title = e.get("title") or ""
        link = e.get("link") or ""
        if link:
            top_sources.append(f"{domain}: {title[:120]} ({link})")
        else:
            top_sources.append(f"{domain}: {title[:120]}")

    top_sources_text = "\n- ".join(top_sources) if top_sources else "No strong web sources found."

    checks = []
    if report_entry.get("gemini_verdict"):
        checks.append("Gemini model analysis")
    hf = report_entry.get("hf_classifier")
    if hf and isinstance(hf, dict) and "labels" in hf:
        checks.append("HF zero-shot classifier")
    if report_entry.get("image_analysis") and report_entry["image_analysis"].get("fetched"):
        checks.append("Image analysis (EXIF / ELA / pHash / reverse-image)")
    phish = report_entry.get("phishing_analysis") or {}
    sb = (phish.get("safe_browsing") or {})
    vt = (phish.get("virustotal") or {})
    phish_notes = []
    try:
        if sb and sb.get("safe") is False:
            phish_notes.append("Safe Browsing flagged the site")
        if vt and vt.get("safe") is False:
            phish_notes.append("VirusTotal flagged the site")
        if not phish_notes and (sb or vt):
            phish_notes.append("Phishing checks performed (no clear flags)")
    except Exception:
        pass
    if phish_notes:
        checks.append("; ".join(phish_notes))

    checks_text = ", ".join(checks) if checks else "Model and web-snippet analysis (no special checks detected)."

    q2_lines = [
        "Q2: How was it verified?",
        "A: Verified by:",
        f"- Top web references (up to 3):\n- {top_sources_text}" if top_sources else f"- Top web references: {top_sources_text}",
        f"- Automated checks: {checks_text}"
    ]
    q2 = "\n".join(q2_lines)

    # Q3: What should you do next? — concise, actionable advice
    next_steps = []
    if top_sources:
        next_steps.append("Read the listed sources for full context and check publication dates.")
        next_steps.append("Cross-check with official channels (government, company, or primary source).")
    else:
        next_steps.append("No strong sources found — seek independent confirmation from trusted outlets before sharing.")
        next_steps.append("If this concerns safety or fraud, check official alerts or regulator pages.")

    # If phishing checks flagged the site, emphasize safety first
    if (sb.get("safe") is False) or (vt.get("safe") is False):
        next_steps.insert(0, "Do NOT click links from this page; treat it as potentially unsafe and report it.")

    q3 = "Q3: What should you do next?\nA: " + " ".join([f"- {s}" for s in next_steps])

    return f"{q1}\n\n{q2}\n\n{q3}"


def _extract_jsonld_from_soup(soup: BeautifulSoup) -> Optional[dict]:
    try:
        scripts = soup.find_all("script", type="application/ld+json")
        for s in scripts:
            try:
                txt = s.string or s.get_text()
                if not txt or not txt.strip():
                    continue
                parsed = json.loads(txt)
                # parsed may be dict or list
                items = parsed if isinstance(parsed, list) else [parsed]
                for item in items:
                    # sometimes nested graph
                    if isinstance(item, dict) and item.get("@type") in ("NewsArticle", "Article", "Report"):
                        return item
                    # handle @graph
                    if isinstance(item, dict) and "@graph" in item and isinstance(item["@graph"], list):
                        for g in item["@graph"]:
                            if isinstance(g, dict) and g.get("@type") in ("NewsArticle", "Article", "Report"):
                                return g
            except Exception:
                continue
    except Exception:
        pass
    return None

def fetch_article_text_from_url(url: str) -> tuple[str, str]:
    try:
        if not _is_host_public(url):
            logger.warning("Blocked fetch_article_text_from_url for private host: %s", url)
            return "", ""

        headers = {"User-Agent": "newsorchestra/1.0"}
        html = ""
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=10, headers=headers)
                r.raise_for_status()
                html = r.text
                break
            except requests.RequestException as e:
                logger.debug("fetch attempt %s failed for %s: %s", attempt + 1, url, e)
                html = ""
                if attempt == 1:
                    raise

        if not html:
            return "", ""

        soup = BeautifulSoup(html, "html.parser")

        # 1) JSON-LD extraction (best for MSN and other modern publishers)
        jld = _extract_jsonld_from_soup(soup)
        if jld:
            headline = jld.get("headline") or jld.get("name") or ""
            body = jld.get("articleBody") or jld.get("description") or ""
            if isinstance(body, list):
                body = " ".join([str(x) for x in body if x])
            if body:
                return sanitize_text(str(body)), sanitize_text(str(headline) or "")

        # 2) OpenGraph / meta
        og_title = (soup.find("meta", property="og:title") or {}).get("content")
        og_desc = (soup.find("meta", property="og:description") or {}).get("content")
        if og_desc:
            return sanitize_text(og_desc), sanitize_text(og_title or "")

        # 3) readability fallback if available
        try:
            from readability import Document 
            doc = Document(html)
            article_html = doc.summary()
            headline = doc.short_title() or ""
            soup2 = BeautifulSoup(article_html, "html.parser")
            paras = [p.get_text(" ", strip=True) for p in soup2.find_all("p")]
            article_text = "\n\n".join([p for p in paras if len(p) > 30])
            if article_text:
                return article_text, headline
        except Exception:
            logger.debug("readability extraction not available or failed; using BeautifulSoup fallback")

        # 4) BeautifulSoup fallback
        article_tag = soup.find("article")
        if article_tag:
            paras = [p.get_text(" ", strip=True) for p in article_tag.find_all("p")]
        else:
            main = soup.find("main") or soup.find(id="main") or soup.find(class_="article") or soup
            paras = [p.get_text(" ", strip=True) for p in main.find_all("p")]
        article_text = "\n\n".join([p for p in paras if len(p) > 40])
        headline = soup.title.get_text(strip=True) if soup.title else ""

        # 5) fallback to meta description if no body text
        if not article_text:
            meta = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
            if meta and meta.get("content"):
                article_text = meta.get("content", "")

        return article_text or "", headline or ""
    except Exception:
        logger.exception("fetch_article_text_from_url failed")
        return "", ""


# on_analyze handler
# - uses SERP fallback for snippets when article extraction fails
# - surfaces QA fallback note
# - infers phishing_tag for frontend convenience

def on_analyze(text_or_url: str, image_url: str, run_serp: bool):
    try:
        txt = (text_or_url or "").strip()
        is_url = bool(re.match(r"^https?://", txt))
        article_text, headline, url = "", "", None
        qa_fallback_note = ""

        if is_url:
            url = txt
            article_text, headline = fetch_article_text_from_url(txt)
            if not article_text and headline and run_serp and SERPAPI_KEY:
                serpapi_result = serpapi_web_search(headline, num=8)
                snippets = [res.get("snippet", "") for res in serpapi_result.get("result", {}).get("organic_results", [])]
                serp_text = "\n\n".join([s for s in snippets if s])[:3000]
                if serp_text:
                    article_text = f"(SERP fallback - extracted snippets for headline: {headline})\n\n{serp_text}"
                    qa_fallback_note = (
                        "Note: we couldn't extract full article text from the URL. "
                        "Analysis used SERP snippets as a fallback — verify date/location in original sources."
                    )
                else:
                    article_text = f"(No extractable article text) Headline: {headline}"
                    qa_fallback_note = (
                        "Note: no article text could be extracted; analysis used the page headline only."
                    )
        else:
            article_text = txt or ""

        claim = ""  
        report = ORCH.run(
            claim_text=claim,
            article_text=article_text,
            url=url,
            image_url=image_url or None,
            run_serpapi=run_serp,
        )

        extracted_claims = [r.get("claim") for r in report.get("reports", [])]
        qa_text = ""
        if report.get("reports"):
            qa_text = report["reports"][0].get("qa_summary", "") or ""
            if (not qa_text) and qa_fallback_note:
                qa_text = qa_fallback_note

            summary_phish_flag = report.get("summary", {}).get("phishing_flag")
            if summary_phish_flag is True:
                phishing_tag = "Unsafe"
            elif summary_phish_flag is False:
                phishing_tag = "Safe"
            else:
                # if phish data exists per-report, try to infer
                first_phish = report["reports"][0].get("phishing_analysis", {}) or {}
                sb = (first_phish.get("safe_browsing") or {})
                vt = (first_phish.get("virustotal") or {})
                inferred = None
                try:
                    if sb and sb.get("safe") is False:
                        inferred = "Unsafe"
                    elif vt and vt.get("safe") is False:
                        inferred = "Unsafe"
                    elif sb and sb.get("safe") is True and vt and vt.get("safe") is True:
                        inferred = "Safe"
                except Exception:
                    inferred = None
                phishing_tag = inferred or "Unknown"

            # set phishing_tag in the report for frontend convenience
            try:
                report["reports"][0]["phishing_tag"] = phishing_tag
            except Exception:
                pass

        phish = report.get("reports", [{}])[0].get("phishing_analysis", {}) or {}
        # ensure returned phish object includes tag
        try:
            phish = dict(phish)
            phish["phishing_tag"] = report.get("reports", [{}])[0].get("phishing_tag", "Unknown")
        except Exception:
            phish = phish

        return report, qa_text, extracted_claims, phish

    except Exception:
        logger.exception("on_analyze failed")
        return {"error": traceback.format_exc()}, "", [], {}
        
def verdict_to_str(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if not v:
        return "Unclear"
    return str(v).strip()


# Orchestrator

class Orchestrator:
    def run(self, claim_text: str, article_text: Optional[str], url: Optional[str], image_url: Optional[str],
            run_serpapi: bool = True) -> dict:
        claim_text = sanitize_text(claim_text or "")
        article_text = sanitize_text(article_text or "")

        image_analysis = analyze_image_url(image_url) if image_url else None
        phish_report = phishing_checks(url) if url else {}
        if phish_report is None:
            phish_report = {}

        serpapi_result = {"available": False}
        if run_serpapi and SERPAPI_KEY:
            q = claim_text or article_text or url or image_url
            if q:
                serpapi_result = serpapi_web_search(q, num=8)

        # prepare claims
        claims_to_check = []
        if article_text:
            claims_struct = gemini_extract_claims_from_text(article_text, max_claims=3)
            if claims_struct:
                claims_to_check = [c["claim"] for c in claims_struct if c.get("claim")]
            else:
                paras = [p for p in article_text.split("\n") if p.strip()]
                if paras:
                    claims_to_check = [paras[0][:800]]
        else:
            if claim_text:
                claims_to_check = [claim_text]
            elif image_url:
                auto = None
                try:
                    auto = gemini_generate_claim_from_image(image_url)
                except Exception:
                    logger.exception("gemini image claim gen failed")
                if auto:
                    claims_to_check = [auto]
                else:
                    ia = image_analysis or analyze_image_url(image_url)
                    ela = ia.get("ela", {}).get("ela_score") if ia else None
                    phash = ia.get("phash", {}).get("phash") if ia else None
                    serp_note = ""
                    if ia and ia.get("serpapi_reverse", {}).get("available"):
                        serp_note = " Reverse-image search results available."
                    fallback_claim = f"Auto-generated (unverified): Image provided ({image_url}). Content unclear.{(' ELA=' + str(ela)) if ela else ''}{(' phash=' + str(phash)) if phash else ''}{serp_note}"
                    claims_to_check = [fallback_claim]

        unique_claims = []
        seen = set()
        for c in claims_to_check:
            if not c:
                continue
            key = c.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique_claims.append(c)
        verdicts_per_claim = []

        reports = []
        for claim in unique_claims:
            serpapi_for_claim = serpapi_result
            if run_serpapi and SERPAPI_KEY:
                try:
                    serpapi_for_claim = serpapi_web_search(claim, num=6)
                except Exception:
                    serpapi_for_claim = serpapi_result

            hf_result = hf_zero_shot_classify(claim)
            gemini_verdict = gemini_verify_claim(claim, serpapi_for_claim, image_analysis)
            evidence_agg = aggregate_search_results(serpapi_for_claim)
            combined = combine_signals(gemini_verdict, hf_result, evidence_agg)

            try:
                media_flagged_fake = False
                # 1) check gemini_verdict issues (structured)
                g_issues = (gemini_verdict or {}).get("issues", []) or []
                if isinstance(g_issues, (list, tuple)):
                    for it in g_issues:
                        try:
                            if isinstance(it, dict):
                                typ = str(it.get("type", "")).lower()
                                desc = str(it.get("description", "")).lower()
                                if "ai" in typ or "ai_generation" in typ or "ai-generation" in desc or "ai-generated" in desc or "fabricat" in desc or "deepfake" in desc or "generated image" in desc:
                                    media_flagged_fake = True
                                    break
                            elif isinstance(it, str):
                                low = it.lower()
                                if any(k in low for k in ("ai-generated", "fabricat", "fake", "deepfake", "generated image", "computer-generated")):
                                    media_flagged_fake = True
                                    break
                        except Exception:
                            continue

                # 2) check gemini_verdict overall text
                overall_text = str((gemini_verdict or {}).get("overall", "") or "").lower()
                if any(k in overall_text for k in ("ai-generated", "ai generated", "fabricat", "deepfake", "computer-generated", "generated image", "not authentic", "fake")):
                    media_flagged_fake = True

                # 3) check reverse-image search (SERP) snippets/titles for keywords
                if image_analysis and image_analysis.get("serpapi_reverse", {}).get("result"):
                    rlist = image_analysis["serpapi_reverse"]["result"].get("organic_results", []) or []
                    for r in rlist:
                        txt = (str(r.get("title", "") or "") + " " + str(r.get("snippet", "") or "")).lower()
                        if any(k in txt for k in ("ai-generated", "generated image", "computer-generated", "fake image", "deepfake", "ai image", "fabricat")):
                            media_flagged_fake = True
                            break

                # 4)check aggregated evidence titles/snippets
                for e in evidence_agg.get("evidence", [])[:6]:
                    try:
                        txt = (e.get("title", "") or "") + " " + (e.get("snippet", "") or "")
                        low = txt.lower()
                        if any(k in low for k in ("ai-generated", "generated image", "computer-generated", "fake image", "deepfake", "ai image", "fabricat")):
                            media_flagged_fake = True
                            break
                    except Exception:
                        continue

                if media_flagged_fake:
                    combined["final_verdict"] = "False"
                    combined["confidence"] = max(combined.get("confidence", 0.4), 0.6)
                    reasons = combined.get("reasons", []) or []
                    media_reason = {
                        "type": "AI_GENERATION",
                        "description": "Media-authenticity override: image appears to be AI-generated/fabricated per model findings or trusted fact-checks."
                    }
                    try:
                        if not any(isinstance(r, dict) and r.get("type") == "AI_GENERATION" for r in reasons):
                            reasons.insert(0, media_reason)
                    except Exception:
                        reasons.insert(0, media_reason)
                    combined["reasons"] = reasons
            except Exception:
                logger.exception("Media override check failed")
            verdicts_per_claim.append([
                verdict_to_str(gemini_verdict.get("verdict")) if gemini_verdict else "Unclear",
                verdict_to_str(hf_result.get("labels", ["Unclear"])[0]) if hf_result and isinstance(hf_result, dict) and "labels" in hf_result else "Unclear",
                verdict_to_str(combined.get("final_verdict"))
            ])


            trust_pct = _trust_score_pct_from_final_score(combined.get("final_score", 0.0))

            report_entry = {
                "claim": claim,
                "context": article_text[:400] if article_text else "",
                "image_analysis": image_analysis,
                "hf_classifier": hf_result,
                "gemini_verdict": gemini_verdict,
                "serpapi_result": serpapi_for_claim,
                "evidence_agg": evidence_agg,
                "phishing_analysis": phish_report,
                "final_verdict": combined["final_verdict"],
                "confidence": combined["confidence"],
                "reasons": combined.get("reasons", []),
                "final_score": combined.get("final_score"),
                "trust_score_pct": trust_pct
            }

            report_entry["qa_summary"] = format_user_friendly_explanation(report_entry)

            reports.append(report_entry)
        ground_truth_labels = [verdict_to_str(r.get("truth_label")) for r in reports if r.get("truth_label")]

        if ground_truth_labels:
            modal_acc = compute_modal_accuracy(verdicts_per_claim, ground_truth_labels)
        else:
            modal_acc = 0.0
            
        if not ground_truth_labels:
            ground_truth_labels = [verdict_to_str(r["final_verdict"]) for r in reports]


            
        summary = {"counts": {}, "dominant_verdict": "Unclear", "modal_accuracy": modal_acc}

        for r in reports:
            v = r["final_verdict"]
            summary["counts"][v] = summary["counts"].get(v, 0) + 1
        if reports:
            dominant = max(summary["counts"].items(), key=lambda x: x[1])[0]
            summary["dominant_verdict"] = dominant

        if phish_report:
            sb = (phish_report or {}).get("safe_browsing") or {}
            vt = (phish_report or {}).get("virustotal") or {}
            sb_safe = sb.get("safe")
            vt_safe = vt.get("safe")
            summary["phishing_flag"] = True if (sb_safe is False or vt_safe is False) else False
        else:
            summary["phishing_flag"] = False

        return {
            "claims_analyzed": len(reports),
            "reports": reports,
            "summary": summary,
            "url": url,
            "timestamp": time.time()
        }

ORCH = Orchestrator()


# Gradio UI

title = "NewsOrchestra — Gemini multimodal verifier (upgraded)"
description = "Gemini required. Set GEMINI_API_KEY. SerpApi optional. SAFE_BROWSING_KEY/VIRUSTOTAL_KEY optional."

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}\n\n{description}")
    inp = gr.Textbox(lines=6, label="Article text or URL")
    image_url_in = gr.Textbox(lines=1, label="Image URL (optional)")
    run_serp_cb = gr.Checkbox(label="Run SerpApi (requires SERPAPI_KEY)", value=bool(SERPAPI_KEY))
    analyze_btn = gr.Button("Analyze")

    out_json = gr.JSON(label="Full Report (JSON)")
    out_qa = gr.Textbox(label="Q&A Summary (user-friendly)", lines=12)
    out_claims = gr.JSON(label="Extracted Claims")
    out_phish = gr.JSON(label="Phishing Analysis")



def _gradio_on_analyze(text_or_url, image_url, run_serp):
    return on_analyze(text_or_url, image_url, run_serp)

with gr.Blocks(title=title) as demo2:
    gr.Markdown(f"# {title}\n\n{description}")

    with gr.Row():
        inp = gr.Textbox(lines=6, label="Article text or URL")
        image_url_in = gr.Textbox(lines=1, label="Image URL (optional)")
        run_serp_cb = gr.Checkbox(label="Run SerpApi (requires SERPAPI_KEY)", value=bool(SERPAPI_KEY))
        analyze_btn = gr.Button("Analyze")

    with gr.Row():
        out_json = gr.JSON(label="Full Report (JSON)", visible=True)
        out_qa = gr.Textbox(label="Q&A Summary (user-friendly)", lines=12)
        out_claims = gr.JSON(label="Extracted Claims")
        out_phish = gr.JSON(label="Phishing Analysis")

    analyze_btn.click(
        fn=_gradio_on_analyze,
        inputs=[inp, image_url_in, run_serp_cb],
        outputs=[out_json, out_qa, out_claims, out_phish],
        show_progress=True
    )

if __name__ == "__main__":
    demo2.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
