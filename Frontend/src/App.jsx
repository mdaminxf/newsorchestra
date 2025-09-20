import React, { useState, useMemo, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import demoReport from "./output.json";
import { motion } from "framer-motion";
import { Client } from "@gradio/client";
import {
  BookOpen,
  Layers,
  Search,
  ExternalLink,
  Info,
  CheckCircle as CheckCircleIcon,
  Code,
  List,
  FileText,
  ListTree,
  ChevronRight,
  ChevronDown,
  HelpCircle,
  AlertTriangle,
  XCircle,
  CheckCircle2,
  BarChart2,
  Link2,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import LoadingPage from "./Loading";

/* ============================
   Styling tokens & helpers
   ============================ */
const VERDICT_COLORS = {
  True: "bg-green-100 text-green-900 border border-green-300",
  False: "bg-red-100 text-red-900 border border-red-300",
  Misleading: "bg-orange-100 text-orange-900 border border-orange-300",
  Unclear: "bg-gray-100 text-gray-900 border border-gray-300",
  Mixed: "bg-yellow-100 text-yellow-900 border border-yellow-300",
  Unsupported: "bg-gray-100 text-gray-900 border border-gray-300",
};

const BADGE_COLORS = {
  True: "bg-green-600 text-white",
  False: "bg-red-600 text-white",
  Misleading: "bg-orange-600 text-white",
  Unclear: "bg-gray-600 text-white",
  Mixed: "bg-yellow-600 text-black",
  Unsupported: "bg-gray-500 text-white",
};

const DOMAIN_COLORS = [
  "bg-teal-500",
  "bg-green-500",
  "bg-purple-500",
  "bg-pink-500",
  "bg-indigo-500",
  "bg-yellow-500",
  "bg-red-500",
  "bg-teal-500",
];

const Gemini_COLORS = {
  True: "bg-green-50 text-green-900 border-l-4 border-green-500",
  False: "bg-red-50 text-red-900 border-l-4 border-red-500",
  Misleading: "bg-orange-50 text-orange-900 border-l-4 border-orange-500",
  Unclear: "bg-gray-50 text-gray-900 border-l-4 border-gray-500",
  Mixed: "bg-yellow-50 text-yellow-900 border-l-4 border-yellow-500",
  Unsupported: "bg-gray-50 text-gray-900 border-l-4 border-gray-500",
};

function useDevToolsProtection() {
  useEffect(() => {
    // 1) Disable right-click
    const handleContextMenu = (e) => e.preventDefault();
    document.addEventListener("contextmenu", handleContextMenu);

    // 2) Detect F12, Ctrl+Shift+I, Cmd+Option+I
    const handleKeyDown = (e) => {
      if (
        e.key === "F12" ||
        (e.ctrlKey && e.shiftKey && e.key === "I") ||
        (e.metaKey && e.altKey && e.key === "I")
      ) {
        e.preventDefault();
        alert("DevTools is disabled on this page!");
      }
    };
    document.addEventListener("keydown", handleKeyDown);

    // 3) Detect DevTools open by viewport difference
    let devtoolsOpen = false;
    const threshold = 160;
    const detectDevTools = setInterval(() => {
      const widthThreshold = window.outerWidth - window.innerWidth > threshold;
      const heightThreshold = window.outerHeight - window.innerHeight > threshold;
      if (widthThreshold || heightThreshold) {
        if (!devtoolsOpen) {
          devtoolsOpen = true;
          console.log("DevTools detected!");
          // Optional: redirect, hide sensitive info, etc.
        }
      } else {
        devtoolsOpen = false;
      }
    }, 1000);

    // Cleanup
    return () => {
      document.removeEventListener("contextmenu", handleContextMenu);
      document.removeEventListener("keydown", handleKeyDown);
      clearInterval(detectDevTools);
    };
  }, []);
}

const safeHref = (maybeUrl) => {
  try {
    if (!maybeUrl) return null;
    let s = typeof maybeUrl === "string" ? maybeUrl.trim() : String(maybeUrl).trim();
    if (!s) return null;
    // if it already has a scheme, return it
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(s)) return s;
    // protocol-relative
    if (s.startsWith("//")) return `https:${s}`;
    // starts with www -> prepend https
    if (s.startsWith("www.")) return `https://${s}`;
    // simple host-only (example.com) -> prepend https
    if (!s.includes(" ") && s.includes(".")) return `https://${s}`;
    // otherwise, unknown/unsafe
    return null;
  } catch {
    return null;
  }
};

const getDomainFromUrl = (u) => {
  if (!u) return "";
  if (typeof u !== "string") u = String(u);
  try {
    return new URL(u).hostname.replace(/^www\./, "");
  } catch {
    // fallback to simple parse
    try {
      return u.split("/")[0];
    } catch {
      return u;
    }
  }
};

const avatarColorClass = (key) => {
  if (!key) return DOMAIN_COLORS[0];
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return DOMAIN_COLORS[h % DOMAIN_COLORS.length];
};
const avatarLetter = (domainOrName) => {
  if (!domainOrName) return "?";
  const s = domainOrName.replace(/^www\./, "").trim();
  return s[0] ? s[0].toUpperCase() : "?";
};

/* ============================
   Normalization helpers
   ============================ */
const getAllReports = (report) => {
  if (!report) return [];
  if (Array.isArray(report.reports)) return report.reports;
  if (Array.isArray(report.claims)) return report.claims;
  if (Array.isArray(report.data)) return report.data;
  return [];
};

const ConfidenceTrustBars = ({ confidence = 0, trust = 0 }) => {
  return (
    <div className="flex flex-col gap-1 w-full">
      <div className="text-xs text-gray-600">Confidence: {(confidence * 100).toFixed(1)}%</div>
      <div className="w-full bg-gray-200 h-2 rounded">
        <div className="bg-green-500 h-2 rounded" style={{ width: `${confidence * 100}%` }} />
      </div>
      <div className="text-xs text-gray-600">Trust: {(trust * 100).toFixed(1)}%</div>
      <div className="w-full bg-gray-200 h-2 rounded">
        <div className="bg-teal-500 h-2 rounded" style={{ width: `${trust * 100}%` }} />
      </div>
    </div>
  );
};

const combineAndDedupSources = (rpt) => {
  const list = [];
  const pushIf = (obj) => {
    if (!obj) return;
    const url = obj.url || obj.link || obj.source_url || null;
    const title = obj.source || obj.title || obj.headline || obj.domain || "";
    const snippet = obj.snippet || obj.summary || obj.text || "";
    list.push({ url, title, snippet });
  };

  const gCits = (rpt?.gemini_verdict?.citations) || [];
  gCits.forEach(pushIf);
  const webE = (rpt?.evidence_agg?.evidence) || (rpt?.sources) || [];
  webE.forEach(pushIf);

  const seen = new Map();
  list.forEach((s) => {
    const key = s.url || s.title;
    if (!key) return;
    if (!seen.has(key)) seen.set(key, s);
  });
  return Array.from(seen.values());
};

/* Confidence / Trust normalization helpers */
const normalizeConfidence = (v) => {
  if (v == null) return null;
  const n = Number(v);
  if (Number.isNaN(n)) return null;
  if (n >= 0 && n <= 1) return n; // already 0..1
  if (n > 1 && n <= 100) return n / 100; // 0..100 -> 0..1
  return Math.max(0, Math.min(1, n));
};

const normalizeTrustPct = (v) => {
  if (v == null) return null;
  const n = Number(v);
  if (Number.isNaN(n)) return null;
  if (n >= 0 && n <= 1) return Math.round(n * 100);
  if (n >= 0 && n <= 100) return Math.round(n);
  return Math.round(Math.max(0, Math.min(100, n)));
};

const fmtPct = (v, decimals = 0) => {
  if (v == null) return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return `${n.toFixed(decimals)}%`;
};

/* ============================
   Complexity helpers
   ============================ */
const computeComplexityData = (nValues) => {
  return nValues.map((n) => {
    const logn = Math.log2(Math.max(2, n));
    return {
      n,
      O1: 1,
      Ologn: logn,
      On: n,
      Onlogn: n * logn,
      On2: n * n,
      O2n: Math.min(1e6, Math.pow(2, Math.min(20, Math.floor(Math.log2(Math.max(1, n)))))),
    };
  });
};

function normalizedWithMap(s) {
  const lower = String(s || "").toLowerCase();
  let norm = "";
  const map = [];
  for (let i = 0; i < lower.length; i++) {
    const ch = lower[i];
    if (/\w|[^\x00-\x7F]/.test(ch)) {
      // keep letters/digits/underscore and non-ascii (to support unicode)
      norm += ch;
      map.push(i);
    } else {
      // collapse punctuation to single space (avoid duplicate spaces)
      if (norm.length && norm[norm.length - 1] !== " ") {
        norm += " ";
        map.push(i);
      }
    }
  }
  // trim leading/trailing spaces and adjust map
  while (norm.length && norm[0] === " ") {
    norm = norm.slice(1);
    map.shift();
  }
  while (norm.length && norm[norm.length - 1] === " ") {
    norm = norm.slice(0, -1);
    map.pop();
  }
  return { norm, map };
}

function findPhraseInArticle(originalArticle, phrase) {
  if (!originalArticle || !phrase) return -1;
  const art = String(originalArticle);
  const artLower = art.toLowerCase();
  const p = String(phrase).trim();
  if (!p) return -1;

  // 1) direct case-insensitive search
  let idx = artLower.indexOf(p.toLowerCase());
  if (idx >= 0) return idx;

  // 2) variant replacing hyphen with space
  const pNoHyphen = p.replace(/-/g, " ");
  if (pNoHyphen !== p) {
    idx = artLower.indexOf(pNoHyphen.toLowerCase());
    if (idx >= 0) return idx;
  }

  // 3) normalized search (collapse punctuation/whitespace)
  const artNorm = normalizedWithMap(art);
  const phraseNorm = normalizedWithMap(p);
  if (!phraseNorm.norm) return -1;
  const posNorm = artNorm.norm.indexOf(phraseNorm.norm);
  if (posNorm >= 0) {
    return artNorm.map[posNorm];
  }

  return -1;
}

function extractIssuePhraseMatchesFromReport(reportItem, articleText, minWords = 2, maxWords = 12) {
  if (!reportItem) return [];
  const issues = reportItem.gemini_verdict?.issues || reportItem.issues || [];
  if (!issues || !issues.length) return [];

  const claim = String(reportItem.claim || reportItem.text || "").trim();
  if (!claim) return [];

  const article = String(articleText || "");
  const claimWords = claim.split(/\s+/).filter(Boolean);
  const maxLen = Math.min(maxWords, claimWords.length);
  const found = new Map(); // map phrase@pos -> { phrase, pos, end }

  // Normalize issue values for "contains" checks
  const issuesValues = issues.map((it) => {
    const v = typeof it === "string" ? it : it?.value ?? String(it ?? "");
    return String(v || "");
  });
  const normIssueValues = issuesValues.map((v) => normalizedWithMap(v).norm);

  // iterate from longest contiguous phrase to shortest to prefer longer matches
  for (let len = maxLen; len >= minWords; len--) {
    for (let start = 0; start <= claimWords.length - len; start++) {
      const phrase = claimWords.slice(start, start + len).join(" ");
      const normPhrase = normalizedWithMap(phrase).norm;
      if (!normPhrase) continue;

      // check if phrase is present in any issue.value (normalized)
      let presentInIssue = false;
      for (let vNorm of normIssueValues) {
        if (!vNorm) continue;
        if (vNorm.includes(normPhrase)) {
          presentInIssue = true;
          break;
        }
      }
      if (!presentInIssue) continue;

      // phrase is mentioned in an issue -> ensure it appears in the article
      const pos = findPhraseInArticle(article, phrase);
      if (pos >= 0) {
        const key = `${phrase}||${pos}`;
        if (!found.has(key)) {
          found.set(key, { phrase, pos, end: pos + phrase.length });
        }
      } else {
        // try hyphen variant
        const alt = phrase.replace(/-/g, " ");
        if (alt !== phrase) {
          const posAlt = findPhraseInArticle(article, alt);
          if (posAlt >= 0) {
            const key = `${alt}||${posAlt}`;
            if (!found.has(key)) {
              found.set(key, { phrase: alt, pos: posAlt, end: posAlt + alt.length });
            }
          }
        }
      }
    }
  }

  return Array.from(found.values());
}

function FaviconAvatar({ url, title, size = 40 }) {
  const domain = url ? getDomainFromUrl(url) : (title || "").split(" ")[0];
  const faviconUrl = domain ? `https://www.google.com/s2/favicons?domain=${domain}&sz=64` : null;
  const [src, setSrc] = useState(null);
  const [failed, setFailed] = useState(false);
  const imgRef = useRef(null);
  const wrapperRef = useRef(null);

  useEffect(() => {
    if (!faviconUrl) return;
    let obs;
    let mounted = true;
    const el = wrapperRef.current;
    if (!el) {
      setSrc(faviconUrl);
      return;
    }

    if ("IntersectionObserver" in window) {
      obs = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting && mounted) {
              setSrc(faviconUrl);
              obs.disconnect();
            }
          });
        },
        { root: null, rootMargin: "200px", threshold: 0.01 }
      );
      obs.observe(el);
    } else {
      setSrc(faviconUrl);
    }

    return () => {
      mounted = false;
      if (obs) obs.disconnect();
    };
  }, [faviconUrl]);

  const color = avatarColorClass(domain || title || "");
  const letter = avatarLetter(domain || title || "");

  return (
    <div ref={wrapperRef} style={{ width: size, height: size }} className="rounded-full overflow-hidden flex-shrink-0">
      {src && !failed ? (
        <img
          ref={imgRef}
          src={src}
          alt={domain || title || "site"}
          loading="lazy"
          onError={() => setFailed(true)}
          className="w-full h-full object-cover"
        />
      ) : (
        <div className={`w-full h-full flex items-center justify-center text-white font-semibold ${color}`}>
          <span>{letter}</span>
        </div>
      )}
    </div>
  );
}

function ExternalLinkIcon() {
  return (
    <svg className="w-3 h-3 inline-block" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H21v7.5M21 3L10 14" />
    </svg>
  );
}

// Enhanced Accordion with better styling
const Accordion = ({ items }) => {
  const [openIndex, setOpenIndex] = useState(null);

  return (
    <div className="divide-y divide-gray-200 border border-gray-200 rounded-lg overflow-hidden">
      {items.map((item, idx) => (
        <div key={idx} className="bg-white">
          <button
            onClick={() => setOpenIndex(openIndex === idx ? null : idx)}
            className="w-full text-left px-4 py-3 font-medium text-gray-900 bg-gray-50 hover:bg-gray-100 flex items-center justify-between"
          >
            <span className="font-semibold">{item.title}</span>
            {openIndex === idx ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>
          {openIndex === idx && (
            <div className="px-4 py-3 bg-white border-t border-gray-100">
              {typeof item.content === "string" ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                >
                  {item.content}
                </ReactMarkdown>
              ) : (
                item.content
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

/* ============================
   QA parsing & rendering
   - Accepts many backend shapes:
     * string containing "Q1: ... A: ..." blocks
     * array [{question, answer}]
     * object with keys like { "Q1": "...", "Q2": "..."} or { question:..., answer:... }
     * array of strings (assumed [ "Q1...A..." ]).
   Returns normalized array: [{ question, answer }]
   ============================ */
// ------------------ QA parsing & rendering (replace old QA code) ------------------

const DESIRED_QUESTIONS = [
  "Why did we reach this verdict?",
  "How was it verified?",
  "What should you do next?",
];

// normalize single QA-like object into {question, answer} or array of them
function normalizeQAObject(obj) {
  if (!obj) return null;
  if (typeof obj === "string") {
    const parsed = parseQABlockString(obj);
    if (parsed.length) return parsed[0];
    return { question: "", answer: obj };
  }
  if (Array.isArray(obj)) return null; // handled upstream
  // common keys
  const q = obj.question || obj.q || obj.prompt || obj.title || obj.key || "";
  const a = obj.answer || obj.a || obj.response || obj.text || obj.value || obj.content || "";
  if (q || a) return { question: String(q || "").trim(), answer: String(a || "").trim() };

  // maybe shape like { Q1: "...", Q2: "..." }
  const keys = Object.keys(obj || {});
  const qKeys = keys.filter((k) => /^Q\d/i.test(k));
  if (qKeys.length) {
    const res = [];
    qKeys.sort().forEach((k) => {
      const block = String(obj[k] || "").trim();
      const m = block.match(/^(?:Q\d:)?\s*(.*?)\nA:\s*([\s\S]*)$/i);
      if (m) res.push({ question: m[1].trim(), answer: m[2].trim() });
      else res.push({ question: k, answer: block });
    });
    return res.length === 1 ? res[0] : res;
  }
  // fallback: stringify
  return { question: "", answer: JSON.stringify(obj) };
}

function parseQABlockString(qaSummaryString) {
  if (!qaSummaryString || typeof qaSummaryString !== "string") return [];
  // capture Q blocks (multi-line answers allowed)
  const regex = /(Q\d:\s*.+?\nA:\s*[\s\S]*?)(?=(?:\nQ\d:)|$)/gi;
  const matches = qaSummaryString.match(regex);
  if (matches && matches.length) {
    return matches.map((block) => {
      const lines = block.trim().split("\n");
      const questionLine = lines.shift() || "";
      const question = questionLine.replace(/^Q\d:\s*/i, "").trim();
      const answerText = lines.join("\n").replace(/^A:\s*/i, "").trim();
      return { question, answer: answerText };
    });
  }

  // fallback simpler pattern
  const regex2 = /(Q\d:\s.*?\nA:.*?)(?=Q\d:|$)/gs;
  const m2 = qaSummaryString.match(regex2);
  if (m2 && m2.length) {
    return m2.map((block) => {
      const [questionLine, ...answerLines] = block.trim().split("\n");
      const question = questionLine.replace(/^Q\d:\s*/i, "").trim();
      const answer = answerLines.join("\n").replace(/^A:\s*/i, "").trim();
      return { question, answer };
    });
  }

  // last resort: split by double newline into chunks
  const naive = qaSummaryString.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
  if (naive.length) {
    // try to heuristically map the first three chunks to desired questions
    const out = [];
    for (let i = 0; i < Math.min(3, naive.length); i++) {
      const q = DESIRED_QUESTIONS[i] || `Q${i + 1}`;
      out.push({ question: q, answer: naive[i] });
    }
    return out;
  }

  return [];
}

const parseQASummary = (qaSummaryInput) => {
  if (!qaSummaryInput) return [];

  // Already an array
  if (Array.isArray(qaSummaryInput)) {
    const normalized = qaSummaryInput
      .map((it) => {
        if (!it) return null;
        if (typeof it === "string") {
          const parsed = parseQABlockString(it);
          if (parsed.length) return parsed;
          return { question: "", answer: it };
        }
        const n = normalizeQAObject(it);
        return Array.isArray(n) ? n : n;
      })
      .flat()
      .filter(Boolean)
      .map((it) => ({ question: String(it.question || "").trim(), answer: String(it.answer || "").trim() }));
    return normalized;
  }

  // Object
  if (typeof qaSummaryInput === "object") {
    const n = normalizeQAObject(qaSummaryInput);
    if (Array.isArray(n)) return n.map((it) => ({ question: it.question, answer: it.answer }));
    if (n) return [{ question: n.question || "", answer: n.answer || "" }];
    return [];
  }

  // String
  if (typeof qaSummaryInput === "string") {
    const parsed = parseQABlockString(qaSummaryInput);
    if (parsed.length) return parsed;
    // fallback: single blob
    return [{ question: "Summary", answer: qaSummaryInput }];
  }

  return [];
};

// ---- helpers for formatting answers ----
const jsonToBullets = (str) => {
  try {
    const obj = JSON.parse(str);
    return Object.entries(obj)
      .map(([k, v]) => `- **${k}**: ${v}`)
      .join("\n");
  } catch {
    return str;
  }
};

const prettifyAnswer = (answer) => {
  if (!answer) return "";

  let formatted = String(answer);

  // strip common Q/A tags
  formatted = formatted.replace(/^Q\d:\s*/gm, "");
  formatted = formatted.replace(/^A:\s*/gm, "");

  // convert inline JSON objects (single-level) to bullets
  const jsonRegex = /{[^{}]+}/g;
  formatted = formatted.replace(jsonRegex, (match) => jsonToBullets(match));

  // semicolons -> bullet list (common in reasons)
  if (formatted.includes(";")) {
    formatted =
      "- " +
      formatted
        .split(";")
        .map((s) => s.trim())
        .filter(Boolean)
        .join("\n- ");
  }

  // If lines with "- " already present, keep as is
  return formatted.trim();
};

// small heat indicator color mapper
const getHeatColor = (answer) => {
  if (!answer) return "bg-gray-300";
  const lower = String(answer).toLowerCase();
  if (lower.includes("ai-generated") || lower.includes("fabricat") || lower.includes("false")) return "bg-red-500";
  if (lower.includes("insufficient") || lower.includes("uncertain") || lower.includes("unclar")) return "bg-yellow-400";
  if (lower.includes("true") || lower.includes("verified") || lower.includes("corroborate")) return "bg-green-500";
  return "bg-gray-300";
};

// ---- QA item: collapsible single question block ----
const QAItem = ({ question, answer, defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  const heatColor = getHeatColor(answer);
  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((s) => !s)}
        className="w-full px-4 py-3 bg-gray-50 hover:bg-gray-100 flex items-center justify-between gap-3 text-left"
      >
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${heatColor}`} />
          <span className="font-medium text-gray-900">{question || "Q"}</span>
        </div>
        <div className="text-sm text-gray-500">{open ? "Hide" : "Show"}</div>
      </button>

      {open && (
        <div className="px-4 py-3 bg-white text-sm text-gray-700 prose prose-sm max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
            {prettifyAnswer(answer) || "—"}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
};

// ---- QA Section: renders up to 3 desired questions as separate collapsibles ----
const QASection = ({ qaSummary, defaultOpenIndex = 0 }) => {
  // Accept either a parsed array, or a raw string/object from backend
  const parsed = useMemo(() => parseQASummary(qaSummary), [qaSummary]);

  if (!parsed || parsed.length === 0) return null;

  // Build map for easy lookup by question text
  const qaMap = new Map();
  parsed.forEach((qa) => {
    const q = (qa.question || "").trim();
    const a = (qa.answer || "").trim();
    if (q || a) qaMap.set(q, a);
  });

  const finalList = [];

  // push desired questions in order if present
  DESIRED_QUESTIONS.forEach((q) => {
    if (qaMap.has(q)) {
      finalList.push({ question: q, answer: qaMap.get(q) });
      qaMap.delete(q);
    }
  });

  // fill with remaining parsed QAs until 3 total
  if (finalList.length < 3) {
    for (const qa of parsed) {
      if (finalList.length >= 3) break;
      if (DESIRED_QUESTIONS.includes(qa.question)) continue;
      finalList.push({ question: qa.question || "Summary", answer: qa.answer || "" });
    }
  }

  // ensure unique and trim to 3
  const seen = new Set();
  const unique = [];
  for (const it of finalList) {
    const k = (it.question || "").trim();
    if (seen.has(k)) continue;
    seen.add(k);
    unique.push(it);
    if (unique.length >= 3) break;
  }

  if (!unique.length) return null;

  return (
    <div className="bg-white rounded-xl shadow-md p-4 space-y-3">
      <h4 className="font-semibold text-gray-900 mb-1">Q&A Summary</h4>
      <div className="space-y-2">
        {unique.map((qa, idx) => (
          <QAItem key={idx} question={qa.question} answer={qa.answer} defaultOpen={idx === defaultOpenIndex} />
        ))}
      </div>
    </div>
  );
};


/* ============================
   Main component
   ============================ */
export default function NewsOrchestraUI() {
  const [textOrUrl, setTextOrUrl] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [mode, setMode] = useState("gradio");
  const [loading, setLoading] = useState(false);
  const [pageLoading, setPageLoading] = useState(true);
  const [report, setReport] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [selectedClaimIndex, setSelectedClaimIndex] = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const [qaSummary, setQaSummary] = useState(null);
  const [extractedClaims, setExtractedClaims] = useState(null);
  const [phishingAnalysis, setPhishingAnalysis] = useState(null);
  const [nRange, setNRange] = useState({ start: 10, end: 1000, step: 50 });
  const allReports = useMemo(() => getAllReports(report), [report]);

 
  const nValues = useMemo(() => {
    const arr = [];
    for (let i = nRange.start; i <= nRange.end; i += nRange.step) arr.push(i);
    return arr;
  }, [nRange]);

  const complexityData = useMemo(() => computeComplexityData(nValues), [nValues]);

  const analyze = async () => {
    setLoading(true);
    setReport(null);
    setQaSummary(null);
    setExtractedClaims(null);
    setPhishingAnalysis(null);
    setErrorMsg(null);
    setSelectedClaimIndex(null);

    try {
      if (mode === "demo") {
        setReport(demoReport);
        const first = (demoReport.reports && demoReport.reports[0]) || demoReport;
        setQaSummary(first?.qa_summary ?? first?.qaSummary ?? demoReport.qa_summary ?? null);
        setExtractedClaims(first?.extracted_claims ?? first?.extractedClaims ?? null);
        setPhishingAnalysis(first?.phishing_analysis ?? first?.phishingAnalysis ?? null);
        setLoading(false);
        return;
      }

      if ((!textOrUrl || !textOrUrl.trim()) && (!imageUrl || !imageUrl.trim())) {
        setErrorMsg("Please provide text or an image URL");
        setLoading(false);
        return;
      }

      const client = await Client.connect("ANISA09/atlas");
      const payload = {
        text_or_url: textOrUrl?.trim() || "",
        image_url: imageUrl?.trim() || "",
        run_serp: true,
      };

      const result = await client.predict("/_gradio_on_analyze", payload);
      const full = result?.data?.[0] ?? null;
      const qas = result?.data?.[1] ?? null;
      const extracted = result?.data?.[2] ?? null;
      const phishing = result?.data?.[3] ?? null;

      setReport(full);
      setQaSummary(qas);
      setExtractedClaims(extracted);
      setPhishingAnalysis(phishing);
    } catch (err) {
      console.error(err);
      setErrorMsg(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  };


  const modelAccuracies = useMemo(() => {
    if (!allReports.length) return [];
    const modelMap = new Map();
    allReports.forEach((r) => {
      const model = r.model_name || r.model || "";
      if (!modelMap.has(model)) modelMap.set(model, { correct: 0, total: 0 });
      const entry = modelMap.get(model);
      const predicted = (r.final_verdict || r.verdict || r.gemini_verdict?.verdict || "").toString();
      const expected = (r.expected_label || r.ground_truth || r.true_label || "").toString();
      if (expected) {
        entry.total += 1;
        if (predicted && predicted.toLowerCase() === expected.toLowerCase()) entry.correct += 1;
      }
    });
    const arr = [];
    modelMap.forEach((v, k) => {
      arr.push({ model: k, correct: v.correct, total: v.total, accuracy: v.total ? +(v.correct / v.total * 100).toFixed(1) : null });
    });
    return arr;
  }, [allReports]);

  const confidenceTrustAggregates = useMemo(() => {
    if (!allReports.length) return { avgConfidencePct: null, avgTrustPct: null, count: 0 };
    let confSum = 0;
    let confCount = 0;
    let trustSum = 0;
    let trustCount = 0;
    allReports.forEach((r) => {
      const c = normalizeConfidence(r.confidence ?? r.confidence_score ?? r.confidence_pct ?? r.confidence_pct_normalized);
      if (c != null) { confSum += c; confCount += 1; }
      const t = normalizeTrustPct(r.trust_score_pct ?? r.trust_pct ?? r.trust_score ?? r.trust);
      if (t != null) { trustSum += t; trustCount += 1; }
    });
    const avgConfidencePct = confCount ? Math.round((confSum / confCount) * 100) : null;
    const avgTrustPct = trustCount ? Math.round(trustSum / trustCount) : null;
    return { avgConfidencePct, avgTrustPct, count: allReports.length, confCount, trustCount };
  }, [allReports]);

const finalVerdictSummary = useMemo(() => {
  if (!allReports.length) return null;

  // Collect all claim summaries
  const claimSummaries = allReports.map((r, idx) => {
    const label =
      r?.final_verdict ||
      r?.verdict ||
      r?.gemini_verdict?.verdict ||
      "Unclear";
    const overall =
      r?.gemini_verdict?.overall || r?.summary || r?.qa_summary || "";
    return {
      idx,
      claim: r.claim || r.text || `Claim ${idx + 1}`,
      label,
      overall,
    };
  });

  // Merge into one summary
  const mergedClaim = claimSummaries.map((c) => c.claim).join(" | ");
  const mergedOverall = claimSummaries
    .map((c) => c.overall)
    .filter(Boolean)
    .join(" ");

  // Pick dominant label (most frequent)
  const labelCounts = {};
  for (const c of claimSummaries) {
    labelCounts[c.label] = (labelCounts[c.label] || 0) + 1;
  }
  const dominantLabel = Object.entries(labelCounts).sort(
    (a, b) => b[1] - a[1]
  )[0][0];

  return {
    claim: mergedClaim,
    label: dominantLabel,
    overall: mergedOverall,
  };
}, [allReports]);


  // useDevToolsProtection();
  const renderArticleWithHighlights = () => {
    const text =
      report?.input_text ||
      report?.article ||
      report?.text ||
      allReports.map((r) => r.claim || r.text || "").join(" ") ||
      "";
    if (!text) return <p className="text-gray-500">No article text available.</p>;

    const rawMatches = [];
    allReports.forEach((rpt, idx) => {
      const matches = extractIssuePhraseMatchesFromReport(rpt, text, 2, 12);
      matches.forEach((m) =>
        rawMatches.push({
          ...m,
          reportIndex: idx,
          verdict: rpt.final_verdict || rpt.verdict || rpt.gemini_verdict?.verdict || "Unclear",
        })
      );
    });

    if (rawMatches.length === 0) {
      return (
        <>
          <div className="mt-3 flex flex-wrap gap-2">
            {allReports.map((r, i) => (
              <button
                key={`chip-${i}`}
                onClick={() => setSelectedClaimIndex(i)}
                className={`text-sm px-3 py-1 rounded cursor-pointer ${VERDICT_COLORS[r.final_verdict || r.verdict || "Unclear"] || "bg-yellow-200 text-yellow-900"}`}
              >
                {r.claim || r.text || `Claim ${i + 1}`}
              </button>
            ))}
          </div>
        </>
      );
    }

    rawMatches.sort((a, b) => {
      const la = a.end - a.pos;
      const lb = b.end - b.pos;
      if (lb !== la) return lb - la;
      if (a.pos !== b.pos) return a.pos - b.pos;
      return a.reportIndex - b.reportIndex;
    });

    const accepted = [];
    const occupied = [];
    const overlaps = (s, e) => occupied.some(([os, oe]) => !(e <= os || s >= oe));

    rawMatches.forEach((m) => {
      if (!overlaps(m.pos, m.end)) {
        accepted.push(m);
        occupied.push([m.pos, m.end]);
      }
    });

    accepted.sort((a, b) => a.pos - b.pos);

    const fragments = [];
    let cursor = 0;
    accepted.forEach((m, i) => {
      const start = m.pos;
      const end = m.end;
      if (start > cursor) {
        fragments.push(<span key={`t-${cursor}`}>{text.slice(cursor, start)}</span>);
      }
      const originalSlice = text.slice(start, end);
      fragments.push(
        <button
          key={`c-${m.reportIndex}-${start}-${i}`}
          onClick={() => setSelectedClaimIndex(m.reportIndex)}
          className={`rounded px-1 py-[2px] mr-0.5 transition inline-block text-sm ${VERDICT_COLORS[m.verdict] || "bg-yellow-200 text-yellow-900"} ${selectedClaimIndex === m.reportIndex ? "ring-2 ring-teal-500" : ""}`}
        >
          {originalSlice}
        </button>
      );
      cursor = end;
    });
    if (cursor < text.length) fragments.push(<span key={`t-last`}>{text.slice(cursor)}</span>);

    return (
      <>
        <p className="whitespace-pre-wrap leading-relaxed text-gray-800">{fragments.length ? fragments : text}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {accepted.map((m, i) => (
            <button
              key={`chip-${i}`}
              onClick={() => setSelectedClaimIndex(m.reportIndex)}
              className={`text-sm px-3 py-1 rounded ${VERDICT_COLORS[m.verdict] || "bg-yellow-200 text-yellow-900"}`}
            >
              {m.phrase}
            </button>
          ))}
        </div>
      </>
    );
  };
 useEffect(()=>{
    const timer = setTimeout(()=> setPageLoading(false),3000);
    return ()=> clearTimeout(timer);
  },[])

  if(pageLoading)
  {
    return <LoadingPage/>;
  }
  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 to-indigo-50 py-8 px-4 text-gray-900">
      <div className="max-w-7xl mx-auto">
        {/* header */}
        <motion.div  initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }} className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
              <span className="text-5xl text-white"></span>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">NewsOrchestra</h1>
              <p className="text-sm text-gray-600 mt-1">Fact-checking and analysis platform powered by AI</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex rounded-lg bg-white shadow p-1 border border-gray-200">
              <button onClick={() => {setMode("gradio");setTextOrUrl("");
                    setImageUrl("");
                    setReport(null);
                    setQaSummary(null);
                    setExtractedClaims(null);
                    setPhishingAnalysis(null);
                    setErrorMsg(null);
                    setSelectedClaimIndex(null);}} className={`px-4 py-1 rounded-lg ${mode === "gradio" ? "bg-teal-600 text-white" : "text-gray-600"}`}>Live</button>
              <button onClick={() => setMode("demo")} className={`px-4 py-1 rounded-lg ${mode === "demo" ? "bg-zinc-600 text-white" : "text-gray-600"}`}>Demo</button>
            </div>
          </div>
        </motion.div>

        {/* inputs */}
         <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay:0.2 }}
      >
        <div className="bg-white rounded-xl shadow-md p-6 mb-6 border border-gray-200">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            <textarea
              value={
                mode !== "demo"
                  ? textOrUrl
                  : "Apple launched new features across its devices to help parents protect children and teens online."
              }
              onChange={(e) => setTextOrUrl(e.target.value)}
              placeholder="Paste article text or URL..."
              rows={4}
              className="md:col-span-8 w-full p-4 border border-gray-200 rounded-lg resize-none 
                         focus:border-zinc-800 focus:ring-2 focus:ring-zinc-200 focus:outline-none
                         placeholder-gray-400 text-gray-800 shadow-sm transition"
            />

            <div className="md:col-span-4 flex flex-col gap-4">
              <input
                type="text"
                value={imageUrl}
                disabled={mode === "demo" || loading}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="Optional image URL"
                className={`p-3 border border-gray-200 rounded-lg shadow-sm 
                            focus:border-zinc-500 focus:zinc-2 focus:ring-zinc-200 focus:outline-none
                            placeholder-gray-400 text-gray-800 transition 
                            ${mode === "demo" ? "cursor-not-allowed bg-gray-100 text-gray-400" : ""}`}
                style={{ height: 56 }}
              />

              <motion.dev   
              initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1,delay:0.5 }}className="flex gap-3">
                <button
                  onClick={analyze}
                  disabled={loading}
                  className={`flex-1 ${mode !== "demo" ? `bg-gradient-to-r from-teal-600 to-teal-500 
                             text-white px-6 py-3 rounded-lg shadow-md 
                             hover:from-teal-700 hover:to-teal-600
                             focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2
                             `:`bg-gradient-to-r from-zinc-600 to-zinc-500 
                             text-white px-6 py-3 rounded-lg shadow-md 
                             hover:from-zinc-700 hover:to-zinc-600
                             focus:outline-none focus:ring-2 focus:ring-zinc-500 focus:ring-offset-2`}
                             disabled:opacity-60 transition font-medium inline-flex items-center justify-center gap-2`}
                >
                  {loading ? (
                    "Analyzing…"
                  ) : (
                    <>
                      <Search className="w-5 h-5" />
                      Analyze
                    </>
                  )}
                </button>

                <button
                  onClick={() => {
                    setTextOrUrl("");
                    setImageUrl("");
                    setReport(null);
                    setQaSummary(null);
                    setExtractedClaims(null);
                    setPhishingAnalysis(null);
                    setErrorMsg(null);
                    setSelectedClaimIndex(null);
                  }}
                  className="px-5 py-3 rounded-lg bg-gray-100 text-gray-700 
                             hover:bg-gray-200 shadow-sm transition font-medium focus:outline-none focus:ring-2 focus:ring-gray-500"
                >
                  Clear
                </button>
              </motion.dev>
            </div>
          </div>
        </div>
      </motion.section>

        {/* error */}
        {errorMsg && (
          <div className="mb-6 rounded-lg p-4 bg-red-50 border border-red-200 text-red-700 flex items-center gap-2">
            <AlertTriangle size={18} />
            {errorMsg}
          </div>
        )}
 <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1,delay:0.5 }}
      >
        {/* summary and analytics */}
        {report && finalVerdictSummary && (
          <div className="bg-white rounded-xl shadow-md p-6 mb-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Summary</h3>
                <div className="space-y-4">
                 {finalVerdictSummary && (
  <div className={`p-4 rounded ${ VERDICT_COLORS[finalVerdictSummary.label] || "bg-gray-300 text-white"}`}>
    <div className="font-semibold text-gray-800">
      {finalVerdictSummary.claim}
    </div>
    <div className="mt-2 flex items-center gap-2">
      <span
        className={`px-3 py-1 rounded-full text-sm font-medium ${
          BADGE_COLORS[finalVerdictSummary.label] || "bg-gray-600 text-white"
        }`}
      >
        {finalVerdictSummary.label}
      </span>
      <span className="text-sm text-gray-600">
        {confidenceTrustAggregates.avgConfidencePct != null
          ? `${confidenceTrustAggregates.avgConfidencePct}% confidence`
          : ""}
      </span>
    </div>
    {finalVerdictSummary.overall && (
      <div className="mt-3 text-sm text-gray-700 bg-white p-3 rounded-lg border border-gray-200">
        {finalVerdictSummary.overall}
      </div>
    )}
  </div>
)}

                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-semibold text-gray-700 mb-3">Performance Metrics</h4>
                <div className="space-y-4">
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Average Confidence</div>
                    <div className="text-2xl font-bold text-teal-600">
                      {confidenceTrustAggregates.avgConfidencePct != null ? `${confidenceTrustAggregates.avgConfidencePct}%` : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Average Trust Score</div>
                    <div className="text-2xl font-bold text-green-600">
                      {confidenceTrustAggregates.avgTrustPct != null ? `${confidenceTrustAggregates.avgTrustPct}%` : "—"}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 mb-1">Claims Analyzed</div>
                    <div className="text-2xl font-bold text-purple-600">{allReports.length}</div>
                  </div>
                </div>

                {modelAccuracies.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-sm font-semibold text-gray-700 mb-2">Model Accuracy</h4>
                    <div className="space-y-2">
                      {modelAccuracies.map((m) => (
                        <div key={m.model} className="flex justify-between items-center text-sm">
                          <span className="text-gray-600">{m.model}</span>
                          <span className="font-medium text-gray-900">
                            {m.total ? `${m.accuracy}% (${m.correct}/${m.total})` : "No labels"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </motion.section>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* left: article */}
           <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay:0.6}}
      >
          <div className="bg-white rounded-xl shadow-md p-5 overflow-auto max-h-[100vh]">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-red-600" />
                <h4 className="font-semibold text-gray-900">Article & Highlights</h4>
              </div>
              <div className="text-sm text-gray-500">Click highlights to view analysis</div>
            </div>

            <div className="prose max-w-none border border-gray-200 rounded-lg p-4 bg-gray-50">
              {renderArticleWithHighlights()}
            </div>

            <div className="mt-6">
              <h5 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
                <List className="w-4 h-4" />
                Claims Found
              </h5>
              <div className="space-y-3">
                {allReports.length === 0 && <p className="text-sm text-gray-500">No claims found.</p>}
                {allReports.map((r, i) => {
                  const label = r?.final_verdict || r?.verdict || (r?.gemini_verdict?.verdict) || "Unclear";
                  const cNorm = normalizeConfidence(r.confidence ?? r.confidence_score ?? r.confidence_pct ?? null);
                  const tPct = normalizeTrustPct(r.trust_score_pct ?? r.trust_pct ?? r.trust_score ?? r.trust ?? null);
                  return (
                    <div
                      key={i}
                      onClick={() => setSelectedClaimIndex(i)}
                      className={`cursor-pointer p-4 rounded-lg border hover:shadow-md transition-all flex justify-between items-start gap-4 ${selectedClaimIndex === i ? "ring-2 ring-teal-500 border-teal-300 bg-teal-50" : "border-gray-200 bg-white"}`}
                    >
                      <div className="flex-1">
                        <div className="text-sm font-medium text-gray-800">{r.claim || r.text || `Claim ${i + 1}`}</div>
                        <div className="text-xs text-gray-500 mt-1 line-clamp-2">{(r.context || r.title || "").slice(0, 140)}</div>
                        <div className="mt-3 flex gap-4 text-xs text-gray-600">
                          <div>Confidence: <span className="font-medium">{cNorm != null ? `${Math.round(cNorm * 100)}%` : "—"}</span></div>
                          <div>Trust: <span className="font-medium">{tPct != null ? `${tPct}%` : "—"}</span></div>
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <span className={`text-xs px-2 py-1 rounded-full ${BADGE_COLORS[label] || "bg-gray-200 text-gray-800"}`}>{label}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
      </motion.section>

          {/* right: AI Reasoning & QA / extracted / phishing */}
          <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay:0.6 }}
      >

          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-md p-5 overflow-auto max-h-[70vh]">
              <div className="flex items-center gap-3 mb-4">
                <Layers className="w-5 h-5 text-purple-600" />
                <h4 className="font-semibold text-gray-900">AI Reasoning & Evidence</h4>
              </div>

              {!Number.isInteger(selectedClaimIndex) && (
                <div className="text-center py-8 text-gray-500">
                  <HelpCircle className="w-12 h-12 mx-auto text-gray-300 mb-2" />
                  <p>Select a claim to view detailed analysis</p>
                </div>
              )}

              {Number.isInteger(selectedClaimIndex) && (() => {
                const rpt = allReports[selectedClaimIndex];
                if (!rpt) return <div className="text-sm text-gray-500">Claim missing.</div>;

                const verdictData = rpt.gemini_verdict || {};
                const label = rpt?.final_verdict || rpt?.verdict || (rpt?.gemini_verdict?.verdict) || "Unclear";
                const overall = verdictData.overall || rpt.human_explanation || rpt.qa_summary || rpt.summary || "";
                const issues = verdictData.issues || rpt.issues || [];
                const combinedSources = combineAndDedupSources(rpt);
                const trust_score_pct = normalizeTrustPct(rpt?.trust_score_pct ?? rpt?.trust);
                const confNorm = normalizeConfidence(rpt?.confidence ?? rpt?.confidence_score ?? rpt?.confidence_pct ?? null);

                return (
                  <div className="space-y-4">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-lg font-semibold text-gray-800">{rpt.claim || rpt.text || `Claim ${selectedClaimIndex + 1}`}</div>
                        <div className="text-xs text-gray-500 mt-1">{rpt.context || rpt.source || ""}</div>
                      </div>

                      <div className="text-right">
                        <div className="text-xs text-gray-500">Trust Score</div>
                        <div className="font-medium text-lg">{trust_score_pct != null ? `${trust_score_pct}%` : "—"}</div>
                        <div className={`text-xs px-2 py-1 rounded-full mt-2 ${BADGE_COLORS[label] || "bg-gray-200"}`}>{label}</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Confidence</div>
                        <ConfidenceTrustBars confidence={confNorm} trust={trust_score_pct ? trust_score_pct / 100 : 0} />
                      </div>
                      <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded-lg">
                        Trust score indicates how reliable the source evidence is for this claim.
                      </div>
                    </div>

                    {overall && (
                      <div className={`mt-4 p-4 rounded-lg ${Gemini_COLORS[label] || "bg-gray-100"}`}>
                        <div className="text-sm font-medium text-gray-900 mb-2">Gemini Analysis</div>
                        <div className="mt-1 text-sm whitespace-pre-wrap">{overall}</div>
                      </div>
                    )}

                    {issues?.length > 0 && (
                      <div className="mt-4">
                        <h5 className="text-sm font-medium text-gray-700 mb-2">Identified Issues</h5>
                        <ul className="list-disc list-inside text-sm text-gray-700 mt-2 space-y-1">
                          {issues.map((it, ii) => {
                            if (!it) return null;
                            if (typeof it === "string") return <li key={ii}>{it}</li>;
                            const key = it.key || it.name || it.type || null;
                            const value = it.value || it.description || JSON.stringify(it);
                            return <li key={ii}><strong>{key ? `${key}: ` : ""}</strong>{value}</li>;
                          })}
                        </ul>
                      </div>
                    )}

                    <div className="mt-4">
                      <h5 className="text-sm font-medium text-gray-700 mb-3">Supporting Sources</h5>
                      <div className="grid grid-cols-1 gap-3">
                        {combinedSources.length === 0 && <div className="text-sm text-gray-500">No sources available.</div>}
                        {combinedSources.map((s, i) => {
                          const domain = s.url ? getDomainFromUrl(s.url) : (s.title || "").split(" ")[0];
                          const href = safeHref(s.url);
                          return (
                            <div key={i} className="flex gap-3 items-start border border-gray-200 rounded-lg p-3 bg-gray-50">
                              <FaviconAvatar url={s.url} title={s.title} size={44} />
                              <div className="flex-1">
                                <div className="text-sm font-semibold text-gray-800">{s.title || domain || "Source"}</div>
                                <div className="text-sm text-gray-600 mt-1 line-clamp-2">{s.snippet}</div>
                                {href && (
                                  <a
                                    href={href}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-teal-600 text-sm mt-2 inline-flex items-center gap-1 hover:text-teal-800"
                                  >
                                    <ExternalLinkIcon /> Visit source
                                  </a>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>

          {/* Enhanced QA Section */}
<div className="mt-1">
  <QASection
    qaSummary={
      // prefer the explicit qaSummary state (returned by Gradio), else fallback to the selected claim's qa_summary or top-level report
      qaSummary ??
      (report?.reports?.[selectedClaimIndex]?.qa_summary) ??
      report?.qa_summary
    }
    defaultOpenIndex={0}
  />
</div>


          </div>
      </motion.section>
        </div>

        {/* raw modal */}
        {showRaw && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
            <div className="bg-white w-full max-w-5xl rounded-xl shadow-lg overflow-hidden flex flex-col h-4/5">
              <div className="flex items-center justify-between p-4 border-b">
                <div className="flex items-center gap-2">
                  <Code className="w-5 h-5 text-gray-700" />
                  <h3 className="font-semibold">Raw JSON</h3>
                </div>
                <button
                  className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1 rounded hover:bg-gray-100"
                  onClick={() => setShowRaw(false)}
                >
                  Close
                </button>
              </div>
              <div className="p-4 flex-1 overflow-auto">
                <pre className="text-xs font-mono whitespace-pre-wrap bg-gray-50 p-4 rounded-lg">
                  {JSON.stringify(report || demoReport, null, 2)}
                </pre>
              </div>
              <div className="p-3 border-t flex justify-end">
                <button
                  onClick={() => setShowRaw(false)}
                  className="px-4 py-2 rounded bg-gray-100 text-gray-700 hover:bg-gray-200"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}

        <footer className="mt-12 text-center text-xs text-gray-500 select-none">
          <div className="max-w-3xl mx-auto p-4 rounded-lg bg-white border border-gray-200">
            <div className="font-semibold">© {new Date().getFullYear()} NewsOrchestra</div>
            <div className="mt-1">Misinformation Detection and analysis platform</div>
          </div>
        </footer>
      </div>
    </div>
  );
}
