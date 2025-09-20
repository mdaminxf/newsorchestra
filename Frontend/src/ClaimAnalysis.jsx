// src/NewsOrchestraUI.jsx
import React, { useState } from "react";
import {
  Container,
  TextField,
  Button,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Grid,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Box,
  Paper,
  Divider,
  IconButton,
  Tabs,
  Tab,
  Tooltip,
  Collapse,
  Stack,
  Fab,
  Zoom,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import FactCheckIcon from '@mui/icons-material/FactCheck';
import DescriptionIcon from '@mui/icons-material/Description';
import InsightsIcon from '@mui/icons-material/Insights';
import ArticleIcon from '@mui/icons-material/Article';
import ListIcon from '@mui/icons-material/List';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import FolderCopyIcon from '@mui/icons-material/FolderCopy';
import BookIcon from '@mui/icons-material/Book';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import PublicIcon from '@mui/icons-material/Public';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import FindInPageIcon from '@mui/icons-material/FindInPage';
import CodeIcon from '@mui/icons-material/Code';
import DataObjectIcon from '@mui/icons-material/DataObject';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import PlaylistAddCheckIcon from '@mui/icons-material/PlaylistAddCheck';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import ImageIcon from "@mui/icons-material/Image";
import LinkIcon from "@mui/icons-material/Link";
import InfoIcon from "@mui/icons-material/Info";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import DoneIcon from "@mui/icons-material/Done";
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { Client } from "@gradio/client";

/* ============================
   Utilities & constants
   ============================ */
const verdictColors = {
  True: "#4caf50",
  False: "#f44336",
  Misleading: "#ff9800",
  Unclear: "#9e9e9e",
  Mixed: "#ffb300",
  Unsupported: "#9e9e9e",
};

const fadeSx = {
  animation: "fadeIn 0.45s ease",
  "@keyframes fadeIn": {
    "0%": { opacity: 0, transform: "translateY(6px)" },
    "100%": { opacity: 1, transform: "translateY(0)" },
  },
};

const safeRender = (value) => {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number") return value;
  if (Array.isArray(value)) {
    return value
      .map((v) => (typeof v === "object" ? JSON.stringify(v) : String(v)))
      .join(", ");
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const safeHref = (maybeUrl) => {
  if (!maybeUrl) return null;
  if (typeof maybeUrl === "string") return maybeUrl;
  try {
    return String(maybeUrl);
  } catch {
    return null;
  }
};

const copyToClipboard = async (text, setCopied) => {
  try {
    await navigator.clipboard.writeText(text);
    if (setCopied) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  } catch (e) {
    // ignore
  }
};

/* ============================
   Component
   ============================ */
export default function NewsOrchestraUI() {
  const [textOrUrl, setTextOrUrl] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [runSerp, setRunSerp] = useState(true);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [extractedArticle, setExtractedArticle] = useState("");
  const [extractedClaims, setExtractedClaims] = useState([]);
  const [errorMsg, setErrorMsg] = useState(null);
  const [copied, setCopied] = useState(false);
  const [openWhy, setOpenWhy] = useState(true);
  const [showArticlePanel, setShowArticlePanel] = useState(false);
  const [summaryExpanded, setSummaryExpanded] = useState(false);
  const [auditTab, setAuditTab] = useState(0);
  const [showRawData, setShowRawData] = useState(false);
  
  const firstReport =
    report && Array.isArray(report.reports) && report.reports.length > 0
      ? report.reports[0]
      : null;

  // trigger backend analyze
  const analyze = async () => {
    setLoading(true);
    setReport(null);
    setExtractedArticle("");
    setExtractedClaims([]);
    setErrorMsg(null);
    setShowArticlePanel(false);
    setSummaryExpanded(false);

    try {
      const client = await Client.connect("ANISA09/atlas");
      const payload = {
        text_or_url: textOrUrl || "",
        image_url: imageUrl || "",
        run_serp: !!runSerp,
      };

      const result = await client.predict("/on_analyze", payload);
      const outputs = result?.data;
      if (!outputs || !Array.isArray(outputs) || outputs.length < 3) {
        throw new Error("Unexpected backend response format — expected 3 outputs");
      }
      const receivedReport = outputs[0] ?? null;
      const receivedArticle = outputs[1] ?? "";
      const receivedClaims = outputs[2] ?? [];

      if (!receivedReport || typeof receivedReport !== "object") {
        throw new Error("Backend report missing or invalid");
      }

      setReport(receivedReport);
      setExtractedArticle(receivedArticle || "");
      setExtractedClaims(Array.isArray(receivedClaims) ? receivedClaims : []);
      setShowArticlePanel(Boolean(receivedArticle && receivedArticle.length > 30));
    } catch (err) {
      console.error(err);
      setErrorMsg(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  /* Build a concise AI summary string */
  const buildAISummary = () => {
    if (!report) return "No analysis yet — press Analyze to start.";
    const gemOver = firstReport?.gemini_verdict?.overall;
    const reasons = firstReport?.reasons?.length ? firstReport.reasons.join(" • ") : null;
    const short = gemOver ? String(gemOver) : reasons ? String(reasons) : `Verdict: ${report?.summary?.dominant_verdict || "Unclear"}`;
    return short;
  };

  const renderReason = (r) => {
    if (!r) return "";
    if (typeof r === "string") return r;
    if (typeof r === "object" && r.description) {
      return `${r.type ? `[${r.type}] ` : ""}${r.description}`;
    }
    return JSON.stringify(r);
  };
  const buildQA = () => {
    const qa = [];

const whyBase = firstReport?.reasons?.length
  ? firstReport.reasons.map(renderReason).join(" • ")
  : "The system aggregates model verdicts, search evidence, and image analysis to make a conservative judgment.";
    qa.push({ q: "Why did the AI reach this verdict?", a: whyBase });

    const evidenceCount = firstReport?.evidence_agg?.evidence?.length || 0;
    const imageFound = firstReport?.image_analysis?.fetched ? true : false;
    const howArr = [
      `Web evidence items aggregated: ${evidenceCount}`,
      imageFound ? "Image analyzed for metadata, ELA, and reverse-image search." : null,
      firstReport?.hf_classifier ? "Auxiliary zero-shot classifier contributed label signals." : null,
    ].filter(Boolean);
    qa.push({ q: "How did the AI verify the claim?", a: howArr.join(" • ") });

    const suggestions = [
      "Check primary source links (open the source cards).",
      "Look for multiple independent corroborations.",
      "For images: check reverse-image matches and EXIF/metadata.",
      "If claim is vague, ask for exact dates/numbers before trusting.",
    ];
    qa.push({ q: "What should I check next?", a: suggestions.join(" • ") });

    return qa;
  };

  const buildRelationalEvidence = (r) => {
    const map = { citations: [], web: [], reverse: [] };
    if (!r) return map;
    const webE = r?.evidence_agg?.evidence || [];
    map.web = webE.map((e) => ({ title: e.title, snippet: e.snippet, link: e.link, domain: e.domain, trust: e.trust }));
    const rev = r?.image_analysis?.serpapi_reverse?.result?.organic_results || [];
    map.reverse = rev.map((x) => ({ title: x.title, snippet: x.snippet, link: x.link || x.displayed_link, domain: (x.link && (() => { try { return new URL(x.link).hostname } catch { return x.displayed_link } })()) }));
    const cits = r?.gemini_verdict?.citations || [];
    map.citations = cits.map((c) => ({ source: c.source, snippet: c.snippet, url: c.url }));
    return map;
  };

  const aiSummary = buildAISummary();
  const qa = buildQA();
  const evid = buildRelationalEvidence(firstReport);

  const displayConfidence = firstReport && typeof firstReport.confidence === "number" ? firstReport.confidence : 0;
  const displayFinalScore = firstReport && typeof firstReport.final_score === "number" ? firstReport.final_score : 0;

  return (
    <Box sx={{
      minHeight: '100vh',
      background: 'linear-gradient(120deg, rgba(224,195,252,0.6) 0%, rgba(255, 255, 255, 0.89) 100%)',
      padding: '20px 0',
      position: 'relative',
      overflow: 'hidden',
      '&::before': {
        content: '""',
        position: 'absolute',
        width: '400px',
        height: '400px',
        background: 'radial-gradient(rgba(240, 20, 57, 0.4) 0%, transparent 70%)',
        borderRadius: '50%',
        top: '-100px',
        left: '-100px',
        zIndex: 0,
      },
      '&::after': {
        content: '""',
        position: 'absolute',
        width: '500px',
        height: '500px',
        background: 'radial-gradient(rgba(0, 255, 85, 0.3) 0%, transparent 70%)',
        borderRadius: '50%',
        bottom: '-150px',
        top:'400px',
        right: '-150px',
        zIndex: 0,
      }
    }}>
      <Container maxWidth="lg" sx={{ 
        py: 4, 
        position: 'relative',
        zIndex: 1 
      }}>
        
        {/* Header */}
        <Box sx={{ textAlign: "center", mb: 4, position: 'relative' }}>
          <Box sx={{
            position: 'absolute',
            top: -20,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 100,
            height: 100,
            background: 'rgba(255,255,255,0.2)',
            borderRadius: '50%',
            filter: 'blur(20px)',
            zIndex: -1
          }} />
          <Typography variant="h3" sx={{ 
            fontWeight: 800, 
            background: 'linear-gradient(45deg, #ebebebff, #141414ff)',
            backgroundClip: 'text',
            textFillColor: 'transparent',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 1
          }}>
            NewsOrchestra
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ 
            mb: 2,
            background: 'rgba(255,255,255,0.7)',
            backdropFilter: 'blur(10px)',
            padding: '8px 16px',
            borderRadius: '20px',
            display: 'inline-block'
          }}>
            Gemini-powered fact-checking with multimodal analysis
          </Typography>
        </Box>

        {/* Input Section */}
        <Paper elevation={3} sx={{ 
          p: 3, 
          mb: 3, 
          borderRadius: 3,
          background: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.5)',
          ...fadeSx 
        }}>
          <Grid container spacing={2} alignItems="stretch">
            <Grid item xs={12} md={7}>
              <TextField
                label="Article text or URL"
                placeholder="Paste article text or enter a URL (https://...)"
                multiline
                rows={4}
                fullWidth
                value={textOrUrl}
                onChange={(e) => setTextOrUrl(e.target.value)}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    background: 'rgba(255,255,255,0.7)',
                    borderRadius: 2,
                  }
                }}
              />
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                Accepts raw article text or a URL. If the URL has no extractable article text, the backend will fallback to SERP snippets.
              </Typography>
            </Grid>

            <Grid item xs={12} md={7}>
              <TextField
                label="Image URL"
                placeholder="https://example.com/image.jpg"
                fullWidth
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    background: 'rgba(255,255,255,0.7)',
                    borderRadius: 2,
                  }
                }}
              />
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                Used for multimodal claim generation and image analysis.
              </Typography>
            </Grid>

            <Grid item sx={{ display: 'flex', alignItems: 'center',marginBottom:'100px' }}>
              <Button
                variant="contained"
                onClick={analyze}
                disabled={loading || (!textOrUrl && !imageUrl)}
                fullWidth
                startIcon={loading ? null : <AutoAwesomeIcon />}
              >
                {loading ? "Analyzing…" : "Analyze"}
              </Button>
            </Grid>
          </Grid>
        </Paper>

        {/* Error */}
        {errorMsg && (
          <Alert severity="error" sx={{ mb: 2, borderRadius: 2, ...fadeSx }}>
            <Typography variant="body2">{safeRender(errorMsg)}</Typography>
          </Alert>
        )}

        {/* Loading */}
        {loading && (
          <Box sx={{ mb: 2, background: 'rgba(255,255,255,0.7)', p: 2, borderRadius: 2, backdropFilter: 'blur(10px)' }}>
            <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center' }}>
              <AutoAwesomeIcon sx={{ mr: 1, animation: 'pulse 1.5s infinite' }} />
              Working — this can take a few seconds
            </Typography>
            <LinearProgress sx={{ mt: 1, borderRadius: 1 }} />
          </Box>
        )}

        {/* Results Section */}
        {report && (
          <Box sx={{ ...fadeSx }}>
            {/* Claim and Verdict */}
            <Paper elevation={2} sx={{ 
              p: 3, 
              mb: 3, 
              borderRadius: 3,
              background: 'rgba(255,255,255,0.8)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255,255,255,0.5)',
            }}>
              <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2, alignItems: { md: 'center' } }}>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h6" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center' }}>
                    <FactCheckIcon sx={{ mr: 1, color: 'primary.main' }} />
                    Claim Analysis
                  </Typography>
                  <Typography variant="body1" sx={{ mt: 1, fontWeight: 500 }}>
                    {firstReport?.claim || "No specific claim was extracted"}
                  </Typography>
                </Box>
                
                <Chip 
                  label={firstReport?.final_verdict || report.summary?.dominant_verdict || "Unclear"} 
                  sx={{ 
                    backgroundColor: verdictColors[firstReport?.final_verdict] || "#9e9e9e", 
                    color: "#fff", 
                    fontWeight: 800,
                    fontSize: '1rem',
                    minWidth: 120,
                    height: 40,
                    py: 1
                  }} 
                />
              </Box>
            </Paper>

            {/* AI Summary */}
            <Paper elevation={2} sx={{ 
              p: 3, 
              mb: 3, 
              borderRadius: 3,
              background: 'rgba(255,255,255,0.8)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255,255,255,0.5)',
            }}>
              <Typography variant="h6" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', mb: 2 }}>
                <InsightsIcon sx={{ mr: 1, color: 'primary.main' }} />
                AI Explanation
              </Typography>
              <Box sx={{ 
                p: 2, 
                borderRadius: 2, 
                background: 'rgba(255,255,255,0.5)',
                borderLeft: `4px solid ${verdictColors[firstReport?.final_verdict] || '#9e9e9e'}` 
              }}>
                <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
                  {aiSummary}
                </Typography>
              </Box>
            </Paper>

            {/* Confidence & Trust */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ 
                  p: 2, 
                  borderRadius: 3,
                  background: 'rgba(255,255,255,0.8)',
                  backdropFilter: 'blur(10px)',
                  ...fadeSx 
                }}>
                  <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center' }}>
                    <InfoIcon sx={{ mr: 1, fontSize: 20 }} />
                    Confidence Level
                  </Typography>
                  <Typography variant="h4" sx={{ my: 1, fontWeight: 700 }}>{((displayConfidence) * 100).toFixed(1)}%</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(displayConfidence) * 100} 
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    How confident the system is in the overall conclusion.
                  </Typography>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ 
                  p: 2, 
                  borderRadius: 3,
                  background: 'rgba(255,255,255,0.8)',
                  backdropFilter: 'blur(10px)',
                  ...fadeSx 
                }}>
                  <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center' }}>
                    <DoneIcon sx={{ mr: 1, fontSize: 20 }} />
                    Trust Score
                  </Typography>
                  <Typography variant="h4" sx={{ my: 1, fontWeight: 700 }}>{((displayFinalScore) * 100).toFixed(1)}%</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(displayFinalScore) * 100} 
                    color="success" 
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Higher = more supporting, trustworthy sources found.
                  </Typography>
                </Card>
              </Grid>
            </Grid>

            {/* Q/A educational */}
            <Paper elevation={2} sx={{ 
              p: 3, 
              mb: 3, 
              borderRadius: 3,
              background: 'rgba(255,255,255,0.8)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255,255,255,0.5)',
            }}>
              <Typography variant="h6" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', mb: 2 }}>
                <HelpOutlineIcon sx={{ mr: 1, color: 'primary.main' }} />
                Analysis Details
              </Typography>
              
              <Grid container spacing={2}>
                {qa.map((item, idx) => (
                  <Grid item xs={12} md={6} key={idx}>
                    <Card variant="outlined" sx={{ 
                      p: 2, 
                      height: '100%',
                      borderRadius: 2,
                      background: 'rgba(255,255,255,0.5)',
                    }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: 'primary.main' }}>
                        {item.q}
                      </Typography>
                      <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", mt: 1 }}>
                        {safeRender(item.a)}
                      </Typography>
                    </Card>
                  </Grid>
                ))}
              </Grid>

              <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                  Quick Verification Checklist
                </Typography>
                <Box component="ul" sx={{ pl: 2, m: 0 }}>
                  <li><Typography variant="body2">Does the claim cite primary sources?</Typography></li>
                  <li><Typography variant="body2">Are multiple independent sources reporting the same facts?</Typography></li>
                  <li><Typography variant="body2">Does the image (if used) have verifiable origin / EXIF / reverse-image matches?</Typography></li>
                  <li><Typography variant="body2">Is the claim precise and testable (not vague or opinion)?</Typography></li>
                </Box>
              </Box>
            </Paper>

            {/* Evidence Section */}
            <Paper elevation={2} sx={{ 
              p: 3, 
              mb: 3, 
              borderRadius: 3,
              background: 'rgba(255,255,255,0.8)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255,255,255,0.5)',
            }}>
              <Typography variant="h6" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', mb: 3 }}>
                <FolderCopyIcon sx={{ mr: 1, color: 'primary.main' }} />
                Supporting Evidence
              </Typography>

              {/* Citations */}
              {firstReport?.gemini_verdict?.citations?.length > 0 && (
                <Box sx={{ mb: 4 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center' }}>
                    <BookIcon sx={{ mr: 1, fontSize: 20 }} />
                    Research Citations ({firstReport.gemini_verdict.citations.length})
                  </Typography>
                  <Grid container spacing={2}>
                    {firstReport.gemini_verdict.citations.map((c, idx) => {
                      const href = safeHref(c.url);
                      return (
                        <Grid key={idx} item xs={12} md={6}>
                          <Card 
                            variant="outlined" 
                            sx={{ 
                              height: "100%", 
                              display: "flex", 
                              flexDirection: "column",
                              borderRadius: 2,
                              transition: 'all 0.2s ease',
                              '&:hover': {
                                boxShadow: 3,
                                transform: 'translateY(-2px)'
                              }
                            }}
                          >
                            <Box sx={{ p: 2, flexGrow: 1 }}>
                              <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 1 }}>
                                <BookmarkIcon sx={{ fontSize: 16, mr: 1, mt: 0.25, color: 'primary.main' }} />
                                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                  {safeRender(c.source) || "Unknown Source"}
                                </Typography>
                              </Box>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                {safeRender(c.snippet) || "No snippet available"}
                              </Typography>
                            </Box>
                            <Box sx={{ 
                              display: "flex", 
                              justifyContent: "space-between", 
                              alignItems: "center", 
                              p: 1.5, 
                              backgroundColor: 'grey.50',
                              borderTop: '1px solid',
                              borderColor: 'divider'
                            }}>
                              {href ? (
                                <Button
                                  size="small"
                                  component="a"
                                  href={href}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  startIcon={<OpenInNewIcon />}
                                  sx={{ textTransform: 'none' }}
                                >
                                  View Source
                                </Button>
                              ) : (
                                <Typography variant="caption" color="text.secondary">
                                  No URL available
                                </Typography>
                              )}
                            </Box>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                </Box>
              )}

              {/* Web evidence */}
              {firstReport?.evidence_agg?.evidence?.length > 0 && (
                <Box sx={{ mb: 4 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center' }}>
                    <PublicIcon sx={{ mr: 1, fontSize: 20 }} />
                    Web Evidence ({firstReport.evidence_agg.evidence.length})
                  </Typography>
                  <Grid container spacing={2}>
                    {firstReport.evidence_agg.evidence.map((ev, idx) => {
                      const href = safeHref(ev.link);
                      const trustScore = Math.round((ev.trust || 0) * 100);
                      return (
                        <Grid item xs={12} md={6} key={idx}>
                          <Card 
                            variant="outlined" 
                            sx={{ 
                              height: "100%", 
                              display: "flex", 
                              flexDirection: "column",
                              borderRadius: 2,
                              transition: 'all 0.2s ease',
                              '&:hover': {
                                boxShadow: 3,
                                transform: 'translateY(-2px)'
                              }
                            }}
                          >
                            <Box sx={{ p: 2, flexGrow: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                {safeRender(ev.title) || "Untitled"}
                              </Typography>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                {safeRender(ev.snippet) || "No description available"}
                              </Typography>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Typography variant="caption" color="text.secondary">
                                  {safeRender(ev.domain) || "Unknown domain"}
                                </Typography>
                              </Box>
                            </Box>
                            <Box sx={{ 
                              display: "flex", 
                              justifyContent: "space-between", 
                              alignItems: "center", 
                              p: 1.5, 
                              backgroundColor: 'grey.50',
                              borderTop: '1px solid',
                              borderColor: 'divider'
                            }}>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Typography variant="caption" sx={{ mr: 1 }}>
                                  Trust score:
                                </Typography>
                                <Chip 
                                  label={`${trustScore}%`} 
                                  size="small" 
                                  color={
                                    trustScore >= 80 ? "success" : 
                                    trustScore >= 60 ? "warning" : "error"
                                  }
                                  variant="filled"
                                />
                              </Box>
                              {href && (
                                <Button
                                  size="small"
                                  component="a"
                                  href={href}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  startIcon={<OpenInNewIcon />}
                                  sx={{ textTransform: 'none' }}
                                >
                                  Visit
                                </Button>
                              )}
                            </Box>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                </Box>
              )}

              {/* Reverse-image */}
              {firstReport?.image_analysis?.serpapi_reverse?.result?.organic_results?.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center' }}>
                    <ImageSearchIcon sx={{ mr: 1, fontSize: 20 }} />
                    Reverse Image Search Results ({firstReport.image_analysis.serpapi_reverse.result.organic_results.length})
                  </Typography>
                  <Grid container spacing={2}>
                    {firstReport.image_analysis.serpapi_reverse.result.organic_results.map((ir, idx) => {
                      const href = safeHref(ir.link || ir.displayed_link);
                      return (
                        <Grid item xs={12} md={6} key={idx}>
                          <Card variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
                            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                              <Box sx={{ flex: 1 }}>
                                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                  {safeRender(ir.title) || "Untitled"}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  {safeRender(ir.snippet) || "No description available"}
                                </Typography>
                              </Box>
                              {href && (
                                <IconButton 
                                  size="small" 
                                  component="a" 
                                  href={href} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  sx={{ ml: 1 }}
                                >
                                  <OpenInNewIcon />
                                </IconButton>
                              )}
                            </Box>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                </Box>
              )}

              {/* Image analysis */}
              {firstReport?.image_analysis && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center' }}>
                    <PhotoCameraIcon sx={{ mr: 1, fontSize: 20 }} />
                    Image Analysis
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2"><strong>Content Type:</strong> {safeRender(firstReport.image_analysis.content_type || "—")}</Typography>
                      </Grid>
                      <Grid item xs={12} sm={6} md={4}>
                        <Typography variant="body2"><strong>File Size:</strong> {safeRender(firstReport.image_analysis.bytes_length || "—")}</Typography>
                      </Grid>
                      {firstReport.image_analysis.ela?.available && (
                        <Grid item xs={12} sm={6} md={4}>
                          <Typography variant="body2"><strong>ELA Score:</strong> {safeRender(firstReport.image_analysis.ela.ela_score)}</Typography>
                        </Grid>
                      )}
                      {firstReport.image_analysis.phash?.available && (
                        <Grid item xs={12} sm={6} md={4}>
                          <Typography variant="body2"><strong>Perceptual Hash:</strong> {safeRender(firstReport.image_analysis.phash.phash)}</Typography>
                        </Grid>
                      )}
                      {firstReport.image_analysis.exif?.has_exif && (
                        <Grid item xs={12} sm={6} md={4}>
                          <Typography variant="body2"><strong>EXIF Data:</strong> Available</Typography>
                        </Grid>
                      )}
                    </Grid>
                  </Paper>
                </Box>
              )}
            </Paper>

            {/* Extracted article */}
            <Collapse in={showArticlePanel && !!extractedArticle}>
              <Paper elevation={2} sx={{ 
                p: 3, 
                mb: 3, 
                borderRadius: 3,
                background: 'rgba(255,255,255,0.8)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255,255,255,0.5)',
              }}>
                <Typography variant="h6" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', mb: 2 }}>
                  <ArticleIcon sx={{ mr: 1, color: 'primary.main' }} />
                  Extracted Article Text
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, maxHeight: 300, overflow: 'auto' }}>
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                    {safeRender(extractedArticle)}
                  </Typography>
                </Paper>
              </Paper>
            </Collapse>
          </Box>
        )}

        {/* Floating Action Button for Raw Data */}
        {report && (
          <Zoom in={true}>
            <Fab
              color="primary"
              aria-label="raw-data"
              sx={{ 
                position: 'fixed', 
                bottom: 16, 
                right: 16,
                background: 'linear-gradient(45deg, #1976d2 30%, #0d47a1 90%)'
              }}
              onClick={() => setShowRawData(true)}
            >
              <CodeIcon />
            </Fab>
          </Zoom>
        )}

        {/* Raw Data Dialog */}
        <Dialog
          open={showRawData}
          onClose={() => setShowRawData(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
              <DataObjectIcon sx={{ mr: 1 }} />
              Raw Data (For Audit & Transparency)
            </Typography>
          </DialogTitle>
          <DialogContent>
            <Tabs value={auditTab} onChange={(e, newValue) => setAuditTab(newValue)} sx={{ mb: 2 }}>
              <Tab icon={<DataObjectIcon sx={{ fontSize: 20 }} />} label="Report JSON" />
              <Tab icon={<ArticleIcon sx={{ fontSize: 20 }} />} label="Article Text" />
              <Tab icon={<FormatListBulletedIcon sx={{ fontSize: 20 }} />} label="Extracted Claims" />
            </Tabs>
            
            {auditTab === 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Complete analysis report in JSON format
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 1, backgroundColor: 'grey.50', maxHeight: 300, overflow: 'auto' }}>
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: '0.75rem' }}>
                    {JSON.stringify(report, null, 2)}
                  </Typography>
                </Paper>
              </Box>
            )}
            
            {auditTab === 1 && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Extracted article text used for analysis
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 1, backgroundColor: 'grey.50', maxHeight: 300, overflow: 'auto' }}>
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                    {safeRender(extractedArticle) || "No article text was extracted"}
                  </Typography>
                </Paper>
              </Box>
            )}
            
            {auditTab === 2 && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Claims extracted from the content
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 1, backgroundColor: 'grey.50', maxHeight: 300, overflow: 'auto' }}>
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                    {safeRender(extractedClaims) || "No claims were extracted"}
                  </Typography>
                </Paper>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button 
              onClick={() => copyToClipboard(
                auditTab === 0 ? JSON.stringify(report, null, 2) : 
                auditTab === 1 ? safeRender(extractedArticle) : 
                safeRender(extractedClaims), 
                setCopied
              )} 
              startIcon={<ContentCopyIcon />}
            >
              Copy
            </Button>
            <Button onClick={() => setShowRawData(false)}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Initial State */}
        {!report && !loading && (
          <Paper elevation={1} sx={{ 
            p: 4, 
            textAlign: 'center', 
            borderRadius: 3,
            background: 'rgba(255,255,255,0.8)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.5)',
          }}>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              <PlaylistAddCheckIcon sx={{ fontSize: 40, mb: 2, color: 'grey.400' }} />
            </Typography>
            <Typography variant="h6" color="text.secondary">
              Ready to Analyze Content
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 2 }}>
              Enter an article (or URL) and optionally an image, then click Analyze. The AI will extract claims, 
              search for evidence, and produce an explainable verdict.
            </Typography>
       
          </Paper>
        )}
      </Container>
    </Box>
  );
}