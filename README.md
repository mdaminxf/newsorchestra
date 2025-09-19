# 📰 AI-Powered Fake News & Misinformation Detection Tool

## 📌 Overview
This project is an AI-powered tool designed to detect, analyze, and visualize the spread of misinformation across social media and online platforms. The system leverages cutting-edge **machine learning models, NLP techniques, and generative AI (Gemini)** to identify misleading information and present clear, user-friendly insights.

---

## 🚀 Features
- 🔍 **Fake News Detection** – Classifies content as *reliable* or *misleading* using ML/NLP.  
- 🖼 **Image Verification** – Uses **Pillow + imagehash** to check manipulated/duplicate media.  
- 🤖 **AI Reasoning** – Integrates **Google Gemini & Hugging Face Transformers** for claim verification.  
- 🌐 **Web Scraping & Search APIs** – Cross-checks facts using **SerpAPI** & online sources.  
- 📊 **Visualization Dashboard** – Displays spread patterns, sources, and credibility scores.  
- 🎥 **Cinematic Reports** – Auto-generates summaries with professional storytelling.  

---

## 🛠️ Tech Stack

### 🎨 Frontend
- **React.js** – Component-based UI.  
- **TailwindCSS** – Modern responsive styling.  

### ⚙️ Backend
- **Pillow** – Image preprocessing.  
- **ImageHash** – Detect duplicate/manipulated images.  
- **Transformers (Hugging Face)** – NLP models for text classification & claim detection.  
- **GenAI (Gemini)** – Reasoning, fact verification, and summarization.  

### 🔗 APIs
- **Gemini API** – Advanced multimodal AI reasoning.  
- **Hugging Face Inference API** – ML/NLP predictions.  
- **SerpAPI** – Search engine fact cross-verification.  

---

## 📂 Project Structure
```bash
├── frontend/        # React + Tailwind UI
├── backend/         # Flask/FastAPI ML server
│   ├── models/      # Hugging Face + custom models
│   ├── utils/       # ImageHash, Pillow, parsing tools
├── public/          # Assets & icons
└── README.md        # Documentation
```

---

## ⚡ Workflow
1. 🖼 User uploads **image/text/news snippet**.  
2. 🔍 Backend verifies with **imagehash + NLP models**.  
3. 🤖 Gemini + Transformers provide **reasoning & fact-check**.  
4. 🌐 SerpAPI fetches real-time trusted sources.  
5. 📊 Dashboard shows **results, credibility score & spread analysis**.  

---

## 📸 Screenshots
*(Add UI dashboard, upload screen, results panel, etc.)*  

---

## 🔮 Future Enhancements
- 📡 Real-time monitoring of **Twitter/X, Facebook feeds**.  
- 🗣 Multilingual misinformation detection.  
- 📱 Mobile app version.  

---

## 👨‍💻 Contributors
- **Your Name** – Developer & Researcher  
