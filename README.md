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
├── frontend/            # React + Tailwind UI
├── backend/             # Flask/FastAPI ML server
│   ├── app.py           # Hugging Face + custom models, ImageHash, Pillow, parsing tools, Genai
|   |── requirements.txt # All required Packages
├── public/              # Assets
└── README.md            # Documentation
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
Input Fields
![Input Fields](https://github.com/mdaminxf/newsorchestra/blob/main/public/image%20(4).png)
Analysis Summary
![Analysis Summary](https://github.com/mdaminxf/newsorchestra/blob/main/public/image%20(1).png)
Ai Resoning
![Ai Resoning](https://github.com/mdaminxf/newsorchestra/blob/main/public/image%20(2).png)
Evidence Section
![Evidence Section](https://github.com/mdaminxf/newsorchestra/blob/main/public/image%20(3).png)
---

## 🔮 Future Enhancements
- 🤖 **Vertex AI Integration** – Leverage Google Cloud’s Vertex AI for scalable training and inference.  
- 📽️ **Video Misinformation Detection** – Extend analysis to video content (frames + metadata).  
- 🔉 **Audio Misinformation Detection** – Speech-to-text + NLP pipeline for detecting manipulated audio.  
- 📡 **Real-Time Social Media Monitoring** – Continuous tracking of misinformation across Twitter/X, Facebook, and other platforms.  
- 🗣 **Multilingual Support** – Detect and analyze misinformation in multiple languages.  
- 📱 **Mobile Application** – Cross-platform mobile app for accessibility and wider adoption.  


---

## 👨‍💻 Contributors
- **Anisa Fatima** - Team Lead & Designer
- **Muhammad Amin Jilani** – Developer & Researcher  
