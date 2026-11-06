# AI-Powered Image Editing Platform 🎨

> **Course Project — Module 5: Week 2**  
> **Student:** Muskan Dawar  
> **Deliverable:** Editing + Versioning  

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/muskan-1301/ai-image-editor.git
cd ai-image-editor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
cp .env.example .env
# Edit .env and add GEMINI_API_KEY=AIza...

# 4. Run the app
streamlit run app/main.py
```

The app opens automatically at **http://localhost:8501**

---

## 🗂️ Project Structure

```
ai-image-editor/
├── app/
│   ├── main.py          ← Streamlit UI (Image Library & Edit Modes)
│   ├── ai_clients.py    ← Google Gemini Vision & Generative AI engine
│   ├── database.py      ← SQLite schema for images and version tracking
│   └── image_utils.py   ← File saving, path routing, and thumbnail helpers
├── data/
│   ├── images/          ← Master records of uploaded original images
│   ├── versions/        ← Edited image versions (v001, v002...)
│   ├── thumbnails/      ← Auto-generated 256x256 thumbnails
│   └── image_editor.db  ← SQLite relational database
├── .streamlit/
│   └── config.toml      ← Dark theme configuration
├── .env.example         ← API key template
├── requirements.txt
└── README.md
```

---

## ✨ Features (Week 2 Deliverable)

### 📚 Image Library
- Upload images in PNG, JPG, JPEG, and WEBP formats.
- **Automated Gemini Vision Captioning** on upload using `gemini-2.5-flash`.
- **Semantic Natural Language Search** using `gemini-embedding-001` and cosine similarity.
- Delete images and all associated version trees with one click.

### ✏️ Edit Mode
- Select any uploaded photo from your library.
- **8 preset one-click edits** (Ocean, Sunset, Palm trees, Oil painting, Winter, Night lights, B&W, Studio lighting).
- **Custom conversational edit instructions** (e.g., *"add a calm blue ocean with gentle waves in the background"*).
- Interactive side-by-side comparison (Original vs. Edited version).
- Instant download button for all generated versions.

### 🕒 Version History Logic
- Every edit automatically saves a sequential version (`v001`, `v002`, `v003`, etc.).
- Relational tracking in SQLite with timestamps, prompts used, model metadata, and parent image IDs.
- Thumbnail preview card for each version.
- **Interactive Version Slider** allowing users to inspect past states.

---

## 🤖 AI Architecture & Pipeline

### 1. Vision & Multimodal Edit Interpretation
```
User Prompt + Original Image
            │
            ▼
Google Gemini Vision (gemini-2.5-flash)
"Inspect image composition, subjects, and style.
 Contextualize the user's edit instruction into a 
 refined scene synthesis prompt."
            │
            ▼
Generative Diffusion Synthesis
Renders the modified scene preserving the original subject
            │
            ▼
Save Version PNG → Generate 256x256 Thumbnail → Store in SQLite
```

### 2. Semantic Search Pipeline
```
User Query (e.g. "two girls smiling")
            │
            ▼
gemini-embedding-001 (3072-dimensional vector)
            │
            ▼
Cosine Similarity against stored caption embeddings in SQLite
            │
            ▼
Ranked results returned to Library View
```

---

## 📝 Prompt Engineering Templates

### A. Vision Captioning Prompt
```text
"Describe this image in 2–3 sentences. Be specific about colours, objects, setting, and mood. Output plain text only."
```

### B. Multimodal Edit Formulation Prompt
```text
"The user wants to edit this picture with the instruction: '{prompt}'.
Describe the final edited scene in 1 concise, vivid sentence for an AI image generator.
Keep the core subject/style from the original image and incorporate the edit.
Return only the scene description."
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend UI | Streamlit 1.56+ |
| Vision & Planning | Google Gemini 2.5 Flash (`google-genai`) |
| Semantic Search | Gemini Embedding 001 |
| Database | SQLite3 |
| Image Processing | Pillow (PIL) |
| Environment | python-dotenv |
