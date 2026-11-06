# AI-Powered Image Editing Platform 🎨

> **Course Project — Module 5, Week 2**  
> Edit Interface + Version History Deliverable

---

## 🚀 Quick Start

```bash
# 1. Clone / navigate to project
cd ai-image-editor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API keys
cp .env.example .env
# Edit .env and add OPENAI_API_KEY and/or GEMINI_API_KEY

# 4. Run the app
streamlit run app/main.py
```

The app opens at **http://localhost:8501**

---

## 🗂️ Project Structure

```
ai-image-editor/
├── app/
│   ├── main.py          ← Streamlit UI (all pages)
│   ├── ai_clients.py    ← OpenAI + Gemini API wrappers
│   ├── database.py      ← SQLite persistence layer
│   └── image_utils.py   ← File I/O + thumbnail helpers
├── data/
│   ├── images/          ← Original uploaded images
│   ├── versions/        ← Edited image versions
│   ├── thumbnails/      ← Auto-generated thumbnails
│   └── image_editor.db  ← SQLite database
├── .streamlit/
│   └── config.toml      ← Dark theme config
├── .env.example         ← API key template
├── requirements.txt
└── README.md
```

---

## ✨ Features (Week 2)

### 📚 Image Library
- Upload images (PNG, JPG, WEBP)
- Auto-captioning with GPT-4o or Gemini 1.5 Flash on upload
- Natural language search using text embeddings (cosine similarity)
- Delete images with all their versions

### ✏️ Edit Image
- Select any uploaded image
- **8 preset one-click edits** (sunset background, B&W, oil painting style, etc.)
- **Custom natural language prompt** — any instruction you can describe
- Side-by-side comparison (original vs edited)
- Download any edited version

### 🕒 Version History
- Every edit saves a numbered version (`v001`, `v002`, …)
- Timeline showing: version number, timestamp, provider, model, prompt used
- Thumbnail preview per version
- Download individual versions
- Version slider to preview any past version

---

## 🤖 AI Architecture

### Image Captioning
| Provider | Model |
|----------|-------|
| Gemini   | `gemini-1.5-flash` |
| OpenAI   | `gpt-4o` |

### Image Editing Pipeline
```
User Prompt + Original Image
        │
        ▼
   Vision Model (GPT-4o / Gemini Flash)
   "Given this image + instruction, write a detailed
    generation prompt describing the edited result"
        │
        ▼
   Image Generation Model (DALL·E 3 / Imagen 3)
   Generates the edited image
        │
        ▼
   Saved as new version → Thumbnail generated → DB record inserted
```

### Natural Language Search
```
User Query → embed_text() → query embedding (float[])
                                │
All images → caption → embedding stored in DB (JSON blob)
                                │
                        cosine_similarity()
                                │
                        Ranked results (threshold 0.3)
```

---

## 🔑 API Keys

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | GPT-4o captioning + planning, DALL·E 3 editing, text-embedding-3-small search |
| `GEMINI_API_KEY` | Gemini 1.5 Flash captioning + planning, Imagen 3 editing, text-embedding-004 search |

You only need **one** key to use the app. The sidebar lets you switch providers per session, and the edit panel lets you choose the provider per individual edit.

---

## 📝 Prompt Engineering Notes

### Captioning prompt
```
"Describe this image in 2–3 sentences. Be specific about colours,
objects, setting, and mood. Output plain text only."
```
> Short, structured output prevents the model from adding markdown, lists, or preamble — which would break embedding quality and search relevance.

### Edit planning prompt (sent to Vision model)
```
"The user wants to edit this image: '{prompt}'.
Write a detailed [DALL·E 3 / Imagen] image generation prompt describing
the RESULT image (after the edit is applied). Keep background, style,
and other elements intact. Output the prompt only, no extra text."
```
> Two-stage approach: the vision model **sees** the image and **knows** what to preserve. The generation model then produces a faithful edit rather than hallucinating a fresh scene.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.35+ |
| AI — Vision / Planning | GPT-4o, Gemini 1.5 Flash |
| AI — Image Generation | DALL·E 3, Imagen 3 |
| AI — Embeddings | text-embedding-3-small, text-embedding-004 |
| Database | SQLite (via Python `sqlite3`) |
| Image Processing | Pillow |
| Config | python-dotenv |

---

## 🌐 Deployment (Streamlit Cloud)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → set **Main file path** = `app/main.py`
4. Add `OPENAI_API_KEY` and/or `GEMINI_API_KEY` as **Secrets**
5. Deploy 🚀

> [!NOTE]
> For cloud deployment, consider switching `data/` storage to a cloud bucket (GCS/S3) since Streamlit Cloud has an ephemeral filesystem.
