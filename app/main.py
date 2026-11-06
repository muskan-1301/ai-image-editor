"""
main.py — AI-Powered Image Editing Platform
Powered by Gemini Vision & Generative AI
"""

from __future__ import annotations
import io
import json
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "app"))
load_dotenv(ROOT / ".env")

from database import (
    init_db,
    insert_image,
    update_image_caption,
    insert_version,
    get_all_images,
    get_image,
    get_versions,
    next_version_num,
    delete_image,
)
from image_utils import save_uploaded_image, save_pil_image, timestamp_now
from ai_clients import caption_image, edit_image, embed_text, cosine_similarity

# ── Initialise DB ─────────────────────────────────────────────────────────────
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] { background: #0f1117; }
section[data-testid="stSidebar"] { background: #1a1d27; }

/* ---- Card grid ---- */
.img-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 8px;
    margin-bottom: 12px;
    cursor: pointer;
    transition: box-shadow .2s;
}
.img-card:hover { box-shadow: 0 0 0 2px #7c6af0; }
.img-caption {
    font-size: 0.75rem;
    color: #9fa8c0;
    margin-top: 4px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c9d1e9;
    border-left: 3px solid #7c6af0;
    padding-left: 10px;
    margin: 16px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🎨 AI Image Editor")
    st.caption("Powered by Google Gemini Vision & Generative AI")
    st.divider()

    st.markdown("### ⚙️ AI Engine")
    gem_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())
    st.markdown(
        f"{'✅' if gem_ok else '❌'} **Gemini AI Active**",
        unsafe_allow_html=True,
    )

    if not gem_ok:
        k = st.text_input("Enter Gemini API key", type="password")
        if k:
            os.environ["GEMINI_API_KEY"] = k
            st.rerun()

    st.divider()

    st.markdown("### 📂 Navigation")
    page = st.radio(
        "Go to",
        ["📚 Image Library", "✏️ Edit Image"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Storage path:\n`{ROOT / 'data' / 'images'}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: IMAGE LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def page_library():
    st.header("📚 Image Library")

    # ── Upload ────────────────────────────────────────────────────────────────
    with st.expander("⬆️ Upload a new image", expanded=False):
        up = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        col_up1, col_up2 = st.columns([3, 1])
        with col_up2:
            auto_caption = st.checkbox("Auto-caption with Gemini Vision", value=True)
        if up and st.button("Upload & Save", type="primary"):
            with st.spinner("Saving and analyzing image..."):
                img_id, filepath, thumb = save_uploaded_image(up)
                caption = ""
                embedding_json = ""
                if auto_caption:
                    try:
                        with st.spinner("Analyzing with Gemini Vision..."):
                            caption = caption_image(filepath, "gemini")
                            emb = embed_text(caption, "gemini")
                            embedding_json = json.dumps(emb)
                    except Exception as e:
                        st.warning(f"Captioning warning: {e}")
                insert_image({
                    "id": img_id,
                    "filename": up.name,
                    "filepath": filepath,
                    "thumbnail": thumb,
                    "caption": caption,
                    "embedding": embedding_json,
                    "uploaded_at": timestamp_now(),
                    "provider": "gemini" if caption else None,
                })
            st.success(f"✅ Uploaded **{up.name}**")
            if caption:
                st.info(f"**Gemini Caption:** {caption}")
            st.rerun()

    st.divider()

    # ── Search ────────────────────────────────────────────────────────────────
    all_images = get_all_images()
    col_s1, col_s2 = st.columns([5, 1])
    with col_s1:
        search_q = st.text_input(
            "🔍 Search images by natural language description",
            placeholder="e.g. two girls smiling, ocean sunset, red dress...",
            label_visibility="collapsed",
        )
    with col_s2:
        do_search = st.button("Search", width="stretch")

    if search_q and do_search:
        with st.spinner("Searching with AI semantic embeddings..."):
            try:
                q_emb = embed_text(search_q, "gemini")
                scored = []
                for img in all_images:
                    if img["embedding"]:
                        try:
                            img_emb = json.loads(img["embedding"])
                            score = cosine_similarity(q_emb, img_emb)
                            scored.append((score, img))
                        except Exception:
                            pass
                scored.sort(key=lambda x: x[0], reverse=True)
                display_images = [img for s, img in scored if s > 0.25]
                if not display_images:
                    st.warning("No close matches found. Showing all images.")
                    display_images = list(all_images)
            except Exception as e:
                st.error(f"Search error: {e}")
                display_images = list(all_images)
    else:
        display_images = list(all_images)

    # ── Grid ─────────────────────────────────────────────────────────────────
    if not display_images:
        st.info("No images uploaded yet. Upload one above! 👆")
        return

    st.markdown(f"<div class='section-header'>🖼️ {len(display_images)} Image(s)</div>",
                unsafe_allow_html=True)

    cols = st.columns(4)
    for i, img_row in enumerate(display_images):
        col = cols[i % 4]
        with col:
            thumb_path = img_row["thumbnail"]
            if thumb_path and Path(thumb_path).exists():
                col.image(thumb_path, width="stretch")
            else:
                col.image(img_row["filepath"], width="stretch")

            caption_text = img_row["caption"] or "*(no caption)*"
            st.markdown(
                f"<div class='img-caption'>{caption_text[:120]}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"📅 {img_row['uploaded_at'][:10]} · {img_row['filename']}")

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("✏️ Edit", key=f"edit_{img_row['id']}", width="stretch"):
                    st.session_state["selected_image_id"] = img_row["id"]
                    st.session_state["_page_override"] = "✏️ Edit Image"
                    st.rerun()
            with bcol2:
                if st.button("🗑️", key=f"del_{img_row['id']}", width="stretch", help="Delete image"):
                    for path in [img_row["filepath"], img_row["thumbnail"]]:
                        if path and Path(path).exists():
                            Path(path).unlink(missing_ok=True)
                    versions = get_versions(img_row["id"])
                    for v in versions:
                        for vp in [v["filepath"], v["thumbnail"]]:
                            if vp and Path(vp).exists():
                                Path(vp).unlink(missing_ok=True)
                    delete_image(img_row["id"])
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDIT IMAGE
# ══════════════════════════════════════════════════════════════════════════════

PRESET_EDITS = [
    ("🌊 Add ocean", "Add a calm blue ocean with gentle waves in the background"),
    ("🌅 Sunset background", "Change the background to a dramatic sunset sky"),
    ("🌿 Add palm trees & greenery", "Add tropical palm trees and lush greenery into the background"),
    ("🎨 Oil painting style", "Re-render the entire image in the artistic style of a colorful oil painting"),
    ("❄️ Snowy winter", "Transform the scene into a snowy winter landscape"),
    ("🌃 Night scene with lights", "Change the setting to a beautiful nighttime city with glowing bokeh lights"),
    ("🖤 B&W classic photo", "Convert to a classic black and white high-contrast portrait"),
    ("✨ Studio portrait lighting", "Enhance with professional cinematic studio lighting"),
]


def set_prompt_preset(text: str):
    """Callback to instantly update the prompt text box"""
    st.session_state["edit_prompt_input"] = text


def page_edit():
    st.header("✏️ Edit Image")

    all_images = get_all_images()
    if not all_images:
        st.info("No images in your library yet. Go to **📚 Image Library** to upload one!")
        return

    img_options = {f"{r['filename']}  [{r['id'][:8]}]": r["id"] for r in all_images}
    default_key = None
    if "selected_image_id" in st.session_state:
        for k, v in img_options.items():
            if v == st.session_state["selected_image_id"]:
                default_key = k
                break

    selected_label = st.selectbox(
        "Select image to edit",
        list(img_options.keys()),
        index=list(img_options.keys()).index(default_key) if default_key else 0,
    )
    image_id = img_options[selected_label]
    img_row = get_image(image_id)
    versions = get_versions(image_id)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("<div class='section-header'>📌 Current View</div>", unsafe_allow_html=True)

        ver_labels = ["Original"] + [f"v{v['version_num']:03d}" for v in versions]

        if len(ver_labels) == 1:
            ver_choice = "Original"
            st.caption("📌 Viewing: **Original** (no edits applied yet)")
        else:
            ver_choice = st.select_slider(
                "Version",
                options=ver_labels,
                value=ver_labels[-1],
            )

        if ver_choice == "Original":
            active_path = img_row["filepath"]
        else:
            vnum = int(ver_choice[1:])
            active_ver = next(v for v in versions if v["version_num"] == vnum)
            active_path = active_ver["filepath"]

        if Path(active_path).exists():
            st.image(active_path, width="stretch", caption=ver_choice)
        else:
            st.error("Image file not found on disk.")

        if img_row["caption"]:
            st.markdown(f"> 🤖 **Gemini Caption:** {img_row['caption']}")

        if st.button("🔄 Re-generate caption with Gemini"):
            with st.spinner("Analyzing image..."):
                try:
                    cap = caption_image(img_row["filepath"], "gemini")
                    emb = embed_text(cap, "gemini")
                    update_image_caption(image_id, cap, json.dumps(emb), "gemini")
                    st.success(f"Caption updated: {cap}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with right:
        st.markdown("<div class='section-header'>🕒 Version History</div>", unsafe_allow_html=True)

        if not versions:
            st.info("No edits yet. Apply an edit below to create version v001.")
        else:
            for v in reversed(versions):
                with st.container():
                    t_col, i_col = st.columns([2, 3])
                    with t_col:
                        st.markdown(
                            f"**v{v['version_num']:03d}**  \n"
                            f"🕐 {v['created_at'][11:16]}  \n"
                            f"📅 {v['created_at'][:10]}  \n"
                            f"🤖 Gemini AI"
                        )
                        st.caption(f"_{v['prompt'][:65] if v['prompt'] else ''}..._")
                    with i_col:
                        if v["thumbnail"] and Path(v["thumbnail"]).exists():
                            st.image(v["thumbnail"], width="stretch")
                        elif v["filepath"] and Path(v["filepath"]).exists():
                            st.image(v["filepath"], width="stretch")

                    if v["filepath"] and Path(v["filepath"]).exists():
                        with open(v["filepath"], "rb") as f:
                            st.download_button(
                                f"⬇️ Download v{v['version_num']:03d}",
                                data=f,
                                file_name=f"{img_row['filename'].rsplit('.', 1)[0]}_v{v['version_num']:03d}.png",
                                mime="image/png",
                                key=f"dl_{v['id']}",
                                width="stretch",
                            )
                    st.divider()

    # ── Edit Controls ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🛠️ Apply an Edit</div>", unsafe_allow_html=True)

    # Initialize session state for prompt box
    if "edit_prompt_input" not in st.session_state:
        st.session_state["edit_prompt_input"] = ""

    st.markdown("**Quick Presets** *(click any button to load it into the instruction box)*:")
    preset_cols = st.columns(4)
    for idx, (label, prompt_text) in enumerate(PRESET_EDITS):
        with preset_cols[idx % 4]:
            st.button(
                label,
                width="stretch",
                key=f"preset_{idx}",
                on_click=set_prompt_preset,
                args=(prompt_text,),
            )

    # Custom prompt text area bound to session_state
    edit_prompt = st.text_area(
        "✍️ Natural Language Edit Instruction",
        key="edit_prompt_input",
        placeholder="e.g. Add a calm blue ocean in the background, add palm trees, make it an oil painting...",
        height=85,
    )

    apply_col, preview_col = st.columns([1, 1])
    with apply_col:
        apply_edit = st.button(
            "🚀 Apply Edit",
            type="primary",
            width="stretch",
            disabled=not edit_prompt.strip(),
        )
    with preview_col:
        compare = st.checkbox("Show side-by-side comparison after edit", value=True)

    if apply_edit and edit_prompt.strip():
        source_path = img_row["filepath"]

        with st.spinner("✨ Gemini Vision is analyzing the image and generating the edit..."):
            try:
                edited_pil = edit_image(source_path, edit_prompt.strip(), "gemini")
                vnum = next_version_num(image_id)
                filepath, thumb = save_pil_image(edited_pil, image_id, vnum)

                insert_version({
                    "id": str(uuid.uuid4()),
                    "image_id": image_id,
                    "version_num": vnum,
                    "filepath": filepath,
                    "thumbnail": thumb,
                    "prompt": edit_prompt.strip(),
                    "provider": "gemini",
                    "model": "gemini-generative",
                    "created_at": timestamp_now(),
                    "notes": None,
                })
                st.success(f"✅ Edit saved as **v{vnum:03d}**!")

                if compare:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original Image**")
                        st.image(source_path, width="stretch")
                    with c2:
                        st.markdown(f"**v{vnum:03d} — Edited Result**")
                        st.image(edited_pil, width="stretch")

                # Download button
                buf = io.BytesIO()
                edited_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Edited Image",
                    data=buf.getvalue(),
                    file_name=f"{img_row['filename'].rsplit('.', 1)[0]}_v{vnum:03d}.png",
                    mime="image/png",
                )

                st.session_state["edit_prompt_input"] = ""
                st.rerun()

            except Exception as e:
                st.error(f"❌ Edit failed: {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if "_page_override" in st.session_state:
    _forced = st.session_state.pop("_page_override")
    if _forced == "✏️ Edit Image":
        page_edit()
    else:
        page_library()
else:
    if page == "📚 Image Library":
        page_library()
    else:
        page_edit()
