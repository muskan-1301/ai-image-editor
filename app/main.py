"""
main.py — AI-Powered Image Editing Platform
Week 3: Image Search + Backend Enhancements

New in Week 3:
  - Dedicated 🔍 Search page with relevance score bars
  - Bulk Re-Index: generate embeddings for all images at once
  - Library Stats dashboard (images, indexed, edits, searches)
  - Enhanced DB: tags, dimensions, file size, search history
  - Error boundaries with friendly messages
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
    log_search,
    get_all_images,
    get_images_without_embeddings,
    get_image,
    get_versions,
    next_version_num,
    delete_image,
    get_search_history,
    get_library_stats,
    update_image_tags,
)
from image_utils import (
    save_uploaded_image,
    save_pil_image,
    timestamp_now,
    human_filesize,
)
from ai_clients import caption_image, edit_image, embed_text, cosine_similarity

# ── Initialise DB (with Week 3 migration) ────────────────────────────────────
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
section[data-testid="stSidebar"] { background: #1a1d27; }

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #c9d1e9;
    border-left: 3px solid #7c6af0; padding-left: 10px;
    margin: 16px 0 10px;
}
.img-caption {
    font-size: 0.75rem; color: #9fa8c0; margin-top: 4px;
    overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.stat-card {
    background: #1e2130; border-radius: 10px; padding: 14px;
    text-align: center; margin-bottom: 8px;
}
.stat-num { font-size: 2rem; font-weight: 700; color: #7c6af0; }
.stat-label { font-size: 0.75rem; color: #9fa8c0; margin-top: 2px; }
.score-bar-wrap { background: #1e2130; border-radius: 6px; height: 8px; margin: 4px 0; }
.score-bar { background: linear-gradient(90deg, #7c6af0, #a78bfa); border-radius: 6px; height: 8px; }
.search-result-card {
    background: #1e2130; border-radius: 12px; padding: 12px;
    margin-bottom: 10px; border: 1px solid #2d3148;
}
.match-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    background: #312e81; color: #c4b5fd;
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
    st.markdown(f"{'✅' if gem_ok else '❌'} **Gemini AI Active**", unsafe_allow_html=True)

    if not gem_ok:
        k = st.text_input("Enter Gemini API key", type="password")
        if k:
            os.environ["GEMINI_API_KEY"] = k
            st.rerun()

    st.divider()

    # Library stats in sidebar
    try:
        stats = get_library_stats()
        st.markdown("### 📊 Library Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-num'>{stats['total_images']}</div>
                <div class='stat-label'>Images</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-num'>{stats['total_edits']}</div>
                <div class='stat-label'>Edits</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-num'>{stats['indexed_images']}</div>
                <div class='stat-label'>Indexed</div></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-num'>{stats['total_searches']}</div>
                <div class='stat-label'>Searches</div></div>""", unsafe_allow_html=True)
    except Exception:
        pass

    st.divider()
    st.markdown("### 📂 Navigate")
    page = st.radio(
        "Go to",
        ["📚 Image Library", "🔍 Search", "✏️ Edit Image"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Storage:\n`{ROOT / 'data' / 'images'}`")


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
            auto_caption = st.checkbox("Auto-caption & index with Gemini", value=True)

        if up and st.button("Upload & Save", type="primary"):
            with st.spinner("Saving image..."):
                try:
                    img_id, filepath, thumb, metadata = save_uploaded_image(up)
                    caption, embedding_json = "", ""

                    if auto_caption:
                        with st.spinner("🤖 Gemini Vision is analyzing your image..."):
                            try:
                                caption = caption_image(filepath, "gemini")
                                emb = embed_text(caption, "gemini")
                                embedding_json = json.dumps(emb)
                            except Exception as e:
                                st.warning(f"⚠️ Auto-caption skipped: {e}")

                    insert_image({
                        "id": img_id,
                        "filename": up.name,
                        "filepath": filepath,
                        "thumbnail": thumb,
                        "caption": caption,
                        "embedding": embedding_json,
                        "uploaded_at": timestamp_now(),
                        "provider": "gemini" if caption else None,
                        "tags": "",
                        "width": metadata.get("width"),
                        "height": metadata.get("height"),
                        "file_size": metadata.get("file_size"),
                    })
                    st.success(f"✅ Uploaded **{up.name}**")
                    if caption:
                        st.info(f"🤖 **Gemini Caption:** {caption}")
                    if metadata.get("width"):
                        st.caption(f"📐 {metadata['width']}×{metadata['height']} · {human_filesize(metadata.get('file_size', 0))}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")

    # ── Bulk Re-Index ─────────────────────────────────────────────────────────
    unindexed = get_images_without_embeddings()
    if unindexed:
        st.warning(f"⚠️ **{len(unindexed)} image(s)** are not indexed yet — they won't appear in search results.")
        if st.button(f"🔄 Bulk Re-Index {len(unindexed)} Image(s) with Gemini", type="primary"):
            progress = st.progress(0, text="Starting...")
            success_count = 0
            for i, img in enumerate(unindexed):
                try:
                    progress.progress((i) / len(unindexed), text=f"Indexing {img['filename']}...")
                    cap = caption_image(img["filepath"], "gemini")
                    emb = embed_text(cap, "gemini")
                    update_image_caption(img["id"], cap, json.dumps(emb), "gemini")
                    success_count += 1
                except Exception as e:
                    st.warning(f"⚠️ Skipped {img['filename']}: {e}")
            progress.empty()
            st.success(f"✅ Successfully indexed {success_count}/{len(unindexed)} images!")
            st.rerun()

    st.divider()

    # ── Image Grid ────────────────────────────────────────────────────────────
    all_images = get_all_images()
    if not all_images:
        st.info("No images yet. Upload one above! 👆")
        return

    st.markdown(f"<div class='section-header'>🖼️ {len(all_images)} Image(s)</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, img_row in enumerate(all_images):
        with cols[i % 4]:
            thumb = img_row["thumbnail"]
            if thumb and Path(thumb).exists():
                st.image(thumb, width="stretch")
            elif Path(img_row["filepath"]).exists():
                st.image(img_row["filepath"], width="stretch")

            caption_text = img_row["caption"] or "*(not captioned)*"
            st.markdown(f"<div class='img-caption'>{caption_text[:100]}</div>", unsafe_allow_html=True)

            # Metadata badges
            meta_parts = [f"📅 {img_row['uploaded_at'][:10]}"]
            if img_row["width"]:
                meta_parts.append(f"📐 {img_row['width']}×{img_row['height']}")
            if img_row["file_size"]:
                meta_parts.append(human_filesize(img_row["file_size"]))
            indexed_icon = "🟢" if img_row["embedding"] else "🔴"
            meta_parts.append(f"{indexed_icon} {'Indexed' if img_row['embedding'] else 'Not indexed'}")
            st.caption(" · ".join(meta_parts))

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
                    for v in get_versions(img_row["id"]):
                        for vp in [v["filepath"], v["thumbnail"]]:
                            if vp and Path(vp).exists():
                                Path(vp).unlink(missing_ok=True)
                    delete_image(img_row["id"])
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEARCH  (New in Week 3)
# ══════════════════════════════════════════════════════════════════════════════

def page_search():
    st.header("🔍 Natural Language Image Search")
    st.caption("Search your image library using plain English — powered by Gemini semantic embeddings.")

    all_images = get_all_images()
    indexed = [img for img in all_images if img["embedding"]]

    # Index coverage banner
    total = len(all_images)
    idx_count = len(indexed)
    if total == 0:
        st.info("No images in your library yet. Upload some in the 📚 Image Library page.")
        return

    cover_pct = int((idx_count / total) * 100) if total > 0 else 0
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Images", total)
    col_b.metric("Search-Indexed", idx_count)
    col_c.metric("Coverage", f"{cover_pct}%")

    if idx_count < total:
        st.warning(f"⚠️ {total - idx_count} image(s) are not indexed. Go to **📚 Image Library → Bulk Re-Index** to include them in search.")

    st.divider()

    # Search bar
    search_q = st.text_input(
        "🔍 Describe what you're looking for",
        placeholder='e.g. "sunset with orange sky", "water filtration system", "people smiling outdoors"...',
        key="search_query",
    )

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        threshold = st.slider("Minimum match score", min_value=0, max_value=100,
                              value=20, step=5, format="%d%%",
                              help="Only show results with at least this similarity score")
    with col_s2:
        max_results = st.selectbox("Max results", [5, 10, 20, "All"], index=1)
    with col_s3:
        do_search = st.button("🔍 Search", type="primary", width="stretch",
                              disabled=not search_q.strip())

    # Search execution
    if search_q.strip() and do_search:
        with st.spinner("🤖 Gemini is computing semantic similarity..."):
            try:
                q_emb = embed_text(search_q.strip(), "gemini")
                scored = []
                for img in indexed:
                    try:
                        img_emb = json.loads(img["embedding"])
                        score = cosine_similarity(q_emb, img_emb)
                        scored.append((score, img))
                    except Exception:
                        pass

                scored.sort(key=lambda x: x[0], reverse=True)
                thresh_float = threshold / 100.0
                results = [(s, img) for s, img in scored if s >= thresh_float]

                if max_results != "All":
                    results = results[:int(max_results)]

                # Log search
                log_search(
                    str(uuid.uuid4()),
                    search_q.strip(),
                    len(results),
                    scored[0][0] if scored else 0.0,
                    timestamp_now(),
                )

                st.markdown(f"<div class='section-header'>📋 {len(results)} Result(s) for \"{search_q}\"</div>",
                            unsafe_allow_html=True)

                if not results:
                    st.warning("No images matched that description above the threshold. Try lowering the minimum match score or rephrasing.")
                else:
                    for score, img_row in results:
                        pct = int(score * 100)
                        bar_width = max(4, pct)

                        with st.container():
                            st.markdown("<div class='search-result-card'>", unsafe_allow_html=True)
                            r_col1, r_col2 = st.columns([1, 3])

                            with r_col1:
                                thumb = img_row["thumbnail"]
                                if thumb and Path(thumb).exists():
                                    st.image(thumb, width="stretch")
                                elif Path(img_row["filepath"]).exists():
                                    st.image(img_row["filepath"], width="stretch")

                            with r_col2:
                                st.markdown(
                                    f"**{img_row['filename']}**  "
                                    f"<span class='match-badge'>Match: {pct}%</span>",
                                    unsafe_allow_html=True,
                                )
                                # Score bar
                                st.markdown(
                                    f"<div class='score-bar-wrap'>"
                                    f"<div class='score-bar' style='width:{bar_width}%'></div></div>",
                                    unsafe_allow_html=True,
                                )
                                caption_text = img_row["caption"] or "No caption"
                                st.markdown(f"_{caption_text}_")

                                meta_parts = [f"📅 {img_row['uploaded_at'][:10]}"]
                                if img_row["width"]:
                                    meta_parts.append(f"📐 {img_row['width']}×{img_row['height']}")
                                if img_row["file_size"]:
                                    meta_parts.append(human_filesize(img_row["file_size"]))
                                st.caption(" · ".join(meta_parts))

                                bcol1, bcol2 = st.columns([1, 1])
                                with bcol1:
                                    if st.button("✏️ Edit this image", key=f"srch_edit_{img_row['id']}", width="stretch"):
                                        st.session_state["selected_image_id"] = img_row["id"]
                                        st.session_state["_page_override"] = "✏️ Edit Image"
                                        st.rerun()
                                with bcol2:
                                    versions = get_versions(img_row["id"])
                                    st.caption(f"🕒 {len(versions)} version(s)")
                            st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                st.exception(e)

    # Search History
    st.divider()
    with st.expander("📜 Recent Search History", expanded=False):
        history = get_search_history(10)
        if not history:
            st.caption("No searches yet.")
        else:
            for h in history:
                col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
                with col_h1:
                    st.markdown(f"🔍 `{h['query']}`")
                with col_h2:
                    st.caption(f"{h['result_count']} results")
                with col_h3:
                    st.caption(h["searched_at"][11:16] + " UTC")


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
            st.caption("📌 Viewing: **Original** — no edits applied yet")
        else:
            ver_choice = st.select_slider("Version", options=ver_labels, value=ver_labels[-1])

        active_path = img_row["filepath"] if ver_choice == "Original" else \
            next(v for v in versions if v["version_num"] == int(ver_choice[1:]))["filepath"]

        if Path(active_path).exists():
            st.image(active_path, width="stretch", caption=ver_choice)
        else:
            st.error("⚠️ Image file not found on disk.")

        # Metadata
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            if img_row["width"]:
                st.caption(f"📐 {img_row['width']} × {img_row['height']} px")
        with meta_col2:
            if img_row["file_size"]:
                st.caption(f"💾 {human_filesize(img_row['file_size'])}")

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
                    st.error(f"❌ Captioning error: {e}")

    with right:
        st.markdown("<div class='section-header'>🕒 Version History</div>", unsafe_allow_html=True)

        if not versions:
            st.info("No edits yet. Apply an edit below to create version v001.")
        else:
            st.caption(f"**{len(versions)} version(s)** saved")
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
                        st.caption(f"_{v['prompt'][:60] if v['prompt'] else ''}..._")
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

    # ── Edit Panel ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🛠️ Apply an Edit</div>", unsafe_allow_html=True)

    if "edit_prompt_input" not in st.session_state:
        st.session_state["edit_prompt_input"] = ""

    st.markdown("**Quick Presets** *(click any to load into the box)*:")
    preset_cols = st.columns(4)
    for idx, (label, prompt_text) in enumerate(PRESET_EDITS):
        with preset_cols[idx % 4]:
            st.button(label, width="stretch", key=f"preset_{idx}",
                      on_click=set_prompt_preset, args=(prompt_text,))

    edit_prompt = st.text_area(
        "✍️ Natural Language Edit Instruction",
        key="edit_prompt_input",
        placeholder="e.g. Add a calm blue ocean in the background, change the sky to purple, make it oil painting style...",
        height=85,
    )

    apply_col, preview_col = st.columns([1, 1])
    with apply_col:
        apply_edit = st.button("🚀 Apply Edit", type="primary", width="stretch",
                               disabled=not edit_prompt.strip())
    with preview_col:
        compare = st.checkbox("Side-by-side comparison after edit", value=True)

    if apply_edit and edit_prompt.strip():
        with st.spinner("✨ Gemini Vision is analyzing and generating the edit..."):
            try:
                edited_pil = edit_image(img_row["filepath"], edit_prompt.strip(), "gemini")
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
                        st.image(img_row["filepath"], width="stretch")
                    with c2:
                        st.markdown(f"**v{vnum:03d} — Edited Result**")
                        st.image(edited_pil, width="stretch")

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
                st.error(f"❌ Edit failed: {str(e)}")
                with st.expander("🔍 Error details"):
                    st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if "_page_override" in st.session_state:
    _forced = st.session_state.pop("_page_override")
    if _forced == "✏️ Edit Image":
        page_edit()
    elif _forced == "🔍 Search":
        page_search()
    else:
        page_library()
else:
    if page == "📚 Image Library":
        page_library()
    elif page == "🔍 Search":
        page_search()
    else:
        page_edit()
