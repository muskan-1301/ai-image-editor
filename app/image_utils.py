"""
image_utils.py — File I/O helpers (save, thumbnail, UUID paths).
Enhanced for Week 3 with metadata extraction.
"""

from __future__ import annotations
import io
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

DATA_ROOT = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_ROOT / "images"
VERSIONS_DIR = DATA_ROOT / "versions"
THUMBS_DIR = DATA_ROOT / "thumbnails"
THUMB_SIZE = (256, 256)


def _ensure_dirs() -> None:
    for d in (IMAGES_DIR, VERSIONS_DIR, THUMBS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_uploaded_image(uploaded_file) -> tuple[str, str, str, dict]:
    """
    Persist a Streamlit UploadedFile.
    Returns (image_id, filepath, thumbnail_path, metadata_dict).
    """
    _ensure_dirs()
    image_id = str(uuid.uuid4())
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    dest = IMAGES_DIR / f"{image_id}{suffix}"

    img_bytes = uploaded_file.read()
    dest.write_bytes(img_bytes)

    # Extract metadata
    try:
        img = Image.open(str(dest))
        width, height = img.size
    except Exception:
        width, height = None, None

    file_size = dest.stat().st_size

    thumb_path = make_thumbnail(str(dest), image_id, prefix="orig")
    metadata = {"width": width, "height": height, "file_size": file_size}
    return image_id, str(dest), thumb_path, metadata


def save_pil_image(img: Image.Image, image_id: str, version_num: int) -> tuple[str, str]:
    """Save an edited PIL image as a new version."""
    _ensure_dirs()
    dest = VERSIONS_DIR / f"{image_id}_v{version_num:03d}.png"
    img.save(str(dest), format="PNG")
    thumb_path = make_thumbnail(str(dest), image_id, prefix=f"v{version_num:03d}")
    return str(dest), thumb_path


def make_thumbnail(filepath: str, image_id: str, prefix: str = "thumb") -> str:
    _ensure_dirs()
    img = Image.open(filepath).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    thumb_path = THUMBS_DIR / f"{prefix}_{image_id}.jpg"
    img.save(str(thumb_path), format="JPEG", quality=80)
    return str(thumb_path)


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def timestamp_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def human_filesize(size_bytes: int) -> str:
    if not size_bytes:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes/1024:.1f} KB"
    return f"{size_bytes/1024**2:.1f} MB"
