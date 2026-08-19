"""Named reference images for /gen_model's @name syntax -- drop a picture in
img-refs/ and refer to it as @filename (extension optional) instead of typing
a full path. Also backs the @-completion popup in ui/app.py.
"""
from pathlib import Path

from gen3d.pipeline import FLOWAI_ROOT

IMG_REFS_DIR = FLOWAI_ROOT / "img-refs"
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_refs() -> list[Path]:
    if not IMG_REFS_DIR.is_dir():
        return []
    return sorted(
        (p for p in IMG_REFS_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS),
        key=lambda p: p.name.lower(),
    )


def resolve_ref(name: str) -> Path | None:
    """`name` without the leading '@'. Tries an exact filename match first,
    then the same stem with each known image extension."""
    if not IMG_REFS_DIR.is_dir():
        return None
    exact = IMG_REFS_DIR / name
    if exact.is_file():
        return exact
    for ext in EXTS:
        candidate = IMG_REFS_DIR / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    return None
