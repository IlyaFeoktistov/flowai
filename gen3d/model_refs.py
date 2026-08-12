"""Named model references for /anim's @name syntax -- refer to a previously
generated model in generated/models/ by name instead of the last-generated
one or a full path. Also backs the @-completion popup in ui/app.py.
"""
from gen3d.pipeline import GENERATED_MODELS_DIR


def list_models() -> list:
    if not GENERATED_MODELS_DIR.is_dir():
        return []
    return sorted(
        (p for p in GENERATED_MODELS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".glb"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )


def resolve_model(name: str):
    """`name` without the leading '@'. Tries an exact filename match first,
    then the same stem with a .glb extension appended."""
    if not GENERATED_MODELS_DIR.is_dir():
        return None
    exact = GENERATED_MODELS_DIR / name
    if exact.is_file():
        return exact
    candidate = GENERATED_MODELS_DIR / f"{name}.glb"
    if candidate.is_file():
        return candidate
    return None
