"""Named model references for /anim's @name syntax -- refer to a previously
generated model in generated/models/ by name instead of the last-generated
one or a full path. Also backs the @-completion popup in ui/app.py.
"""
from gen3d.pipeline import generated_models_dir


def list_models() -> list:
    models_dir = generated_models_dir()
    if not models_dir.is_dir():
        return []
    return sorted(
        (p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() == ".glb"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )


def resolve_model(name: str):
    """`name` without the leading '@'. Tries an exact filename match first,
    then the same stem with a .glb extension appended."""
    models_dir = generated_models_dir()
    if not models_dir.is_dir():
        return None
    exact = models_dir / name
    if exact.is_file():
        return exact
    candidate = models_dir / f"{name}.glb"
    if candidate.is_file():
        return candidate
    return None
