from pathlib import Path
import sys


def test_backend_lexi_exports_resolve():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "backend"))

    import lexi

    for name in ("config", "model_loader", "memory", "persona", "prompt_templates"):
        assert hasattr(lexi, name)
