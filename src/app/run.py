"""Launch ARGO SENTINEL (frontend + FastAPI backend)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    basemap = ROOT / "static" / "basemap.png"
    if not basemap.exists():
        src = ROOT.parent / "outputs" / "maps" / "indian_ocean_grid.png"
        if not src.exists():
            src = ROOT.parent / "argo_sentinel" / "static" / "basemap.png"
        if src.exists():
            basemap.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(src, basemap)
            print(f"Copied basemap from {src}")
        else:
            print("Generating basemap…")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_basemap.py")], check=False)

    from backend.port_util import resolve_port

    port = resolve_port(8050)

    import uvicorn

    print(f"\n  ARGO SENTINEL starting at http://127.0.0.1:{port}")
    print("  First map load uses cached grid stats; full refresh runs in background.\n")

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
