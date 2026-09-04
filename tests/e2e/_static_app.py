from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def create_static_app() -> FastAPI:
    """Serve layout fixtures with the production HTML URL substitutions."""
    static_dir = Path(__file__).resolve().parents[2] / "server" / "static"
    app = FastAPI()

    def render(relative: str) -> HTMLResponse:
        html = (static_dir / relative).read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__AW_VERSION__", "test").replace("__AW_BASE__", ""))

    @app.get("/")
    def index() -> HTMLResponse:
        return render("index.html")

    @app.get("/workspace/")
    def workspace() -> HTMLResponse:
        return render("workspace/index.html")

    app.mount("/_v/test", StaticFiles(directory=static_dir), name="versioned")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
