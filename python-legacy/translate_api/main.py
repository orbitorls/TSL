from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1] / "src"
for p in (str(SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from translate_api.deps import init_dependencies  # noqa: E402
from translate_api.routes import models as models_routes  # noqa: E402
from translate_api.store import SessionStore  # noqa: E402
from translate_api.ws import inference as ws_inference  # noqa: E402

store = SessionStore()
init_dependencies(store, REPO_ROOT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for sid in list(store._sessions.keys()):
        store.delete(sid)


app = FastAPI(title="TSL Translate API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_routes.router)
app.include_router(ws_inference.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
