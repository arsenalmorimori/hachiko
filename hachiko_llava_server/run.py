"""
llava_server.py  –  Fast offline LLaVA server using Ollama + FastAPI
--------------------------------------------------------------------
Requirements:
    pip install fastapi uvicorn

Run:
    python llava_server.py

The server listens on 0.0.0.0:8000 so your phone can reach it over hotspot.
Set the IP in MainPage.xaml.cs → ServerUrl to match your PC's hotspot IP.
Find it via: ipconfig  (Windows) | ip addr (Linux)
"""

import base64
import subprocess
import tempfile
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="LLaVA Offline Server")

OLLAMA_MODEL = "llava"

PROMPT = (
    "Describe this image. "
    "If there is text, read it — the language may be Filipino or English. "
    "If there are people, describe what they are doing, their emotion, pose, and apparent gender. "
    "If there is an expiry date or label, analyze it. "
    "If it is a landscape or scenery, describe it in detail including colors, like a work of art. "
    "Only mention what is applicable; skip any instruction that does not apply."
)


class ImageRequest(BaseModel):
    image: str       # base64-encoded image bytes
    mime_type: str = "image/jpeg"   # "image/jpeg" or "image/png"


@app.post("/analyze")
async def analyze(req: ImageRequest) -> JSONResponse:
    # ── Decode base64 → temp file ─────────────────────────────────────────
    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    ext = ".png" if req.mime_type == "image/png" else ".jpg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        # ── Call Ollama LLaVA ─────────────────────────────────────────────
        # Pass the image path directly after the prompt using the
        # "ollama run llava '<prompt> <path>'" syntax.
        # This is the fastest single-process approach — no HTTP overhead to Ollama.
        cmd = [
            "ollama", "run", OLLAMA_MODEL,
            f"{PROMPT} {tmp_path}"
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300   # 5-minute safety timeout
        )

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama error: {proc.stderr.strip() or 'unknown error'}"
            )

        result = proc.stdout.strip()
        return JSONResponse({"result": result})

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="LLaVA timed out (>5 min)")
    finally:
        # Always clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/health")
def health():
    """Quick ping — MAUI app can check this on startup."""
    return {"status": "ok", "model": OLLAMA_MODEL}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🟣 LLaVA server starting on http://0.0.0.0:{port}")
    print(f"   Model  : {OLLAMA_MODEL}")
    print(f"   Tip    : Set ServerUrl in MainPage.xaml.cs to your PC hotspot IP")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")