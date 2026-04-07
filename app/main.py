from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import os

from app.analyzer import analyze_form
from app.analyzerV2 import analyze_form_v2
app = FastAPI(
    title="OCR Form API",
    description="API d'analyse de formulaires médicaux par OCR",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou mets ton domaine frontend exact
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "API OCR opérationnelle"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    debug: bool = Query(default=False)
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")

    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés: {sorted(ALLOWED_EXTENSIONS)}"
        )

    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = analyze_form(str(temp_path), debug=debug)
        print(f"=== RESULTAT OCR POUR {image.filename} ===")
        print(result)
        return JSONResponse(content={
            "success": True,
            "filename": image.filename,
            "result": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass
        try:
            image.file.close()
        except Exception:
            pass
@app.post("/analyze-v2")
async def analyze_image_v2(
    image: UploadFile = File(...),
    debug: bool = Query(default=False)
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")

    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés: {sorted(ALLOWED_EXTENSIONS)}"
        )

    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = UPLOAD_DIR / temp_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        result = analyze_form_v2(str(temp_path), debug=debug)

        return JSONResponse(content={
            "success": True,
            "filename": image.filename,
            "result": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass
        try:
            image.file.close()
        except Exception:
            pass
