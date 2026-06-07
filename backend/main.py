"""
Road Crack Detection API — FastAPI backend
Production-oriented OpenCV pipeline for smartphone road inspection photos.
"""

import base64
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Road Crack Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Processing parameters (1024px pipeline) --------------------------------
TARGET_MAX_SIDE = 1024

BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

MEDIAN_BLUR_KERNEL = 5

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ADAPTIVE_BLOCK_SIZE = 51
ADAPTIVE_C = 8

MORPH_KERNEL_SHAPE = cv2.MORPH_ELLIPSE
MORPH_KERNEL_SIZE = (5, 5)

MIN_CONTOUR_AREA = 600
MAX_SOLIDITY = 0.45

CONTOUR_DRAW_COLOR = (0, 0, 255)
CONTOUR_DRAW_THICKNESS = 2


def resize_keep_aspect_ratio(img: np.ndarray, target_max_side: int = TARGET_MAX_SIDE) -> np.ndarray:
    """Scale longest side to target_max_side while preserving aspect ratio."""
    height, width = img.shape[:2]
    longest_side = max(height, width)

    if longest_side == 0:
        raise ValueError("Invalid image dimensions")

    if longest_side == target_max_side:
        return img

    scale = target_max_side / longest_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize (max side 1024) and convert to grayscale."""
    resized = resize_keep_aspect_ratio(img, TARGET_MAX_SIDE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, gray


def enhance(gray: np.ndarray) -> np.ndarray:
    """
    Edge-preserving enhancement:
    bilateral filter -> median blur -> CLAHE.
    """
    smoothed = cv2.bilateralFilter(
        gray,
        BILATERAL_DIAMETER,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE,
    )
    denoised = cv2.medianBlur(smoothed, MEDIAN_BLUR_KERNEL)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(denoised)


def segment_crack(enhanced_gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold tuned for high-resolution local lighting variation."""
    return cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE,
        C=ADAPTIVE_C,
    )


def apply_morphology(crack_mask: np.ndarray) -> np.ndarray:
    """MORPH_CLOSE (bridge gaps) then MORPH_OPEN (remove speckle noise)."""
    kernel = cv2.getStructuringElement(MORPH_KERNEL_SHAPE, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


def contour_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    if area <= 0:
        return 1.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 1.0

    return float(area / hull_area)


def is_valid_crack_contour(contour: np.ndarray) -> bool:
    area = cv2.contourArea(contour)
    if area <= MIN_CONTOUR_AREA:
        return False

    solidity = contour_solidity(contour)
    if solidity > MAX_SOLIDITY:
        return False

    return True


def extract_valid_contours(cleaned_mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [contour for contour in contours if is_valid_crack_contour(contour)]


def annotate_cracks(original_img: np.ndarray, valid_contours: list[np.ndarray]) -> np.ndarray:
    """Draw exact crack paths with tight red contours."""
    annotated = original_img.copy()
    if valid_contours:
        cv2.drawContours(
            annotated,
            valid_contours,
            contourIdx=-1,
            color=CONTOUR_DRAW_COLOR,
            thickness=CONTOUR_DRAW_THICKNESS,
        )
    return annotated


def compute_severity(valid_contours: list[np.ndarray], image_shape: tuple[int, ...]) -> float:
    """
    Severity % = sum(convex hull areas of valid contours) / total image area.
    Hull area approximates the overall damaged zone (e.g. alligator cracking).
    """
    total_area = image_shape[0] * image_shape[1]
    if total_area <= 0 or not valid_contours:
        return 0.0

    crack_area = 0.0
    for contour in valid_contours:
        hull = cv2.convexHull(contour)
        crack_area += cv2.contourArea(hull)

    crack_area = min(crack_area, total_area)
    severity_percentage = (crack_area / total_area) * 100
    return float(severity_percentage)


def postprocess(
    crack_mask: np.ndarray, original_img: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
    cleaned_mask = apply_morphology(crack_mask)
    valid_contours = extract_valid_contours(cleaned_mask)
    annotated = annotate_cracks(original_img, valid_contours)
    return cleaned_mask, annotated, valid_contours, len(valid_contours)


def encode_image_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode result image")
    return base64.b64encode(buffer).decode("utf-8")


def analyze_road_image(image_bytes: bytes) -> dict[str, Any]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    resized, gray = preprocess(img)
    enhanced_gray = enhance(gray)
    crack_mask = segment_crack(enhanced_gray)
    cleaned_mask, annotated, valid_contours, blob_count = postprocess(crack_mask, resized)
    severity = compute_severity(valid_contours, cleaned_mask.shape)

    return {
        "status": "success",
        "severity": round(severity, 4),
        "blob_count": blob_count,
        "image_base64": encode_image_base64(annotated),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Road Crack Detection API", "analyze": "POST /analyze"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        return analyze_road_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
