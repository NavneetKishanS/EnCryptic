from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort
import cv2, hashlib, os, time
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = Path(os.getenv("ENCRYPTIC_MODEL_PATH", "model.onnx"))

# Training class order (0..11) exactly as in your YAML
TRAINING_CLASS_ORDER = [
    "biker",
    "car",
    "pedestrian",
    "trafficLight",
    "trafficLight-Green",
    "trafficLight-GreenLeft",
    "trafficLight-Red",
    "trafficLight-RedLeft",
    "trafficLight-Yellow",
    "trafficLight-YellowLeft",
    "truck",
    "obstacles",
]

API_BASELINE_STEPS = [
    {"step": 1, "title": "Got dataset from cars", "detail": "Using pre-trained EnCryptic detector."},
    {"step": 2, "title": "Detecting objects in the image", "detail": ""},
    {"step": 3, "title": "Creating list of detected objects", "detail": ""},
    {"step": 4, "title": "Assigning a random target", "detail": ""},
    {"step": 5, "title": "Classifying images based on the target", "detail": "Demo shows per-image filter."},
    {"step": 6, "title": "Generating keys from classified images", "detail": "SHA-256 over detections."},
    {"step": 7, "title": "Making the master-key", "detail": "Stable across same visual scene."},
    {"step": 8, "title": "Here's the master key", "detail": ""}
]

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="EnCryptic Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Globals
# -----------------------------
sess = None
input_name = None
output_name = None
model_loaded = False
load_error = None
providers_used = None
input_layout = None   # "NCHW" or "NHWC"
input_shape = None

def get_session():
    global sess, input_name, output_name, model_loaded, load_error
    global providers_used, input_layout, input_shape

    if sess is None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

            available = ort.get_available_providers()
            providers_used = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )

            sess = ort.InferenceSession(str(MODEL_PATH), providers=providers_used)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name

            input_shape = sess.get_inputs()[0].shape
            layout = "NCHW"
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
                if input_shape[1] == 3:
                    layout = "NCHW"        # (1,3,640,640)
                elif input_shape[3] == 3:
                    layout = "NHWC"        # (1,640,640,3)
            input_layout = layout

            model_loaded = True
            load_error = None
            print(f"✅ Model loaded from {MODEL_PATH}", flush=True)
            print(f"✅ Providers: {providers_used}", flush=True)
            print(f"✅ Input shape: {input_shape} | layout: {input_layout}", flush=True)

        except Exception as e:
            model_loaded = False
            load_error = str(e)
            print("❌ Model load failed:", e, flush=True)
            raise

    return sess, input_name, output_name


@app.on_event("startup")
def warmup():
    global model_loaded, load_error
    try:
        get_session()
    except Exception as e:
        model_loaded = False
        load_error = str(e)


@app.get("/health")
def health():
    global model_loaded, load_error
    if not model_loaded:
        try:
            get_session()
        except Exception as e:
            load_error = str(e)
    return {
        "ok": True,
        "model_path": str(MODEL_PATH),
        "model_file_exists": MODEL_PATH.exists(),
        "model_loaded": model_loaded,
        "providers_used": providers_used,
        "load_error": load_error,
        "input_shape": input_shape,
        "input_layout": input_layout,
    }


# -----------------------------
# Notebook-identical letterbox
# -----------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape

    scale = min(new_w / w, new_h / h)
    rw, rh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)

    padded = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    pad_x = (new_w - rw) // 2
    pad_y = (new_h - rh) // 2

    padded[pad_y:pad_y+rh, pad_x:pad_x+rw] = resized
    return padded, scale, pad_x, pad_y


# -----------------------------
# Decode EXACTLY like notebook
# Output: cx,cy,w,h,conf,cls_id in 640 letterbox pixels
# -----------------------------
def decode_preds_letterbox(preds, scale, pad_x, pad_y, orig_w, orig_h, conf_thres=0.25):
    dets = []

    preds = np.squeeze(preds)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.ndim != 2 or preds.shape[1] < 6:
        return dets

    for (cx, cy, w, h, conf, cls_id) in preds:
        conf = float(conf)
        if conf < conf_thres:
            continue

        # center -> corners in LETTERBOX space
        x1 = float(cx - w / 2)
        y1 = float(cy - h / 2)
        x2 = float(cx + w / 2)
        y2 = float(cy + h / 2)

        # UNDO padding
        x1 -= pad_x
        x2 -= pad_x
        y1 -= pad_y
        y2 -= pad_y

        # UNDO scale (critical)
        x1 /= scale
        x2 /= scale
        y1 /= scale
        y2 /= scale

        # clip to original image
        x1 = max(0.0, min(x1, orig_w - 1))
        y1 = max(0.0, min(y1, orig_h - 1))
        x2 = max(0.0, min(x2, orig_w - 1))
        y2 = max(0.0, min(y2, orig_h - 1))

        cid = int(cls_id)
        cname = TRAINING_CLASS_ORDER[cid] if cid < len(TRAINING_CLASS_ORDER) else str(cid)

        dets.append({
            "class_id": cid,
            "class_name": cname,
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2],
        })

    return dets


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    t0 = time.time()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    try:
        sess, input_name, output_name = get_session()
    except Exception:
        return JSONResponse({"error": "Model not loaded", "detail": load_error}, status_code=500)

    # match notebook: RGB + remember original size
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # notebook letterbox to 640
    img_lb, scale, pad_x, pad_y = letterbox(img_rgb, (640, 640))

    # notebook input: NHWC float32 /255
    blob = img_lb.astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # (1,640,640,3)

    # if model expects NCHW, transpose
    if input_layout == "NCHW":
        blob = np.transpose(blob, (0, 3, 1, 2))  # (1,3,640,640)

    blob = np.ascontiguousarray(blob)

    try:
        preds = sess.run([output_name], {input_name: blob})[0]
    except Exception as e:
        return JSONResponse({"error": "Inference failed", "detail": str(e)}, status_code=500)

    dets = decode_preds_letterbox(
        preds, scale, pad_x, pad_y, orig_w, orig_h, conf_thres=0.25
    )

    unique = sorted({d["class_name"] for d in dets})
    target = unique[np.random.randint(0, len(unique))] if unique else None

    # Stable tokenization -> master key
    tokens = []
    for d in sorted(dets, key=lambda z: (z["class_id"], z["bbox_xyxy"][0], z["bbox_xyxy"][1])):
        cls = d["class_name"]
        x1, y1, x2, y2 = d["bbox_xyxy"]
        q = d["confidence"]
        tokens.append(f"{cls}:{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}:{q:.3f}")

    blob_key = "|".join(tokens).encode()
    master_key = hashlib.sha256(blob_key).hexdigest()

    # Timeline details
    timeline = [dict(s) for s in API_BASELINE_STEPS]
    timeline[1]["detail"] = f"Found {len(dets)} objects."
    timeline[2]["detail"] = ", ".join(unique) if unique else "None."
    timeline[3]["detail"] = target or "No target (nothing detected)."
    timeline[7]["detail"] = master_key[:32] + "…"

    return {
        "detections": dets,
        "unique_objects": unique,
        "target_object": target,
        "master_key": master_key,
        "timeline": timeline,
        "latency_ms": int((time.time() - t0) * 1000),
        "input_layout_used": input_layout
    }
