import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["FLASK_DEBUG"] = "0"
import gc
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image, ExifTags  # type: ignore
from contextlib import nullcontext
try:
    import torch  # type: ignore
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    torch = None
import requests

# Google Maps API Key configuration
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()

# In-memory Geo Caches to prevent rate limiting
_REVERSE_GEO_CACHE = {}
_SEARCH_GEO_CACHE = {}
_GEO_HEADERS = {
    "User-Agent": "RoadGuardAI-DamageSurveillance/1.0 (https://render.com; contact: support@roadguard.ai)",
    "Accept": "application/json"
}

def fetch_reverse_geocode(lat, lon):
    try:
        cache_key = f"{round(float(lat), 4)}_{round(float(lon), 4)}"
        if cache_key in _REVERSE_GEO_CACHE:
            return _REVERSE_GEO_CACHE[cache_key]

        # Use Google Maps Geocoding API if key is present (instant & accurate)
        if GOOGLE_MAPS_API_KEY:
            try:
                g_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={GOOGLE_MAPS_API_KEY}"
                g_resp = requests.get(g_url, timeout=1.8)
                if g_resp.status_code == 200:
                    g_data = g_resp.json()
                    if g_data.get("results"):
                        formatted_addr = g_data["results"][0].get("formatted_address", "")
                        res = {"success": True, "address": formatted_addr}
                        _REVERSE_GEO_CACHE[cache_key] = res
                        return res
            except Exception:
                pass

        # Fallback to OpenStreetMap with tight 1.5s timeout
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
        resp = requests.get(url, headers=_GEO_HEADERS, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            addr = data.get("address", {})
            road = addr.get("road") or addr.get("pedestrian") or addr.get("suburb") or addr.get("neighbourhood") or ""
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("state_district") or ""
            state = addr.get("state", "")
            
            clean_addr = f"{road}, {city}".strip(", ") if road or city else data.get("display_name", f"{lat:.4f}, {lon:.4f}")
            res = {
                "success": True,
                "address": clean_addr,
                "road": road,
                "city": city,
                "state": state,
                "display_name": data.get("display_name", "")
            }
            if len(_REVERSE_GEO_CACHE) > 500:
                _REVERSE_GEO_CACHE.clear()
            _REVERSE_GEO_CACHE[cache_key] = res
            return res
    except Exception as e:
        print(f"Reverse geocode error: {e}")
    return {"success": False, "address": f"{float(lat):.4f}, {float(lon):.4f}"}


def extract_exif_gps(image_path):
    """
    Extracts latitude and longitude from the image's EXIF metadata if present.
    Returns (lat, lon) as floats or (None, None).
    """
    try:
        with Image.open(image_path) as img:
            gps_info = {}
            exif = img.getexif() if hasattr(img, "getexif") else None
            if exif:
                try:
                    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                    for key, val in gps_ifd.items():
                        sub_tag = ExifTags.GPSTAGS.get(key, key)
                        gps_info[sub_tag] = val
                except Exception:
                    pass

            if not gps_info:
                legacy_exif = getattr(img, "_getexif", lambda: None)()
                if legacy_exif:
                    for tag_id, val in legacy_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag_name == "GPSInfo":
                            for t, v in val.items():
                                sub_tag = ExifTags.GPSTAGS.get(t, t)
                                gps_info[sub_tag] = v

            if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
                return None, None

            def to_float(val):
                if isinstance(val, (int, float)):
                    return float(val)
                if hasattr(val, "numerator") and hasattr(val, "denominator"):
                    return float(val.numerator) / float(val.denominator) if val.denominator != 0 else 0.0
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    return float(val[0]) / float(val[1]) if val[1] != 0 else 0.0
                return float(val)

            lat_dms = gps_info["GPSLatitude"]
            lat_deg = to_float(lat_dms[0]) + (to_float(lat_dms[1]) / 60.0) + (to_float(lat_dms[2]) / 3600.0)
            if gps_info.get("GPSLatitudeRef") == "S":
                lat_deg = -lat_deg

            lon_dms = gps_info["GPSLongitude"]
            lon_deg = to_float(lon_dms[0]) + (to_float(lon_dms[1]) / 60.0) + (to_float(lon_dms[2]) / 3600.0)
            if gps_info.get("GPSLongitudeRef") == "W":
                lon_deg = -lon_deg

            return round(lat_deg, 6), round(lon_deg, 6)
    except Exception as e:
        return None, None

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Auto-load .env file if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v:
                        os.environ[k] = v
    except Exception as e:
        print(f"Note: .env reading exception: {e}")

MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "server" / "uploads"
DB_PATH = BASE_DIR / "server" / "road_damage.db"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
CORS(app)

# Database Setup
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS damage_reports (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            latitude REAL,
            longitude REAL,
            image_filename TEXT,
            annotated_filename TEXT,
            damage_type TEXT,
            confidence REAL,
            priority TEXT,
            severity_score INTEGER,
            box_count INTEGER,
            status TEXT DEFAULT 'PENDING',
            officer_notes TEXT DEFAULT '',
            source TEXT DEFAULT 'Mobile Web Cam',
            address TEXT DEFAULT '',
            accuracy REAL DEFAULT 0.0
        )
    """)
    for col, col_type in [("address", "TEXT DEFAULT ''"), ("accuracy", "REAL DEFAULT 0.0")]:
        try:
            cursor.execute(f"ALTER TABLE damage_reports ADD COLUMN {col} {col_type}")
        except Exception:
            pass
    conn.commit()
    conn.close()

init_db()

# Dual AI Ensemble Models:
# Model A: Trained on Indian Potholes, Road Holes & Cavities (potholes.pt)
# Model B: Trained on RDD2022 Alligator, Longitudinal & Transverse Cracks (cracks.pt)
pothole_model = None
crack_model = None
model_source = "Dual Ensemble AI (Indian Potholes & RDD2022 Cracks)"

def load_ai_model():
    global pothole_model, crack_model, model_source
    from ultralytics import YOLO  # type: ignore
    
    pothole_path = MODELS_DIR / "potholes.pt"
    crack_path = MODELS_DIR / "cracks.pt"
    legacy_path = MODELS_DIR / "best.pt"

    if pothole_path.exists():
        print(f"✅ Loading Indian Pothole model: {pothole_path}")
        pothole_model = YOLO(str(pothole_path))
    elif legacy_path.exists():
        pothole_model = YOLO(str(legacy_path))

    if crack_path.exists():
        print(f"✅ Loading RDD2022 Crack model: {crack_path}")
        crack_model = YOLO(str(crack_path))
    elif legacy_path.exists() and pothole_model is None:
        crack_model = YOLO(str(legacy_path))

    if pothole_model is None and crack_model is None:
        pothole_model = YOLO("yolov8n.pt")
        model_source = "Base YOLOv8 (Fallback)"
    else:
        model_source = "Dual Ensemble AI (Potholes, Cavities & Cracks)"

    try:
        if pothole_model:
            pothole_model.to("cpu")
        if crack_model:
            crack_model.to("cpu")
    except Exception:
        pass
    gc.collect()

load_ai_model()

# Priority calculation algorithm
def compute_priority_and_severity(detections, img_width, img_height):
    """
    Computes priority ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW') and severity score (1-10)
    based on damage classes, surface area ratio of damage, and confidence.
    """
    if not detections:
        return "LOW", 1, "None Detected"
    
    # Priority mapping weights by class keyword (matched to RDD2022 damage classes:
    # 0: longitudinal_crack, 1: transverse_crack, 2: alligator_crack, 3: pothole)
    class_weights = {
        "pothole": 5,             # Deep cavities — high vehicle danger
        "alligator_crack": 4,     # Structural fatigue mesh cracking
        "alligator": 4,
        "transverse_crack": 3,    # Cross-lane structural crack
        "transverse": 3,
        "longitudinal_crack": 2,  # Lengthwise joint crack
        "longitudinal": 2,
        "crack": 3,
        "hole": 5,
        "manhole": 4,
        "drain": 3,
    }
    
    total_area_ratio = 0.0
    highest_weight = 1
    dominant_class = detections[0]["class"]
    highest_conf = 0.0
    
    img_area = max(1, img_width * img_height)

    for d in detections:
        c_name = d["class"].lower()
        conf = d["confidence"]
        if conf > highest_conf:
            highest_conf = conf
            dominant_class = d["class"]

        weight = 2  # default
        for k, w in class_weights.items():
            if k in c_name:
                weight = max(weight, w)
        highest_weight = max(highest_weight, weight)

        # calculate box area
        x1, y1, x2, y2 = d["box"]
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        total_area_ratio += (box_area / img_area)

    # Base severity score from 1 to 10
    score = highest_weight * 1.5  # up to 7.5
    
    # Boost by area affected on road
    if total_area_ratio > 0.15:
        score += 2.5
    elif total_area_ratio > 0.05:
        score += 1.5
    elif total_area_ratio > 0.02:
        score += 0.8
        
    # Multiple damages on same road segment
    if len(detections) >= 3:
        score += 1.2
    elif len(detections) >= 2:
        score += 0.6

    score = int(min(10, max(1, round(score))))

    # Priority tier
    if score >= 8:
        priority = "CRITICAL"
    elif score >= 6:
        priority = "HIGH"
    elif score >= 4:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return priority, score, dominant_class

# Google Maps API Key configuration
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()

# Routes
@app.route("/")
def dashboard_view():
    """Serves the Officer Portal Dashboard"""
    return render_template(
        "dashboard.html",
        model_source=model_source,
        google_maps_api_key=GOOGLE_MAPS_API_KEY
    )

@app.route("/mobile")
def mobile_view():
    """Serves the Mobile Camera & GPS capture interface"""
    return render_template(
        "mobile.html",
        google_maps_api_key=GOOGLE_MAPS_API_KEY
    )

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOADS_DIR), filename)

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/api/detect", methods=["POST"])
def detect_damage():
    """
    Receives image + latitude + longitude from mobile camera,
    runs YOLO detection, annotates image, and stores report.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    lat_val = request.form.get("latitude", type=float)
    lon_val = request.form.get("longitude", type=float)
    accuracy_val = request.form.get("accuracy", type=float, default=0.0)
    address_val = request.form.get("address", default="")
    source_val = request.form.get("source", "Mobile Web")

    report_id = str(uuid.uuid4())
    raw_filename = f"raw_{report_id}.jpg"
    annotated_filename = f"annotated_{report_id}.jpg"
    
    raw_path = UPLOADS_DIR / raw_filename
    annotated_path = UPLOADS_DIR / annotated_filename
    
    image_file.save(str(raw_path))

    # Coordinate hierarchy:
    # 1. Real device coordinates passed from mobile browser (Live Hardware GPS / Pin)
    # 2. Camera EXIF coordinates
    # 3. Fallback demo coordinates
    if lat_val is not None and lon_val is not None and (abs(lat_val) > 0.001 or abs(lon_val) > 0.001):
        location_source = source_val or "Device GPS"
    else:
        exif_lat, exif_lon = extract_exif_gps(raw_path)
        if exif_lat is not None and exif_lon is not None:
            lat_val = exif_lat
            lon_val = exif_lon
            location_source = "Photo EXIF GPS"
            print(f"📍 Extracted EXIF GPS from image metadata: {lat_val}, {lon_val}")
        else:
            lat_val = round(12.9716 + float(np.random.uniform(-0.02, 0.02)), 6)
            lon_val = round(77.5946 + float(np.random.uniform(-0.02, 0.02)), 6)
            location_source = "Default Coordinates"

    # Read image with OpenCV
    img = cv2.imread(str(raw_path))
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
    
    img_h, img_w = img.shape[:2]

    # Downscale image if larger than 1024px to prevent RAM spikes on cloud free tiers
    max_dim = 1024
    if max(img_h, img_w) > max_dim:
        scale = max_dim / float(max(img_h, img_w))
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_h, img_w = new_h, new_w

    # Run Dual AI Ensemble Inference (Indian Potholes & RDD2022 Cracks)
    global pothole_model, crack_model
    if pothole_model is None and crack_model is None:
        load_ai_model()

    raw_candidates = []

    with (torch.inference_mode() if torch is not None else nullcontext()):
        # 1. Pothole & Cavity Detection
        if pothole_model is not None:
            try:
                p_res = pothole_model.predict(img, conf=0.18, imgsz=384, device="cpu", verbose=False)
                for r in p_res:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        raw_cls = pothole_model.names.get(cls_id, f"Class {cls_id}")
                        raw_candidates.append({
                            "raw_cls": raw_cls,
                            "confidence": round(conf, 3),
                            "box": [x1, y1, x2, y2]
                        })
            except Exception as e:
                print(f"Pothole model predict error: {e}")

        # 2. Crack & Structural Fissure Detection
        if crack_model is not None:
            try:
                c_res = crack_model.predict(img, conf=0.18, imgsz=384, device="cpu", verbose=False)
                for r in c_res:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        raw_cls = crack_model.names.get(cls_id, "crack")
                        if "background" in raw_cls.lower():
                            continue
                        raw_candidates.append({
                            "raw_cls": raw_cls,
                            "confidence": round(conf, 3),
                            "box": [x1, y1, x2, y2]
                        })
            except Exception as e:
                print(f"Crack model predict error: {e}")

    # Non-Maximum Suppression (IoU > 0.45) to merge duplicate boxes
    raw_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    detections = []
    for cand in raw_candidates:
        c_box = cand["box"]
        overlap = False
        for kept in detections:
            k_box = kept["box"]
            ix1 = max(c_box[0], k_box[0])
            iy1 = max(c_box[1], k_box[1])
            ix2 = min(c_box[2], k_box[2])
            iy2 = min(c_box[3], k_box[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            c_area = max(0, c_box[2] - c_box[0]) * max(0, c_box[3] - c_box[1])
            k_area = max(0, k_box[2] - k_box[0]) * max(0, k_box[3] - k_box[1])
            union = c_area + k_area - inter
            if union > 0 and (inter / union) > 0.45:
                overlap = True
                break
        if not overlap:
            raw_cls = cand["raw_cls"]
            cls_name = raw_cls.replace("_", " ").replace("-", " ").strip().title()
            detections.append({
                "class": cls_name,
                "confidence": cand["confidence"],
                "box": cand["box"],
                "raw_cls": raw_cls
            })

    annotated_img = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        conf = d["confidence"]
        cls_name = d["class"]
        c_low = d["raw_cls"].lower()

        # Color coding tailored by damage type (BGR format)
        if "pothole" in c_low or "hole" in c_low:
            color = (0, 0, 255)       # Red: Severe cavity
        elif "alligator" in c_low:
            color = (0, 100, 255)     # Deep Orange: Severe structural web crack
        elif "transverse" in c_low:
            color = (0, 180, 255)     # Amber: Cross-lane structural crack
        elif "longitudinal" in c_low:
            color = (255, 170, 0)     # Cyan/Teal: Longitudinal joint crack
        else:
            color = (0, 200, 255)     # Gold: General road damage

        # Draw stylish bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        
        # Label banner
        label = f"{cls_name} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_img, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated_img, label, (x1 + 5, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save annotated image
    cv2.imwrite(str(annotated_path), annotated_img)

    # Clean up large OpenCV & PyTorch buffers immediately
    del annotated_img
    del img
    try:
        del p_res
    except Exception:
        pass
    try:
        del c_res
    except Exception:
        pass
    gc.collect()

    # Compute Priority & Severity
    priority, severity_score, dominant_class = compute_priority_and_severity(detections, img_w, img_h)
    
    avg_conf = round(float(np.mean([d["confidence"] for d in detections])), 2) if detections else 0.0

    # Ensure address is populated via reverse geocode if empty
    if not address_val or address_val.strip() in ("", "--"):
        rev = fetch_reverse_geocode(lat_val, lon_val)
        if rev.get("success"):
            address_val = rev.get("address", "")

    # Save to SQLite Database
    now = datetime.now().isoformat()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO damage_reports 
        (id, timestamp, latitude, longitude, image_filename, annotated_filename, damage_type, confidence, priority, severity_score, box_count, status, source, address, accuracy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id, now, lat_val, lon_val, raw_filename, annotated_filename,
        dominant_class, avg_conf, priority, severity_score, len(detections), "PENDING", location_source,
        address_val, accuracy_val
    ))
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "report_id": report_id,
        "damage_type": dominant_class,
        "priority": priority,
        "severity_score": severity_score,
        "detected_count": len(detections),
        "confidence": avg_conf,
        "latitude": lat_val,
        "longitude": lon_val,
        "accuracy": accuracy_val,
        "address": address_val,
        "location_source": location_source,
        "annotated_url": f"/uploads/{annotated_filename}",
        "detections": detections
    })

@app.route("/api/detect/live", methods=["POST"])
def detect_live_frame():
    """
    Ultra-lightweight endpoint for real-time live camera HUD scanning.
    Performs in-memory inference without disk writes for <80ms response times.
    Returns normalized bounding boxes [0.0 to 1.0] for direct overlay rendering.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "detections": []}), 400

    image_file = request.files["image"]
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "detections": []}), 400

    img_h, img_w = img.shape[:2]

    global pothole_model, crack_model
    if pothole_model is None and crack_model is None:
        load_ai_model()

    raw_candidates = []

    with (torch.inference_mode() if torch is not None else nullcontext()):
        if pothole_model is not None:
            try:
                p_res = pothole_model.predict(img, conf=0.18, imgsz=256, device="cpu", verbose=False)
                for r in p_res:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        raw_cls = pothole_model.names.get(cls_id, f"Class {cls_id}")
                        raw_candidates.append({
                            "raw_cls": raw_cls,
                            "confidence": round(conf, 2),
                            "box": [round(x1 / img_w, 4), round(y1 / img_h, 4), round(x2 / img_w, 4), round(y2 / img_h, 4)]
                        })
            except Exception:
                pass

        if crack_model is not None:
            try:
                c_res = crack_model.predict(img, conf=0.18, imgsz=256, device="cpu", verbose=False)
                for r in c_res:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        raw_cls = crack_model.names.get(cls_id, "crack")
                        if "background" in raw_cls.lower():
                            continue
                        raw_candidates.append({
                            "raw_cls": raw_cls,
                            "confidence": round(conf, 2),
                            "box": [round(x1 / img_w, 4), round(y1 / img_h, 4), round(x2 / img_w, 4), round(y2 / img_h, 4)]
                        })
            except Exception:
                pass

    # Non-Maximum Suppression to merge duplicate boxes
    raw_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    detections = []
    for cand in raw_candidates:
        c_box = cand["box"]
        overlap = False
        for kept in detections:
            k_box = kept["box"]
            ix1 = max(c_box[0], k_box[0])
            iy1 = max(c_box[1], k_box[1])
            ix2 = min(c_box[2], k_box[2])
            iy2 = min(c_box[3], k_box[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            c_area = max(0, c_box[2] - c_box[0]) * max(0, c_box[3] - c_box[1])
            k_area = max(0, k_box[2] - k_box[0]) * max(0, k_box[3] - k_box[1])
            union = c_area + k_area - inter
            if union > 0 and (inter / union) > 0.45:
                overlap = True
                break
        if not overlap:
            raw_cls = cand["raw_cls"]
            cls_name = raw_cls.replace("_", " ").replace("-", " ").strip().title()
            detections.append({
                "class": cls_name,
                "confidence": cand["confidence"],
                "box": cand["box"],
                "raw_cls": raw_cls
            })

    del img
    return jsonify({
        "success": True,
        "count": len(detections),
        "detections": detections
    })

# ========================================================
# GEO SERVICES (Proxy endpoints to bypass Render CORS & 403)
# ========================================================
@app.route("/api/geo/reverse", methods=["GET"])
def geo_reverse():
    """Proxies reverse geocoding via Nominatim with valid headers & server cache"""
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
        res = fetch_reverse_geocode(lat, lon)
        return jsonify(res)
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid coordinates"}), 400

@app.route("/api/geo/search", methods=["GET"])
def geo_search():
    """Search roads, landmarks, and cities worldwide"""
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify({"results": []})

    if query in _SEARCH_GEO_CACHE:
        return jsonify({"results": _SEARCH_GEO_CACHE[query]})

    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(query)}&limit=6&addressdetails=1"
        resp = requests.get(url, headers=_GEO_HEADERS, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for item in data:
                addr = item.get("address", {})
                road = addr.get("road") or addr.get("suburb") or addr.get("neighbourhood") or ""
                city = addr.get("city") or addr.get("town") or addr.get("state_district") or ""
                short_name = f"{road}, {city}".strip(", ") if road or city else item.get("display_name", "").split(",")[0]
                results.append({
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "display_name": item.get("display_name", ""),
                    "short_name": short_name,
                    "type": item.get("type", "")
                })
            if len(_SEARCH_GEO_CACHE) > 200:
                _SEARCH_GEO_CACHE.clear()
            _SEARCH_GEO_CACHE[query] = results
            return jsonify({"results": results})
    except Exception as e:
        print(f"Geo search error: {e}")
    return jsonify({"results": []})

@app.route("/api/geo/ip-lookup", methods=["GET"])
def geo_ip_lookup():
    """
    Resolves client approximate latitude and longitude based on public IP.
    Fallback when client lacks GPS or runs indoors / without HTTPS permissions.
    """
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    is_private = not client_ip or client_ip.startswith(("127.", "192.168.", "10.", "172.16.", "::1", "localhost"))
    target_url = "https://ipapi.co/json/" if is_private else f"https://ipapi.co/{client_ip}/json/"

    try:
        resp = requests.get(target_url, headers={"User-Agent": "RoadGuardAI/1.0"}, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if "latitude" in data and "longitude" in data:
                return jsonify({
                    "success": True,
                    "latitude": float(data["latitude"]),
                    "longitude": float(data["longitude"]),
                    "city": data.get("city", "Unknown City"),
                    "region": data.get("region", ""),
                    "country": data.get("country_name", "India"),
                    "source": "IP Geolocation"
                })
    except Exception:
        pass

    try:
        fallback_url = "http://ip-api.com/json/" if is_private else f"http://ip-api.com/json/{client_ip}"
        resp2 = requests.get(fallback_url, timeout=3)
        if resp2.status_code == 200:
            data2 = resp2.json()
            if data2.get("status") == "success":
                return jsonify({
                    "success": True,
                    "latitude": float(data2["lat"]),
                    "longitude": float(data2["lon"]),
                    "city": data2.get("city", ""),
                    "region": data2.get("regionName", ""),
                    "country": data2.get("country", ""),
                    "source": "IP Geolocation"
                })
    except Exception:
        pass

    return jsonify({
        "success": False,
        "latitude": 12.9716,
        "longitude": 77.5946,
        "city": "Bangalore",
        "region": "Karnataka",
        "source": "Default Location"
    })

@app.route("/api/reports", methods=["GET"])
def get_reports():
    """Returns list of reports with optional filtering"""
    priority_filter = request.args.get("priority")
    status_filter = request.args.get("status")

    query = "SELECT * FROM damage_reports WHERE 1=1"
    params = []

    if priority_filter and priority_filter != "ALL":
        query += " AND priority = ?"
        params.append(priority_filter)

    if status_filter and status_filter != "ALL":
        query += " AND status = ?"
        params.append(status_filter)

    query += " ORDER BY timestamp DESC"

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()

    reports = [dict(r) for r in rows]
    return jsonify(reports)

@app.route("/api/reports/<report_id>", methods=["PATCH"])
def update_report(report_id):
    """Allows Officer to update report status or notes"""
    data = request.json or {}
    new_status = data.get("status")
    notes = data.get("officer_notes")

    conn = get_db()
    cursor = conn.cursor()
    if new_status:
        cursor.execute("UPDATE damage_reports SET status = ? WHERE id = ?", (new_status, report_id))
    if notes is not None:
        cursor.execute("UPDATE damage_reports SET officer_notes = ? WHERE id = ?", (notes, report_id))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Updated successfully"})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Dashboard statistics summary"""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM damage_reports").fetchone()[0]
    critical = conn.execute("SELECT COUNT(*) FROM damage_reports WHERE priority = 'CRITICAL'").fetchone()[0]
    high = conn.execute("SELECT COUNT(*) FROM damage_reports WHERE priority = 'HIGH'").fetchone()[0]
    medium = conn.execute("SELECT COUNT(*) FROM damage_reports WHERE priority = 'MEDIUM'").fetchone()[0]
    low = conn.execute("SELECT COUNT(*) FROM damage_reports WHERE priority = 'LOW'").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM damage_reports WHERE status = 'RESOLVED'").fetchone()[0]
    conn.close()

    return jsonify({
        "total": total,
        "critical": critical,
        "high": high,
        "medium": medium,
        "low": low,
        "resolved": resolved,
        "model_loaded": model_source
    })

@app.route("/api/seed-demo-data", methods=["POST"])
def seed_demo_data():
    """Populates 5 realistic Indian road damage pins for instant presentation demo"""
    sample_points = [
        {"lat": 12.9716, "lon": 77.5946, "type": "Pothole", "prio": "CRITICAL", "score": 9, "status": "PENDING"},
        {"lat": 12.9784, "lon": 77.6012, "type": "Manhole Damage", "prio": "CRITICAL", "score": 8, "status": "IN_PROGRESS"},
        {"lat": 12.9650, "lon": 77.5880, "type": "Alligator Crack", "prio": "HIGH", "score": 7, "status": "PENDING"},
        {"lat": 12.9820, "lon": 77.5900, "type": "Transverse Crack", "prio": "MEDIUM", "score": 5, "status": "PENDING"},
        {"lat": 12.9600, "lon": 77.6100, "type": "Longitudinal Crack", "prio": "LOW", "score": 3, "status": "RESOLVED"},
    ]
    conn = get_db()
    for pt in sample_points:
        rid = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO damage_reports 
            (id, timestamp, latitude, longitude, image_filename, annotated_filename, damage_type, confidence, priority, severity_score, box_count, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rid, datetime.now().isoformat(), pt["lat"], pt["lon"],
            "demo.jpg", "demo.jpg", pt["type"], 0.88, pt["prio"], pt["score"], 1, pt["status"], "Demo Seed"
        ))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Demo sample points created"})

def ensure_ssl_certs():
    cert_path = BASE_DIR / "server" / "cert.pem"
    key_path = BASE_DIR / "server" / "key.pem"
    if not (cert_path.exists() and key_path.exists()):
        from cryptography import x509  # type: ignore
        from cryptography.x509.oid import NameOID  # type: ignore
        from cryptography.hazmat.primitives import hashes  # type: ignore
        from cryptography.hazmat.primitives.asymmetric import rsa  # type: ignore
        from cryptography.hazmat.primitives import serialization  # type: ignore
        import ipaddress, socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'RoadGuard')])
        san_list = [x509.DNSName('localhost'), x509.IPAddress(ipaddress.IPv4Address('127.0.0.1'))]
        try:
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
        except Exception:
            pass

        cert = (x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )
        with open(cert_path, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(key_path, 'wb') as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
    return str(cert_path), str(key_path)

if __name__ == "__main__":
    import sys
    use_https = "--https" in sys.argv
    is_render = bool(os.environ.get("RENDER"))
    is_hf = bool(os.environ.get("SPACE_ID"))
    is_cloud = is_render or is_hf
    default_port = 10000 if is_render else (7860 if is_hf else 5000)
    port = int(os.environ.get("PORT", default_port))

    # Detect active environment URL
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url:
        dashboard_url = render_url
        mobile_url = f"{render_url}/mobile"
        env_label = "☁️ Render Cloud (Live Public HTTPS)"
    elif is_hf:
        dashboard_url = f"https://huggingface.co/spaces/{os.environ.get('SPACE_ID', '')}"
        mobile_url = f"{dashboard_url}/mobile"
        env_label = "☁️ Hugging Face Space"
    else:
        proto = "https" if use_https else "http"
        dashboard_url = f"{proto}://localhost:{port}"
        mobile_url = f"{proto}://localhost:{port}/mobile"
        env_label = "💻 Local Computer"

    print(f"\n=======================================================")
    print(f"🛣️  ROAD DAMAGE AI DETECTION SYSTEM READY")
    print(f"🌐 Environment: {env_label}")
    print(f"👉 Officer Portal Dashboard: {dashboard_url}")
    print(f"👉 Mobile Camera Interface:  {mobile_url}")
    print(f"=======================================================\n")

    debug_mode = os.environ.get("FLASK_DEBUG") == "1"
    if use_https:
        cert_p, key_p = ensure_ssl_certs()
        app.run(host="0.0.0.0", port=port, ssl_context=(cert_p, key_p), debug=debug_mode, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)
