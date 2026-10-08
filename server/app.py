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
try:
    import torch  # type: ignore
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    torch = None

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

# Load YOLO Model (custom best.pt if available, else yolov8n.pt)
model = None
model_source = "None"

def load_ai_model():
    global model, model_source
    from ultralytics import YOLO  # type: ignore
    custom_model_path = MODELS_DIR / "best.pt"
    if custom_model_path.exists():
        print(f"✅ Loading custom trained Indian road damage model: {custom_model_path}")
        model = YOLO(str(custom_model_path))
        model_source = "Custom Indian Road Model (best.pt)"
    else:
        print("⚠️ models/best.pt not found yet. Loading base YOLOv8n model...")
        model = YOLO("yolov8n.pt")
        model_source = "Base YOLOv8 (Pending best.pt download)"
    try:
        model.to("cpu")
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
    
    # Priority mapping weights by class keyword (matched to trained model classes)
    # Model classes: {0: 'Drain Hole', 1: 'circle-drain-clean-', 2: 'drain-clean-',
    #   3: 'drain-not clean-', 4: 'hole', 5: 'manhole', 6: 'pothole', 7: 'sewer cover'}
    class_weights = {
        "pothole": 5,           # Class 6 — most dangerous
        "hole": 5,              # Class 4 — generic hole
        "drain hole": 4,        # Class 0 — open drain hole
        "manhole": 5,           # Class 5 — missing/damaged manhole cover
        "sewer cover": 3,       # Class 7 — damaged sewer cover
        "drain-not clean": 4,   # Class 3 — clogged drain (flood risk)
        "drain-clean": 2,       # Class 2 — clean drain (low risk)
        "circle-drain-clean": 2,# Class 1 — circular drain, clean
        "drain": 3,             # Catch-all for drain variants
        "sewer": 3,
        "crack": 3,
        "alligator": 4,
        "transverse": 3,
        "longitudinal": 2,
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

# Routes
@app.route("/")
def dashboard_view():
    """Serves the Officer Portal Dashboard"""
    return render_template("dashboard.html", model_source=model_source)

@app.route("/mobile")
def mobile_view():
    """Serves the Mobile Camera & GPS capture interface"""
    return render_template("mobile.html")

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

    # Priority 1: Check if photo contains actual EXIF GPS coordinates embedded by camera
    exif_lat, exif_lon = extract_exif_gps(raw_path)
    if exif_lat is not None and exif_lon is not None:
        lat_val = exif_lat
        lon_val = exif_lon
        location_source = "Photo EXIF GPS"
        print(f"📍 Extracted EXIF GPS from image metadata: {lat_val}, {lon_val}")
    elif lat_val is not None and lon_val is not None:
        location_source = source_val or "Device/Selected GPS"
    else:
        # Fallback to demo coordinates
        lat_val = round(12.9716 + float(np.random.uniform(-0.02, 0.02)), 6)
        lon_val = round(77.5946 + float(np.random.uniform(-0.02, 0.02)), 6)
        location_source = "Default (Bangalore)"

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

    # Run inference
    global model
    if model is None:
        load_ai_model()

    if torch is not None:
        with torch.no_grad():
            results = model.predict(img, conf=0.25, imgsz=480, device="cpu", verbose=False)
    else:
        results = model.predict(img, conf=0.25, imgsz=480, device="cpu", verbose=False)
    
    detections = []
    annotated_img = img.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"Class {cls_id}")

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 3),
                "box": [x1, y1, x2, y2]
            })

            # Draw stylish bounding box
            color = (0, 0, 255) if "pothole" in cls_name.lower() or "hole" in cls_name.lower() else (0, 165, 255)
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
    del results
    gc.collect()

    # Compute Priority & Severity
    priority, severity_score, dominant_class = compute_priority_and_severity(detections, img_w, img_h)
    
    avg_conf = round(float(np.mean([d["confidence"] for d in detections])), 2) if detections else 0.0

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
