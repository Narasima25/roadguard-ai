
# 🛣️ RoadGuard AI — Intelligent Road Damage Detection & Officer Portal

An AI-powered computer vision and geospatial surveillance system designed to detect road damage (potholes, cracks, sewer/manhole issues) from cameras or smartphones, geotag coordinates with real-time GPS, categorize damage type, compute priority levels, and display actionable reports on an Officer Dashboard.

---

## 🚀 Quick Start

### 1. Start the Server
Run the starter script:
```bash
python run.py
```
*(Or directly `python server/app.py`)*

### 2. Open the Interfaces

- **🖥️ Officer Portal Dashboard:** [http://localhost:5000](http://localhost:5000)
  - Interactive Leaflet map with color-coded damage pins (Critical, High, Medium, Low)
  - Real-time incident feed and severity score breakdown
  - Status management (Pending ➡️ Dispatched ➡️ Resolved)
  - "+ Seed Demo Pins" button to populate immediate sample data for presentations

- **📱 Mobile Camera & GPS Scanner:** [http://localhost:5000/mobile](http://localhost:5000/mobile)
  - Works on your phone (connect phone and laptop to same Wi-Fi, open `http://<laptop-ip>:5000/mobile`)
  - Uses the back camera with HUD telemetry
  - Reads live GPS coordinates from the phone's browser
  - Manual "📸" snap button or "Auto Scan" switch (scans every 3 seconds)

---

## 🧠 Adding Your Trained Model

Once your Google Colab training finishes:
1. Download **`best.pt`** from Colab.
2. Place it in:
   ```
   d:\ai\models\best.pt
   ```
3. Restart the server (`python run.py`). The server will automatically switch to your custom trained Indian road damage model!

*(Until you drop `best.pt`, the server automatically runs using base YOLOv8 so you can test everything right now).*
