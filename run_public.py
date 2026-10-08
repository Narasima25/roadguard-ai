"""
RoadGuard AI — Instant Public HTTPS Launcher (Cloudflare Tunnel)
Launches the local AI server and provides an official public HTTPS URL
with valid SSL certificates to unlock physical satellite GPS and mobile camera.
"""
import sys
import subprocess
import time
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED = BASE_DIR / "cloudflared.exe"
SERVER_SCRIPT = BASE_DIR / "server" / "app.py"

def main():
    print("==================================================================")
    print("🚀 Initializing RoadGuard AI Server + Cloudflare Public HTTPS...")
    print("==================================================================")

    # 1. Start Flask AI Server
    server_process = subprocess.Popen([sys.executable, str(SERVER_SCRIPT)])
    time.sleep(2)

    # 2. Start Cloudflare Tunnel
    if not CLOUDFLARED.exists():
        print("❌ cloudflared.exe not found.")
        return

    tunnel_cmd = [str(CLOUDFLARED), "tunnel", "--url", "http://127.0.0.1:5000"]
    tunnel_process = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Read output to capture public URL
    start_time = time.time()
    while time.time() - start_time < 20:
        line = tunnel_process.stdout.readline()
        if not line:
            continue
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        print("\n" + "=" * 68)
        print("🎉 SUCCESS! ROADGUARD AI IS LIVE ON VERIFIED PUBLIC HTTPS:")
        print("=" * 68)
        print(f"📱 Phone Mobile Cam (HTTPS): {public_url}/mobile")
        print(f"🖥️ Public Officer Portal:     {public_url}")
        print(f"📍 Local PC Dashboard:        http://localhost:5000")
        print("=" * 68)
        print("🔒 Trusted SSL Green Padlock Active!")
        print("   -> Native phone satellite GPS: UNLOCKED ✅")
        print("   -> Live camera stream:        UNLOCKED ✅")
        print("   -> Works from anywhere in the world (4G/5G/Wi-Fi)!")
        print("=" * 68 + "\n")
    else:
        print("⚠️ Tunnel URL not detected yet. Check cloudflared output.")

    try:
        # Keep running until Ctrl+C
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server and public tunnel...")
        server_process.terminate()
        tunnel_process.terminate()

if __name__ == "__main__":
    main()
