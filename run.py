"""
RoadGuard AI — Start Script
"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = BASE_DIR / "server" / "app.py"

import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    if "--public" in sys.argv:
        import run_public
        run_public.main()
        sys.exit(0)

    local_ip = get_local_ip()
    use_https = "--https" in sys.argv
    proto = "https" if use_https else "http"
    
    print("==================================================")
    print("🚀 Starting RoadGuard AI Server...")
    print(f"📍 PC Dashboard:  {proto}://localhost:5000")
    print(f"📱 Phone Scanner: {proto}://{local_ip}:5000/mobile")
    print("💡 Want a public HTTPS URL with real phone GPS?")
    print("   Run: python run.py --public")
    print("==================================================")

    args = [sys.executable, str(SERVER_SCRIPT)]
    if use_https:
        args.append("--https")
    subprocess.run(args)
