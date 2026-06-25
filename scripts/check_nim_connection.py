"""
Run with: python scripts/check_nim_connection.py

Isolated diagnostic - does ONE minimal request to NIM, nothing from the rest
of the agent. If this hangs or fails, the problem is your NIM connectivity,
not the agent. If this works fine but run_demo.py still times out, the
problem is likely the agent's longer/more complex prompts, not connectivity
itself - useful to know which one you're dealing with.

Uses a short 15s timeout (not the agent's 60s) specifically so you get a
fast, clear diagnosis instead of another long hang.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv()

import requests
import socket

API_KEY = os.environ.get("NVIDIA_API_KEY")
MODEL = os.environ.get("NIM_MODEL_ID")
BASE_URL = "https://integrate.api.nvidia.com/v1"


def check_dns():
    print("1. DNS resolution for integrate.api.nvidia.com ...", end=" ")
    try:
        ip = socket.gethostbyname("integrate.api.nvidia.com")
        print(f"OK -> {ip}")
        return True
    except socket.gaierror as e:
        print(f"FAILED: {e}")
        print("   -> DNS can't resolve the host at all. Check VPN/network, not the API key or model.")
        return False


def check_key_present():
    print("2. NVIDIA_API_KEY present in environment ...", end=" ")
    if not API_KEY or API_KEY.startswith("REPLACE") or API_KEY == "nvapi-...":
        print("MISSING or placeholder")
        return False
    print(f"OK -> starts with {API_KEY[:12]}...")
    return True


def check_model_set():
    print("3. NIM_MODEL_ID set (not the placeholder) ...", end=" ")
    if not MODEL or "REPLACE_WITH" in MODEL:
        print("MISSING or still the placeholder string")
        return False
    print(f"OK -> {MODEL}")
    return True


def check_live_request():
    print(f"4. Live request to {BASE_URL}/chat/completions with model '{MODEL}' ...")
    print("   (15s timeout - if this is the slow part, the issue is the model/endpoint, not your network)")
    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
                "max_tokens": 5,
            },
            timeout=15,
        )
        elapsed = time.time() - start
        print(f"   Response in {elapsed:.1f}s - status {resp.status_code}")
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            print(f"   Model replied: {content!r}")
            print("   -> NIM connectivity and this model ID both work.")
        elif resp.status_code == 401:
            print("   -> 401: API key is invalid or not authorized for this model. Re-check it on build.nvidia.com.")
        elif resp.status_code == 404:
            print(f"   -> 404: model ID '{MODEL}' was not found. Copy the EXACT ID from the model's catalog page.")
        else:
            print(f"   -> Unexpected status. Body: {resp.text[:300]}")
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"   TIMED OUT after {elapsed:.1f}s")
        print("   -> Connection was established but no response came back in time. Likely causes:")
        print("      a) the model ID is wrong/doesn't exist and the gateway hangs instead of 404ing fast")
        print("      b) the model is real but very large/cold-starting")
        print("      c) something on your network (corporate firewall, antivirus TLS inspection, VPN) is")
        print("         interfering with the HTTPS connection after it opens")
        print("   -> Try a smaller, well-known model ID next (check the catalog for one) to isolate (a)/(b) vs (c).")
    except requests.exceptions.ConnectionError as e:
        print(f"   CONNECTION ERROR: {e}")
        print("   -> Couldn't even establish a connection. Check VPN/firewall/proxy settings.")


if __name__ == "__main__":
    print("=== NIM connectivity diagnostic ===\n")
    if not check_dns():
        sys.exit(1)
    key_ok = check_key_present()
    model_ok = check_model_set()
    if not (key_ok and model_ok):
        print("\nFix the above in your .env before testing the live request.")
        sys.exit(1)
    print()
    check_live_request()
