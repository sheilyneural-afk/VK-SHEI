#!/usr/bin/env python3
"""GitHub App device flow.

Uso: device_login.py <perfil> <client_id>
Nunca imprime ni guarda credenciales dentro del repositorio.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

if len(sys.argv) != 3:
    raise SystemExit("uso: device_login.py <perfil> <client_id>")

profile, client_id = sys.argv[1:3]
if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", profile):
    raise SystemExit("perfil inválido")
if not client_id:
    raise SystemExit("falta client_id")

cfg_dir = os.path.expanduser("~/.config/vk-shei-github")
os.makedirs(cfg_dir, mode=0o700, exist_ok=True)
state_file = os.path.join(cfg_dir, f"{profile}.status")
token_file = os.path.join(cfg_dir, f"{profile}.json")

def post(url, data):
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode(),
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def set_status(value):
    fd = os.open(state_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")

device = post("https://github.com/login/device/code", {"client_id": client_id})
if "device_code" not in device or "user_code" not in device:
    raise SystemExit("GitHub no devolvió un código de dispositivo")

print(
    "CODE", device["user_code"], device["verification_uri"],
    "expira_en", device["expires_in"], flush=True,
)
set_status("pending")

# El padre termina para devolver el control al agente; el hijo continúa sondeando.
if os.fork():
    raise SystemExit(0)

interval = int(device.get("interval", 5))
deadline = time.time() + int(device["expires_in"])

while time.time() < deadline:
    time.sleep(interval)
    try:
        token = post("https://github.com/login/oauth/access_token", {
            "client_id": client_id,
            "device_code": device["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        })
    except Exception:
        continue

    if "access_token" in token:
        token["client_id"] = client_id
        token["obtained_at"] = int(time.time())
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(token, handle)
        set_status("ok")
        raise SystemExit(0)

    error = token.get("error")
    if error == "slow_down":
        interval += 5
    elif error != "authorization_pending":
        set_status("error " + str(error))
        raise SystemExit(1)

set_status("expired")
