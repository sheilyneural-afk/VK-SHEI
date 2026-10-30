#!/usr/bin/env python3
"""Imprime un token GitHub válido para usarlo dentro de otro comando."""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

if len(sys.argv) != 2:
    raise SystemExit("uso: gh_token.py <perfil>")
profile = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", profile):
    raise SystemExit("perfil inválido")

path = os.path.expanduser(f"~/.config/vk-shei-github/{profile}.json")
with open(path, encoding="utf-8") as handle:
    token = json.load(handle)

expires_at = token.get("obtained_at", 0) + token.get("expires_in", 28800)
if time.time() <= expires_at - 300:
    print(token["access_token"])
    raise SystemExit(0)

refresh_token = token.get("refresh_token")
if not refresh_token:
    raise SystemExit("REFRESH_FAILED: repite device_login.py")

request = urllib.request.Request(
    "https://github.com/login/oauth/access_token",
    data=urllib.parse.urlencode({
        "client_id": token["client_id"],
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode(),
    headers={"Accept": "application/json"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    renewed = json.load(response)

if "access_token" not in renewed:
    raise SystemExit("REFRESH_FAILED: repite device_login.py")

renewed["client_id"] = token["client_id"]
renewed["obtained_at"] = int(time.time())
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(renewed, handle)
print(renewed["access_token"])
