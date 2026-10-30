# Acceso de GitHub mediante device flow

Estos scripts permiten que un agente que trabaja en una máquina remota use una GitHub App sin recibir la contraseña, el segundo factor ni una clave privada.

## Preparación

1. Crea una GitHub App y activa **Enable Device Flow**.
2. Dale únicamente los permisos necesarios.
3. Instálala en el repositorio.
4. Conserva su **Client ID**. Es público; el token no lo es.

## Primer acceso

Desde la máquina donde trabaja el agente:

```bash
python3 tools/github-device-flow/device_login.py vk-shei CLIENT_ID_DE_LA_APP
```

El comando imprime un código y la dirección oficial:

```text
CODE XXXX-XXXX https://github.com/login/device expira_en 899
```

Abre esa dirección desde el móvil, introduce el código y acepta. El agente seguirá esperando en segundo plano y guardará el token fuera del repositorio.

## Uso

```bash
TOKEN="$(python3 tools/github-device-flow/gh_token.py vk-shei)"
curl -s -H "Authorization: Bearer $TOKEN" https://api.github.com/user
```

Para Git:

```bash
git push "https://x-access-token:$(python3 tools/github-device-flow/gh_token.py vk-shei)@github.com/OWNER/REPO.git" HEAD:mi-rama
```

El remoto permanente debe continuar limpio, sin credenciales:

```bash
git remote set-url origin https://github.com/OWNER/REPO.git
```

Los tokens se guardan en `~/.config/vk-shei-github/` con permisos `600`. Nunca pegues el token en el chat ni lo guardes en el repositorio.

El flujo recomendado es crear una rama y un pull request. La protección de `main` debe exigir las comprobaciones antes de fusionar.
