"""
Servidor local de la demo: sirve las escenas y corre la inferencia real.

Arquitectura:
    navegador  ──JPEG (POST /predict)──▶  este servidor  ──▶ MediaPipe Hands
                ◀──JSON {letra, confianza, landmarks}──         + signa_alphabet.tflite

El navegador se queda con la cámara, el render y las animaciones (el diseño de
los prototipos es HTML, así que se conserva tal cual); Python se queda con el
modelo. Así la demo usa EXACTAMENTE el mismo código de inferencia que
scripts/predict_alphabet_realtime.py — el número que reportamos y el que se ve
en el video vienen del mismo lugar, sin reimplementar features en JS.

Sin dependencias nuevas: http.server de la stdlib alcanza y sobra en localhost
(MediaPipe + TFLite tardan ~20 ms; el HTTP local, menos de 1 ms).

Uso:
    python demo/server.py
    python demo/server.py --port 8800 --threshold 0.6 --confirm-frames 4
    python demo/server.py --model models/exports/signa_alphabet_v2.tflite

Después abrí:
    http://localhost:8000/simple.html    → "Reconocimiento Simple"
    http://localhost:8000/nombre.html    → "Reconocimiento Nombre"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.alphabet_runner import AlphabetRecognizer  # noqa: E402
from src.inference.sign_runner import SignRecognizer  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"
CONFIG_PATH = ROOT / "configs" / "alphabet_config.yaml"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
}

SIGNS_DIR = STATIC_DIR / "signs"

# MediaPipe no es thread-safe y ThreadingHTTPServer atiende cada request en su
# propio hilo, así que serializamos el acceso al reconocedor.
_lock = threading.Lock()
_recognizer: AlphabetRecognizer | None = None
_signs: SignRecognizer | None = None          # se carga sólo si se pide --signs
_stats = {"frames": 0, "total_ms": 0.0}


class SignAnimations:
    """
    Resuelve la animación 3D de cada letra, igual que hace la app.

    En signa-mobile, `animationPreload.ts` pide en lote a
    `POST /signs/animations` las URLs prefirmadas de R2 y las cachea; los
    significados sin seña o sin animación simplemente no vienen en la respuesta
    y el componente cae a un placeholder. Acá se replica eso mismo, con dos
    diferencias que impone el navegador:

      1. El proxy es necesario. signa-api levanta con `.cors(disable)`, así que
         una página en localhost:8000 no puede llamar a localhost:8080: el
         navegador bloquea la respuesta. Pidiéndolo desde Python no hay CORS.
         (El .glb en sí lo baja el navegador directo de R2, que sí tiene CORS
         habilitado — es lo mismo que hace el WebView en la app.)

      2. Qué "significado" corresponde a una letra no está definido en el
         contenido del curso, que usa palabras en minúscula ("hola", "madre").
         Por eso se prueban varias convenciones en la misma request: como el
         endpoint en lote omite lo que no encuentra, preguntar de más es gratis.

    Antes de ir al API se mira `demo/static/signs/<LETRA>.glb`. Eso permite
    tener la demo andando sin backend: alcanza con dejar los archivos ahí.
    """

    # Se prueban en orden; la primera que el API resuelva gana.
    MEANING_TEMPLATES = ("{lower}", "{upper}", "letra {lower}")

    def __init__(self, api_url: str | None, token: str | None,
                 user: str | None = None, password: str | None = None,
                 timeout: float = 6.0):
        self.api_url = api_url.rstrip("/") if api_url else None
        self.token = token
        self.user = user
        self.password = password
        self.refresh_token: str | None = None
        self.timeout = timeout
        self._cache: dict[str, tuple[str | None, float, str]] = {}
        self._lock = threading.Lock()
        self.last_error: str | None = None
        # El endpoint en lote se prueba una vez; si el deploy no lo deja usar,
        # se cae al GET por seña para el resto de la sesión.
        self._batch_ok = True

    def _local(self, letter: str) -> str | None:
        for name in (f"{letter}.glb", f"{letter.lower()}.glb"):
            if (SIGNS_DIR / name).is_file():
                return f"signs/{urllib.parse.quote(name)}"
        return None

    def _meanings(self, letters: list[str]) -> tuple[list[str], dict[str, str]]:
        """Devuelve (significados a pedir, mapa significado → letra)."""
        wanted, owner = [], {}
        for letter in letters:
            for tpl in self.MEANING_TEMPLATES:
                meaning = tpl.format(lower=letter.lower(), upper=letter.upper())
                if meaning not in owner:
                    owner[meaning] = letter
                    wanted.append(meaning)
        return wanted, owner

    def login(self) -> bool:
        """
        POST /auth/login → guarda el access token.

        El API desplegado no corre con el perfil `local`, así que /signs pide
        autenticación. En vez de pegar un JWT a mano —que vence y hay que
        renovar en medio de una grabación— el servidor se loguea solo, igual que
        el cliente móvil, y reintenta cuando el token caduca.
        """
        if not (self.api_url and self.user and self.password):
            return False
        return self._store_tokens(*self._request(
            "/auth/login", {"identifier": self.user, "password": self.password}, auth=False))

    def refresh(self) -> bool:
        """
        POST /auth/refresh — el mismo camino que usa el interceptor de la app
        (`client.ts`): ante un 401 se canjea el refresh token en vez de volver a
        pedir credenciales. Si falla, se cae al login completo.
        """
        if not self.refresh_token:
            return False
        return self._store_tokens(*self._request(
            "/auth/refresh", {"refresh_token": self.refresh_token}, auth=False))

    def _store_tokens(self, payload, err) -> bool:
        """AuthResponse {access_token, refresh_token} — el API serializa en snake_case."""
        if isinstance(payload, dict):
            token = payload.get("access_token") or payload.get("accessToken")
            if token:
                self.token = token
                self.refresh_token = (payload.get("refresh_token")
                                      or payload.get("refreshToken") or self.refresh_token)
                self.last_error = None
                return True
        self.last_error = err if err and err != "AUTH" else "el API rechazó las credenciales"
        return False

    def _get_json(self, path: str, body: dict | None = None) -> tuple[object | None, str | None]:
        """
        Llama al API renovando el token si hace falta.

        Un 401/403 se trata como "el token venció": se reintenta una vez después
        de volver a loguearse. Sin esto, a los ~15 minutos de demo el avatar
        dejaría de cargar sin explicación.
        """
        payload, err = self._request(path, body)
        if err == "AUTH" and (self.refresh() or self.login()):
            payload, err = self._request(path, body)
        if err == "AUTH":
            err = ("el API pide autenticación"
                   + (" (credenciales rechazadas)" if self.user else " — pasá --api-user/--api-password"))
        return payload, err

    def _request(self, path: str, body: dict | None = None,
                 auth: bool = True) -> tuple[object | None, str | None]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            f"{self.api_url}{path}", data=data,
            method="POST" if data else "GET",
            headers={"Accept": "application/json",
                     **({"Content-Type": "application/json"} if data else {})},
        )
        if auth and self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read() or b"null"), None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, None                    # no hay seña / sin animación: normal
            if exc.code in (401, 403):
                return None, "AUTH"                  # lo maneja _get_json
            return None, f"el API respondió {exc.code}"
        except Exception as exc:
            return None, f"no se pudo hablar con el API ({exc.__class__.__name__})"

    def _ask_one(self, letter: str) -> str | None:
        """
        GET /signs/{meaning}/animation — el endpoint por seña.

        Devuelve SignAnimationResponse, que sale del API en snake_case
        (JacksonConfig fija PropertyNamingStrategies.SNAKE_CASE globalmente):

            {"sign_id": "...",
             "animation_url": "https://….r2.cloudflarestorage.com/signa-animations/lsa/hola.glb?X-Amz-…",
             "expires_in_seconds": 900}

        `animation_url` ya viene prefirmada contra R2: es la URL final del .glb,
        lista para que el navegador la cargue. Se acepta también camelCase por si
        alguna versión del API no aplica la estrategia.
        """
        for tpl in self.MEANING_TEMPLATES:
            meaning = tpl.format(lower=letter.lower(), upper=letter.upper())
            payload, err = self._get_json(f"/signs/{urllib.parse.quote(meaning)}/animation")
            if err:
                self.last_error = err
                return None
            if isinstance(payload, dict):
                url = payload.get("animation_url") or payload.get("animationUrl")
                if url:
                    return url
        return None

    def _ask_api(self, letters: list[str]) -> dict[str, str]:
        """
        Resuelve las letras que faltan contra signa-api.

        Primero se intenta el endpoint en lote (POST /signs/animations), que es
        el que usa el reproductor de lecciones en la app: una sola ida y vuelta
        para todas las letras del nombre. Lo que quede sin resolver se reintenta
        una por una con GET /signs/{meaning}/animation, que es el camino
        garantizado y el que sirve si el API desplegado no tiene el endpoint en
        lote.
        """
        if not self.api_url or not letters:
            return {}

        found: dict[str, str] = {}
        err = None

        if self._batch_ok:
            meanings, owner = self._meanings(letters)
            payload, err = self._get_json("/signs/animations", {"meanings": meanings})

            if isinstance(payload, dict):
                for meaning, url in payload.items():
                    letter = owner.get(meaning)
                    # Respetamos el orden de MEANING_TEMPLATES: no pisamos una ya resuelta.
                    if letter and url and letter not in found:
                        found[letter] = url

            if err:
                # El deploy expone el GET por seña como público pero deja el lote
                # detrás de auth: un 403 acá no significa que no podamos resolver
                # nada, sólo que este atajo no está disponible. Lo desactivamos
                # para no pagar el POST fallido (y su reintento de login) en cada
                # request, y seguimos por el camino de a una.
                self._batch_ok = False
                if "no se pudo hablar" in err:
                    self.last_error = err      # API caído: no insistir 26 veces
                    return found

        # El 403 del lote ya está contemplado: si el camino de a una funciona no
        # es un error que el front deba mostrar. Lo olvidamos y dejamos que
        # `_ask_one` reporte sólo los fallos reales.
        self.last_error = None

        for letter in letters:
            if letter in found:
                continue
            url = self._ask_one(letter)
            if url:
                found[letter] = url

        # Si resolvimos algo, la demo funciona y no hay nada que mostrar.
        if found:
            self.last_error = None
        return found

    def resolve(self, letters: list[str]) -> dict[str, dict | None]:
        now = time.time()
        out: dict[str, dict | None] = {}
        pending: list[str] = []

        with self._lock:
            for letter in letters:
                hit = self._cache.get(letter)
                if hit and hit[1] > now:
                    out[letter] = {"url": hit[0], "source": hit[2]} if hit[0] else None
                else:
                    pending.append(letter)

        for letter in list(pending):
            local = self._local(letter)
            if local:
                out[letter] = {"url": local, "source": "local"}
                with self._lock:
                    # Los archivos locales no vencen.
                    self._cache[letter] = (local, now + 86400, "local")
                pending.remove(letter)

        from_api = self._ask_api(pending)
        # Las URLs de R2 vienen prefirmadas y vencen (15 min por defecto);
        # las cacheamos por menos tiempo para no servir uno ya vencido.
        ttl = now + 10 * 60
        with self._lock:
            for letter in pending:
                url = from_api.get(letter)
                self._cache[letter] = (url, ttl if url else now + 60, "api")
                out[letter] = {"url": url, "source": "api"} if url else None

        return out


_animations: SignAnimations | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # el log por request ensucia la consola durante la demo

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200):
        self._send(code, json.dumps(payload).encode("utf-8"), CONTENT_TYPES[".json"])

    # ─── GET ─────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/simple.html"

        if path == "/meta":
            rec = _recognizer
            avg = _stats["total_ms"] / _stats["frames"] if _stats["frames"] else 0.0
            payload = {
                "labels": rec.labels,
                "model": rec.model_path.name,
                "accuracy": rec.meta.get("accuracy"),
                "cv": rec.meta.get("cv"),
                "ensemble": rec.meta.get("ensemble"),
                "threshold": rec.threshold,
                "thresholds": rec.thresholds,
                "confirm_frames": rec.confirm_frames,
                "avg_infer_ms": round(avg, 1),
                "signs_available": _signs is not None,
                "animations_api": _animations.api_url,
                "animations_local": SIGNS_DIR.is_dir(),
            }
            if _signs is not None:
                payload["signs"] = {
                    "labels": _signs.labels,
                    "model": _signs.model_path.name,
                    "accuracy": _signs.meta.get("accuracy"),
                    "sequence_length": _signs.sequence_length,
                }
            return self._json(payload)

        if path == "/sign-animations":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            raw = (query.get("letters") or [""])[0]
            letters = [c.strip().upper() for c in raw.split(",") if c.strip()][:32]
            if not letters:
                return self._json({"error": "falta el parámetro letters"}, 400)
            found = _animations.resolve(letters)
            return self._json({
                "animations": found,
                "api": _animations.api_url,
                "error": _animations.last_error,
            })

        # Servir estáticos, sin salirse de demo/static/
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        if not target.is_file() or STATIC_DIR.resolve() not in target.parents:
            return self._send(404, b"not found", "text/plain; charset=utf-8")

        return self._send(200, target.read_bytes(),
                          CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

    # ─── POST ────────────────────────────────────────────────────────────────

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/reset":
            with _lock:
                _recognizer.reset()
                if _signs is not None:
                    _signs.reset()
            return self._json({"ok": True})

        if path not in ("/predict", "/predict_sign"):
            return self._send(404, b"not found", "text/plain; charset=utf-8")

        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return self._json({"error": "frame vacío"}, 400)

        raw = self.rfile.read(length)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self._json({"error": "no se pudo decodificar el frame"}, 400)

        t0 = time.perf_counter()

        if path == "/predict_sign":
            if _signs is None:
                return self._json({"error": "señas dinámicas no habilitadas "
                                            "(levantá el servidor con --signs)"}, 409)
            with _lock:
                sr = _signs.process(frame)
                sign_labels = _signs.labels
            elapsed = (time.perf_counter() - t0) * 1000
            _stats["frames"] += 1
            _stats["total_ms"] += elapsed

            top = []
            if len(sr.probs):
                order = np.argsort(sr.probs)[::-1][:3]
                top = [{"sign": sign_labels[i], "confidence": round(float(sr.probs[i]), 4)}
                       for i in order]
            return self._json({
                "body": bool(sr.body_present),
                "sign": sr.sign,
                "confidence": round(float(sr.confidence), 4),
                "confirmed": bool(sr.confirmed),
                "resting": bool(sr.resting),
                "progress": round(float(sr.progress), 3),
                "pose": sr.pose_px,
                "hands": sr.hands_px,
                "top": top,
                "infer_ms": round(elapsed, 1),
            })

        # La letra que la escena está pidiendo, si la hay: activa el modo
        # verificación, que es el que hace usables todas las letras.
        target = None
        if "?" in self.path:
            from urllib.parse import parse_qs, urlparse
            target = (parse_qs(urlparse(self.path).query).get("target") or [None])[0]

        h, w = frame.shape[:2]
        with _lock:
            result = _recognizer.process(frame, target=target)
            labels = _recognizer.labels
        elapsed = (time.perf_counter() - t0) * 1000
        _stats["frames"] += 1
        _stats["total_ms"] += elapsed

        # Landmarks normalizados a [0,1] sobre el frame recibido: el navegador
        # los mapea al <video> según su object-fit, sin saber su tamaño real.
        landmarks = None
        if result.landmarks_px is not None:
            landmarks = [[round(float(x) / w, 5), round(float(y) / h, 5)]
                         for x, y in result.landmarks_px]

        top = []
        if result.hand_present and len(result.probs):
            order = np.argsort(result.probs)[::-1][:3]
            top = [{"letter": labels[i], "confidence": round(float(result.probs[i]), 4)}
                   for i in order]

        return self._json({
            "hand": bool(result.hand_present),
            # None si el modelo no usa la cara; False si la necesita y no la ve
            # (ahí T e I se vuelven indistinguibles: la posición va en cero).
            "face": result.face_present,
            "letter": result.letter,
            "confidence": round(float(result.confidence), 4),
            "confirmed": bool(result.confirmed),
            "target": result.target,
            "target_confidence": round(float(result.target_confidence), 4),
            "target_ok": bool(result.target_ok),
            "mismatch": bool(result.mismatch),
            "landmarks": landmarks,
            "top": top,
            "infer_ms": round(elapsed, 1),
        })


def main():
    global _recognizer, _signs

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    inf = config["inference"]

    p = argparse.ArgumentParser(description="Servidor de la demo de reconocimiento LSA.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 para que la app del celular llegue por la red local")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=inf["threshold"])
    p.add_argument("--confirm-frames", type=int, default=inf["confirm_frames"])
    p.add_argument("--smoothing", type=int, default=inf["smoothing_window"])
    p.add_argument("--signs", action="store_true",
                   help="Cargar también el modelo de señas dinámicas (para familia.html)")
    p.add_argument("--sign-model", type=Path, default=None)
    p.add_argument("--sign-threshold", type=float, default=0.85)
    p.add_argument("--api-url", default="http://161.35.105.45",
                   help="signa-api, para las animaciones 3D. Vacío para desactivarlo.")
    p.add_argument("--api-token", default=os.environ.get("SIGNA_API_TOKEN"),
                   help="JWT ya emitido. Alternativa a --api-user/--api-password.")
    p.add_argument("--api-user", default=os.environ.get("SIGNA_API_USER"),
                   help="Usuario o email para POST /auth/login (o env SIGNA_API_USER).")
    p.add_argument("--api-password", default=os.environ.get("SIGNA_API_PASSWORD"),
                   help="Contraseña (o env SIGNA_API_PASSWORD).")
    args = p.parse_args()

    global _animations
    _animations = SignAnimations(args.api_url or None, args.api_token,
                                 args.api_user, args.api_password)

    print("Cargando modelo…")
    _recognizer = AlphabetRecognizer(
        model_path=args.model,
        threshold=args.threshold,
        confirm_frames=args.confirm_frames,
        smoothing_window=args.smoothing,
    )
    acc = _recognizer.meta.get("accuracy")
    print(f"  {_recognizer.model_path.name}"
          + (f" · test {acc:.1%}" if acc is not None else ""))
    print(f"  letras: {' '.join(_recognizer.labels)}")
    print(f"  umbral {args.threshold:.0%} · confirma a los {args.confirm_frames} frames")
    if _recognizer.thresholds:
        flojas = sorted(_recognizer.thresholds.items(), key=lambda kv: kv[1])[:5]
        print("  umbrales calibrados por letra · más permisivos: "
              + ", ".join(f"{l} {t:.2f}" for l, t in flojas))

    print("\nAnimaciones 3D del avatar:")
    if SIGNS_DIR.is_dir():
        n = len(list(SIGNS_DIR.glob("*.glb")))
        print(f"  {n} archivo(s) .glb en demo/static/signs/ (tienen prioridad)")
    if _animations.api_url:
        print(f"  signa-api: {_animations.api_url}")
        if args.api_user:
            ok = _animations.login()
            print(f"    login como {args.api_user}: "
                  + ("ok" if ok else f"FALLÓ — {_animations.last_error}"))
        elif args.api_token:
            print("    usando el token provisto")
        else:
            print("    sin credenciales · si el API pide auth, pasá "
                  "--api-user y --api-password (o SIGNA_API_USER / SIGNA_API_PASSWORD)")
    elif not SIGNS_DIR.is_dir():
        print("  ninguna fuente configurada → se muestra el placeholder")

    if args.signs:
        print("\nCargando modelo de señas dinámicas…")
        _signs = SignRecognizer(model_path=args.sign_model, threshold=args.sign_threshold)
        sacc = _signs.meta.get("accuracy")
        print(f"  {_signs.model_path.name}" + (f" · test {sacc:.1%}" if sacc else ""))
        print(f"  señas: {' '.join(_signs.labels)}")
    print()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.host not in ("127.0.0.1", "localhost"):
        import socket
        s_tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s_tmp.connect(("8.8.8.8", 80))
            lan = s_tmp.getsockname()[0]
        except Exception:
            lan = args.host
        finally:
            s_tmp.close()
        print(f"  Para la app del celular:  EXPO_PUBLIC_ML_URL=http://{lan}:{args.port}")

    print(f"  Nombre   →  http://localhost:{args.port}/nombre.html")
    # El paquete que se comparte lleva sólo nombre.html: no anunciar lo que no está.
    if (STATIC_DIR / "simple.html").is_file():
        print(f"  Simple   →  http://localhost:{args.port}/simple.html")
    if _signs is not None:
        print(f"  Familia  →  http://localhost:{args.port}/familia.html")
    print("\nCtrl+C para cortar.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando.")
    finally:
        server.server_close()
        _recognizer.close()
        if _signs is not None:
            _signs.close()


if __name__ == "__main__":
    main()
