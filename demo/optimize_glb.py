"""
Aliviana un .glb para el picture-in-picture de la demo.

Los avatares que sirve signa-api pesan ~61 MB, de los cuales ~39 MB son texturas
PNG sin comprimir (normal maps de 4K incluidos). El navegador descarga eso rápido
pero se queda minutos decodificando imagen por imagen, y `model-viewer` nunca
llega a emitir `load`: el PiP se queda en el placeholder para siempre.

El PiP mide 104x138 px, así que esas texturas son varios órdenes de magnitud más
grandes de lo que se llega a ver. Acá se reescalan y se pasan a JPEG (PNG sólo si
la imagen usa alfa de verdad), que es lo único que hace falta para que el modelo
cargue en segundos. La geometría y la animación quedan intactas.

    python demo/optimize_glb.py demo/static/signs/*.glb --size 512

Con --backup los originales se guardan al lado como <nombre>.full.glb.bak.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
from pathlib import Path

from PIL import Image

JSON_CHUNK, BIN_CHUNK = b"JSON", b"BIN\x00"


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path.name}: no es un glTF binario v2")
    gltf, buf, off = None, b"", 12
    while off < len(data):
        length, kind = struct.unpack_from("<I4s", data, off)
        chunk = data[off + 8: off + 8 + length]
        if kind == JSON_CHUNK:
            gltf = json.loads(chunk)
        elif kind == BIN_CHUNK:
            buf = chunk
        off += 8 + length + (-length % 4)
    if gltf is None:
        raise ValueError(f"{path.name}: sin chunk JSON")
    return gltf, buf


def write_glb(path: Path, gltf: dict, buf: bytes) -> None:
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)                  # el spec pide chunks alineados a 4
    buf += b"\x00" * (-len(buf) % 4)
    total = 12 + 8 + len(js) + 8 + len(buf)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), JSON_CHUNK)); f.write(js)
        f.write(struct.pack("<I4s", len(buf), BIN_CHUNK)); f.write(buf)


def shrink(raw: bytes, size: int, quality: int) -> tuple[bytes, str]:
    """Reescala y recomprime una textura. Devuelve (bytes, mimeType)."""
    im = Image.open(io.BytesIO(raw))
    im.load()
    # Alfa que en realidad es opaco: se descarta y la imagen puede ir a JPEG.
    has_alpha = im.mode in ("RGBA", "LA") and im.getchannel("A").getextrema()[0] < 255
    if max(im.size) > size:
        im = im.resize(
            tuple(max(1, round(d * size / max(im.size))) for d in im.size),
            Image.LANCZOS,
        )
    out = io.BytesIO()
    if has_alpha:
        im.convert("RGBA").save(out, "PNG", optimize=True)
        return out.getvalue(), "image/png"
    im.convert("RGB").save(out, "JPEG", quality=quality, optimize=True, progressive=False)
    return out.getvalue(), "image/jpeg"


def optimize(path: Path, size: int, quality: int, backup: bool) -> None:
    gltf, buf = read_glb(path)
    images = gltf.get("images", [])
    if any("uri" in im for im in images):
        raise ValueError(f"{path.name}: tiene texturas externas, no soportado")

    views = gltf["bufferViews"]
    replaced: dict[int, bytes] = {}
    for im in images:
        idx = im["bufferView"]
        view = views[idx]
        raw = buf[view["byteOffset"]: view["byteOffset"] + view["byteLength"]]
        new, mime = shrink(raw, size, quality)
        replaced[idx] = new
        im["mimeType"] = mime

    # Se reconstruye el buffer entero: las vistas que no son imágenes se copian
    # tal cual (respetando el orden de índices) y sólo cambian sus offsets.
    out = bytearray()
    for i, view in enumerate(views):
        out += b"\x00" * (-len(out) % 4)                      # accessors piden 4 bytes
        data = replaced.get(i)
        if data is None:
            data = buf[view["byteOffset"]: view["byteOffset"] + view["byteLength"]]
        view["byteOffset"] = len(out)
        view["byteLength"] = len(data)
        out += data
    gltf["buffers"] = [{"byteLength": len(out)}]

    before = path.stat().st_size
    if backup:
        shutil.copy2(path, path.with_suffix(".glb.bak"))
    write_glb(path, gltf, bytes(out))
    after = path.stat().st_size
    print(f"  {path.name}: {before/1e6:.1f} MB → {after/1e6:.1f} MB "
          f"({100 - after * 100 / before:.0f}% menos, {len(images)} texturas)")


def main() -> None:
    p = argparse.ArgumentParser(description="Aliviana los .glb del avatar para la demo.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--size", type=int, default=512, help="lado máximo de cada textura")
    p.add_argument("--quality", type=int, default=85, help="calidad JPEG")
    p.add_argument("--backup", action="store_true", help="guardar el original como .glb.bak")
    args = p.parse_args()
    for f in args.files:
        optimize(f, args.size, args.quality, args.backup)


if __name__ == "__main__":
    main()
