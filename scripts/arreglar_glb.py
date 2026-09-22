#!/usr/bin/env python3
"""
Deja los .glb de los avatares en condiciones de que los lea cualquier visor.

Los archivos que están publicados traen dos cosas que el visor web tolera y los
cargadores estrictos no:

  1. Las texturas en webp (`EXT_texture_webp`). three.js las decodifica porque
     se las pide al navegador; Filament no sabe, y carga el modelo entero sin
     texturas —negro— sin un solo error.
  2. Un canal de animación de `weights` cuyo accessor no tiene `bufferView`, o
     sea que no apunta a ningún dato. three.js lo trata como ceros y sigue;
     Filament descarta la animación ENTERA y el avatar se queda en la pose del
     primer cuadro.

La app hoy arregla las dos cosas en el teléfono la primera vez que ve cada
avatar, y eso cuesta ~2,3 s. Corriendo esto una vez y volviendo a subir los
archivos, ese costo desaparece para siempre y para todos.

Uso:
    python scripts/arreglar_glb.py entrada.glb salida.glb
    python scripts/arreglar_glb.py --bucket madre padre hermano --salida arreglados/
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import urllib.request
from pathlib import Path

from PIL import Image

BUCKET = "https://pub-f40a1de4d1fc46b0b6f07299847c66e0.r2.dev/lsa"
JSON_CHUNK, BIN_CHUNK = 0x4E4F534A, 0x004E4942
WEBP = "EXT_texture_webp"

# Bytes por componente y componentes por tipo, para poder darle su tamaño en
# ceros a un accessor que no tiene de dónde leer.
TAM_COMPONENTE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
COMPONENTES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def leer(datos: bytes) -> tuple[dict, bytes]:
    if datos[:4] != b"glTF":
        raise ValueError("no es un .glb")
    doc, binario, off = None, b"", 12
    while off < len(datos):
        largo, tipo = struct.unpack_from("<II", datos, off)
        trozo = datos[off + 8 : off + 8 + largo]
        if tipo == JSON_CHUNK:
            doc = json.loads(trozo)
        elif tipo == BIN_CHUNK:
            binario = trozo
        off += 8 + largo + (-largo % 4)
    return doc, binario


def escribir(doc: dict, binario: bytes) -> bytes:
    # Cada trozo tiene que empezar en un múltiplo de 4: el JSON se rellena con
    # espacios y el binario con ceros.
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binario += b"\0" * (-len(binario) % 4)
    cabecera = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binario))
    return (cabecera
            + struct.pack("<II", len(js), JSON_CHUNK) + js
            + struct.pack("<II", len(binario), BIN_CHUNK) + binario)


def sin_webp(doc: dict, binario: bytes, calidad: int) -> bytes:
    vistas = doc["bufferViews"]
    extra = bytearray()

    for imagen in doc.get("images", []):
        if imagen.get("mimeType") != "image/webp":
            continue
        vista = vistas[imagen["bufferView"]]
        desde = vista.get("byteOffset", 0)
        cruda = binario[desde : desde + vista["byteLength"]]
        mapa = Image.open(io.BytesIO(cruda))

        # JPEG salvo que la textura tenga transparencia —el pelo la usa—: en
        # PNG los mismos 1024x1024 pesan varias veces más.
        con_alfa = mapa.mode in ("RGBA", "LA") and mapa.getchannel("A").getextrema()[0] < 255
        salida = io.BytesIO()
        if con_alfa:
            mapa.convert("RGBA").save(salida, "PNG", optimize=True)
        else:
            mapa.convert("RGB").save(salida, "JPEG", quality=calidad, optimize=True)

        vistas.append({"buffer": 0, "byteOffset": len(binario) + len(extra), "byteLength": salida.tell()})
        extra += salida.getvalue()
        imagen["bufferView"] = len(vistas) - 1
        imagen["mimeType"] = "image/png" if con_alfa else "image/jpeg"

    # Las texturas apuntaban a la imagen a través de la extensión; ahora lo
    # hacen por el camino de siempre.
    for textura in doc.get("textures", []):
        ext = textura.get("extensions", {})
        if WEBP in ext:
            textura["source"] = ext.pop(WEBP)["source"]
            if not ext:
                textura.pop("extensions")
    for campo in ("extensionsUsed", "extensionsRequired"):
        if campo in doc:
            doc[campo] = [e for e in doc[campo] if e != WEBP]
            if not doc[campo]:
                doc.pop(campo)

    return bytes(binario + extra)


def rellenar_accessores(doc: dict, binario: bytes) -> bytes:
    """Los accessors de Draco tampoco tienen bufferView, pero ésos los rellena
    la extensión al descomprimir la malla: hay que saltearlos."""
    de_draco = set()
    for malla in doc.get("meshes", []):
        for prim in malla["primitives"]:
            de_draco.update(prim["attributes"].values())
            if "indices" in prim:
                de_draco.add(prim["indices"])
            for objetivo in prim.get("targets", []):
                de_draco.update(objetivo.values())

    extra = bytearray()
    for i, accessor in enumerate(doc.get("accessors", [])):
        if "bufferView" in accessor or "sparse" in accessor or i in de_draco:
            continue
        largo = (accessor["count"] * COMPONENTES[accessor["type"]]
                 * TAM_COMPONENTE[accessor["componentType"]])
        doc["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(binario) + len(extra), "byteLength": largo}
        )
        extra += b"\0" * largo
        accessor["bufferView"] = len(doc["bufferViews"]) - 1
        print(f"    accessor {i} sin datos: se le dan {largo} bytes en cero")

    return bytes(binario + extra)


def arreglar(datos: bytes, calidad: int) -> bytes:
    doc, binario = leer(datos)
    binario = sin_webp(doc, binario, calidad)
    binario = rellenar_accessores(doc, binario)
    doc["buffers"][0]["byteLength"] = len(binario)
    return escribir(doc, binario)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("entrada", nargs="?", type=Path)
    p.add_argument("salida", nargs="?", type=Path)
    p.add_argument("--bucket", nargs="*", help="nombres a bajar del bucket público")
    p.add_argument("--salida-dir", type=Path, default=Path("arreglados"))
    p.add_argument("--calidad", type=int, default=90, help="calidad del JPEG")
    args = p.parse_args()

    trabajos: list[tuple[str, bytes, Path]] = []
    if args.bucket:
        args.salida_dir.mkdir(parents=True, exist_ok=True)
        for nombre in args.bucket:
            url = f"{BUCKET}/{nombre}.glb"
            print(f"bajando {url}")
            # El bucket rechaza el User-Agent por omisión de urllib con un 403.
            pedido = urllib.request.Request(url, headers={"User-Agent": "signa-glb/1.0"})
            with urllib.request.urlopen(pedido) as r:
                trabajos.append((nombre, r.read(), args.salida_dir / f"{nombre}.glb"))
    elif args.entrada and args.salida:
        trabajos.append((args.entrada.name, args.entrada.read_bytes(), args.salida))
    else:
        p.error("pasá entrada y salida, o --bucket con nombres")

    for nombre, datos, destino in trabajos:
        arreglado = arreglar(datos, args.calidad)
        destino.write_bytes(arreglado)
        print(f"  {nombre}: {len(datos)/1e6:.2f} MB → {len(arreglado)/1e6:.2f} MB  ({destino})")


if __name__ == "__main__":
    main()
