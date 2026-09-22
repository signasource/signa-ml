#!/usr/bin/env bash
#
# Deja una carpeta de .glb lista para el motor 3D de la app (Filament).
#
# Los avatares tal como salen del exportador traen tres cosas que Filament no
# perdona, y una cuarta que no rompe nada pero se paga en cada cuadro:
#
#   1. Texturas en webp, que su cargador no sabe decodificar: el modelo carga
#      entero y se ve NEGRO, con un solo renglón en el log.
#   2. Un accessor sin datos —el canal que mueve los ojos—, que le hace
#      descartar la animación ENTERA sin avisar: el avatar se queda en la pose
#      del primer cuadro.
#   3. Esqueletos de más de 256 huesos (las letras M y N traen 574): eso no
#      devuelve error, ABORTA el proceso de la app.
#   4. Canales de animación que mueven huesos que no deforman nada, llaves
#      repetidas, vértices duplicados y nodos sueltos.
#
# Este script arregla las cuatro y además pasa las texturas a KTX2, que es el
# formato que la GPU lee comprimido: menos memoria de video y la carga más
# rápida que medimos (257 ms contra 383 del formato actual).
#
# ─── Qué hace falta ──────────────────────────────────────────────────────────
#
#   python3 con Pillow          pip install pillow
#   node y npm                  para las herramientas de glTF
#   la herramienta `ktx`        https://github.com/KhronosGroup/KTX-Software/releases
#                               (bajar el .deb y descomprimirlo alcanza; con
#                                --ktx se le indica dónde quedó)
#
# La primera vez, instalar las dependencias de node:
#
#   cd scripts/avatares && npm install
#
# ─── Cómo se usa ─────────────────────────────────────────────────────────────
#
#   ./preparar.sh --entrada ~/avatares-originales --salida ~/avatares-listos
#
#   # con la herramienta ktx en otro lado:
#   ./preparar.sh -e originales -s listos --ktx ~/KTX-Software/usr/bin
#
#   # dejando las texturas en su tamaño original (por omisión se achican a 1024):
#   ./preparar.sh -e originales -s listos --textura 0
#
# Procesa todos los .glb de la carpeta de entrada y escribe los convertidos en
# la de salida, con el mismo nombre. No toca los originales.
#
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARREGLAR="$AQUI/../arreglar_glb.py"
ENTRADA=""
SALIDA=""
TEXTURA=1024
KTX=""

ayuda() {
  sed -n '3,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--entrada) ENTRADA="$2"; shift 2 ;;
    -s|--salida) SALIDA="$2"; shift 2 ;;
    -t|--textura) TEXTURA="$2"; shift 2 ;;
    --ktx) KTX="$2"; shift 2 ;;
    -h|--ayuda|--help) ayuda ;;
    *) echo "opción desconocida: $1"; ayuda 1 ;;
  esac
done

[[ -z "$ENTRADA" || -z "$SALIDA" ]] && ayuda 1
[[ -d "$ENTRADA" ]] || { echo "no existe la carpeta de entrada: $ENTRADA"; exit 1; }

# La herramienta de Khronos es la que comprime a KTX2, y gltf-transform la
# busca en el PATH con el nombre `ktx`.
if [[ -n "$KTX" ]]; then
  export PATH="$KTX:$PATH"
  export LD_LIBRARY_PATH="${KTX%/bin}/lib:${LD_LIBRARY_PATH:-}"
fi
command -v ktx >/dev/null || {
  echo "falta la herramienta 'ktx' (KTX-Software). Ver el encabezado de este script."
  exit 1
}
[[ -d "$AQUI/node_modules" ]] || { echo "falta 'npm install' en $AQUI"; exit 1; }

mkdir -p "$SALIDA"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

gltf() { npx --yes @gltf-transform/cli@latest "$@" >/dev/null 2>&1; }

total=0
for archivo in "$ENTRADA"/*.glb; do
  [[ -e "$archivo" ]] || { echo "no hay .glb en $ENTRADA"; exit 1; }
  nombre="$(basename "$archivo")"
  antes=$(stat -c%s "$archivo")

  # 1. Huesos que la malla no usa. Es lo único que puede tumbar la app.
  node "$AQUI/podar.mjs" "$archivo" "$TMP/1.glb" | sed "s/^/  $nombre /" || cp "$archivo" "$TMP/1.glb"

  # 2. Canales que no mueven nada, llaves repetidas, vértices duplicados.
  node "$AQUI/afinar.mjs" "$TMP/1.glb" "$TMP/2.glb" | sed "s/^/  $nombre /"

  # 3. Texturas webp a un formato que el cargador entienda, y el accessor vacío
  #    que se lleva puesta la animación.
  python3 "$ARREGLAR" "$TMP/2.glb" "$TMP/3.glb" >/dev/null

  # 4. Texturas a 1024: arriba de eso no se nota en pantalla y la compresión de
  #    GPU, que ocupa lo mismo por píxel comprima lo que comprima, se dispara.
  if [[ "$TEXTURA" != "0" ]]; then
    gltf resize "$TMP/3.glb" "$TMP/4.glb" --width "$TEXTURA" --height "$TEXTURA"
  else
    cp "$TMP/3.glb" "$TMP/4.glb"
  fi

  # 5. KTX2 para las texturas y Draco para las mallas.
  gltf etc1s "$TMP/4.glb" "$TMP/5.glb"
  gltf draco "$TMP/5.glb" "$SALIDA/$nombre"

  despues=$(stat -c%s "$SALIDA/$nombre")
  printf "  %-16s %6.2f MB → %6.2f MB\n" "$nombre" "$(bc -l <<< "$antes/1048576")" "$(bc -l <<< "$despues/1048576")"
  total=$((total + 1))
done

echo "listo: $total archivos en $SALIDA"
echo
echo "Para comprobar uno antes de subirlo:"
echo "  npx @gltf-transform/cli inspect $SALIDA/<archivo>.glb   # huesos, animaciones, texturas"
echo "  npx @gltf-transform/cli validate $SALIDA/<archivo>.glb  # errores de formato"
