"""
Diagnóstico de las animaciones 3D: qué letras están cargadas y con qué nombre.

Corré esto una vez para saber si el abecedario existe en la base y bajo qué
convención de "significado" (`a`, `A`, `letra a`, …). Con eso se ajusta
SignAnimations.MEANING_TEMPLATES en demo/server.py si hiciera falta.

La contraseña se lee del entorno para que no quede en el historial del shell:

    export SIGNA_API_USER=tu@email
    export SIGNA_API_PASSWORD='tu-contraseña'
    python demo/check_signs.py

    python demo/check_signs.py --letters A,B,C --extra hola,madre
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import SignAnimations  # noqa: E402

ABC = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")


def main():
    p = argparse.ArgumentParser(description="¿Qué señas tienen animación cargada?")
    p.add_argument("--api-url", default=os.environ.get("SIGNA_API_URL", "http://161.35.105.45"))
    p.add_argument("--api-user", default=os.environ.get("SIGNA_API_USER"))
    p.add_argument("--api-password", default=os.environ.get("SIGNA_API_PASSWORD"))
    p.add_argument("--api-token", default=os.environ.get("SIGNA_API_TOKEN"))
    p.add_argument("--letters", default=",".join(ABC))
    p.add_argument("--extra", default="hola",
                   help="Significados sueltos para comprobar que la conexión anda")
    args = p.parse_args()

    api = SignAnimations(args.api_url, args.api_token, args.api_user, args.api_password)
    print(f"API: {args.api_url}")

    if args.api_user:
        print(f"Login como {args.api_user}: ", end="")
        if not api.login():
            print(f"FALLÓ — {api.last_error}")
            sys.exit(1)
        print("ok")
    elif not args.api_token:
        print("Sin credenciales: si el API pide auth, todo va a dar 403.\n")

    # Control: un significado que sabemos que existe.
    for meaning in [m.strip() for m in args.extra.split(",") if m.strip()]:
        path = f"/signs/{urllib.parse.quote(meaning)}/animation"
        payload, err = api._get_json(path)
        url = (payload or {}).get("animation_url") if isinstance(payload, dict) else None
        key = url.split("?")[0].split("signa-animations/")[-1] if url else None
        print(f"  control '{meaning}': " + (f"ok → {key}" if key else f"sin animación ({err or '404'})"))

    letters = [c.strip().upper() for c in args.letters.split(",") if c.strip()]
    print(f"\nProbando {len(letters)} letras con las convenciones "
          f"{', '.join(repr(t) for t in api.MEANING_TEMPLATES)}:\n")

    found, missing = [], []
    for letter in letters:
        hit = None
        for tpl in api.MEANING_TEMPLATES:
            meaning = tpl.format(lower=letter.lower(), upper=letter.upper())
            payload, err = api._get_json(
                f"/signs/{urllib.parse.quote(meaning)}/animation")
            if err:
                print(f"  {letter}: error — {err}")
                hit = "error"
                break
            url = (payload or {}).get("animation_url") if isinstance(payload, dict) else None
            if url:
                key = url.split("?")[0].split("signa-animations/")[-1]
                print(f"  {letter}: '{meaning}' → {key}")
                found.append(letter)
                hit = meaning
                break
        if hit is None:
            missing.append(letter)

    print(f"\nCon animación: {len(found)}/{len(letters)}"
          + (f" → {' '.join(found)}" if found else ""))
    if missing:
        print(f"Sin animación: {' '.join(missing)}")
        print("\nSi sabés que están cargadas con otro nombre, decímelo y ajusto "
              "SignAnimations.MEANING_TEMPLATES en demo/server.py.")


if __name__ == "__main__":
    main()
