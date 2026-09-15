# Demo de reconocimiento — cómo levantarla y grabarla

Tres escenas, hechas a partir de los prototipos de Claude Design, corriendo con
los modelos reales del repo.

| escena | archivo | modelo | qué hace |
|---|---|---|---|
| Deletreá tu nombre | `nombre.html` | `signa_alphabet_v1.tflite` (abecedario) | escribís tu nombre y te hace señar letra por letra |
| Práctica libre · Familia | `familia.html` | `signa_model_v3.tflite` (LSTM dinámico) | reconoce señas con movimiento: papá, mamá, hermanos, casa |
| Reconocimiento simple | `simple.html` | `signa_alphabet_v1.tflite` | identifica letras sueltas, sin pedir ninguna en particular |

---

## Levantarlas

Un solo comando levanta el servidor y las tres escenas.

```bash
cd ~/Repos/signa/signa-ml
source .venv/bin/activate

# con los dos modelos (abecedario + señas dinámicas)
python demo/server.py --signs
```

Vas a ver algo así:

```
Cargando modelo…
  signa_alphabet_v1.tflite · test 81.6%
  letras: A B C D E F G H I J K L M N Ñ O P Q R S T U V W X Y
  umbrales calibrados por letra · más permisivos: …

Cargando modelo de señas dinámicas…
  signa_model_v3.tflite · test 97.6%
  señas: gracias hermanos reposo casa nombre estudiar entender repetir gato papa mama computadora lengua_de_senas

  Nombre   →  http://localhost:8000/nombre.html
  Simple   →  http://localhost:8000/simple.html
  Familia  →  http://localhost:8000/familia.html
```

Abrí esas URLs en el navegador y dale permiso de cámara. **Tiene que ser
`localhost`** — los navegadores no dan acceso a la cámara en `http://` salvo en
localhost.

Si sólo vas a usar el abecedario, `python demo/server.py` (sin `--signs`) arranca
más rápido y no carga el modelo pesado.

Opciones útiles:

```bash
python demo/server.py --signs --port 8800        # otro puerto
python demo/server.py --threshold 0.65           # abecedario más permisivo
python demo/server.py --signs --sign-threshold 0.75   # señas más permisivas
```

Opciones por URL:

- `nombre.html?celebration=fiesta` — final con confeti en vez de "lección completa"
- `familia.html?signs=papa,mama,casa,gracias` — qué señas mostrar
- `simple.html?signs=A,B,L,O,V,Y` — qué letras mostrar

---

## Cómo corre por dentro

```
navegador  ──JPEG (POST /predict o /predict_sign)──▶  demo/server.py
            ◀──{letra o seña, confianza, landmarks}──   ├─ MediaPipe Hands + abecedario
                                                        └─ MediaPipe Holistic + LSTM
```

El navegador se queda con la cámara, el diseño y las animaciones; Python se queda
con los modelos. El servidor usa los mismos runners que los scripts de consola
(`src/inference/alphabet_runner.py` y `sign_runner.py`), así que lo que se ve en
el video sale del mismo código que produjo los números que reportamos — no hay
features reimplementadas en JavaScript que puedan desincronizarse.

No agrega dependencias: `http.server` de la stdlib alcanza en localhost.

---

## Las 26 letras, sin evitar ninguna

`nombre.html` corre en **modo verificación**, no identificación.

La diferencia importa mucho. Identificar es "¿cuál de las 26 letras es esta?".
Verificar es "¿esto es una A?" — y en el deletreo la app **ya sabe qué letra
pidió**, así que sólo tiene que verificar. Es un problema bastante más fácil, y
es lo que permite que todas las letras entren en juego, incluso las que en
identificación pierden contra una vecina parecida.

Cada letra tiene su propio umbral, calibrado sobre probabilidades out-of-fold:

```bash
python scripts/calibrate_alphabet.py
```

Un umbral por letra, y no uno solo para todas, porque las letras no tienen la
misma confianza típica: la Y sale con 99% y la Q con 40%. Con un umbral único, o
la Y acepta cualquier cosa o la Q no se acepta nunca. Los resultados quedan en
`reports/alphabet_thresholds.txt` y los umbrales en
`models/exports/signa_alphabet_v1_thresholds.json`, que el runner carga solo.

Resultado sobre fotos de señantes que el modelo nunca vio:

|  | identificación (26 clases) | verificación (por letra) |
|---|---|---|
| media | 82.9% | 89% |
| **peor letra** | 50% (Ñ) | **70% (Ñ)** |
| letras por debajo de 70% | 6 | **ninguna** |

Ese "ninguna" es lo que hace que no haya que evitar ninguna letra.

Hay un piso duro de 0.08 en los umbrales, a propósito: un umbral cerca de cero
aceptaría cualquier mano y la práctica sería un placebo — la app diría
"¡correcto!" hagas lo que hagas. Las letras más permisivas (Ñ, Q, I, W) tienen
precisión 41-61%: se aceptan fácil cuando las hacés bien, pero también aceptan
alguna vecina parecida. El reporte las marca.

**Lo que sigue sin poder hacer un modelo estático:** varias letras del LSA se
distinguen por movimiento, no por la forma de la mano. El caso más claro es
N / Ñ. En modo verificación pedirte una Ñ funciona; lo que no puede es
diferenciarlas si hacés la otra. Es un límite de clasificar imágenes fijas, no
del entrenamiento.

Se evaluó agrupar los pares más confundidos (I/T, Q/X, N/Ñ, C/E, V/W) y aceptar
cualquiera de los dos: sube el recall entre 0 y 21 puntos según el par, pero
aceptar una T cuando pediste una I le enseñaría la letra equivocada a quien está
practicando. No se aplicó.

**Lo que más sube esto:** grabar fotos propias con
`python scripts/collect_alphabet.py --profile <vos>` y re-entrenar. Las 15
fuentes actuales son screenshots de videos de terceros; sumar la cámara y la
mano que van a salir en el video ataca justo las letras flojas.

**Z no está**: el Tracker la marca `FALSE` en las 15 fuentes, así que no hay ni
una foto. Para sumarla: grabala con `python scripts/collect_alphabet.py
--profile <vos> --letters Z`, agregá `- Z` a `configs/alphabet_config.yaml` y
re-corré el pipeline.

---

## La seña en 3D y el toggle de trackeo (`nombre.html`)

Dentro del viewport hay dos controles nuevos:

- **Toggle «Trackeo»** (abajo a la izquierda) — enciende y apaga el dibujo de
  los landmarks sobre la mano. El reconocimiento sigue corriendo igual: sólo se
  deja de dibujar. Para grabar queda mejor apagado. La tecla `L` hace lo mismo.
- **Picture-in-picture de la seña** — un avatar 3D haciendo la letra que se está
  pidiendo. Se **arrastra** y al soltarlo se acomoda en la esquina más cercana;
  **tocándolo se agranda** a todo el viewport, y ahí el arrastre pasa a rotar el
  modelo (con la X se vuelve a achicar).

### De dónde sale el modelo 3D

Se replica lo que hace la app, en este orden:

1. **`demo/static/signs/<LETRA>.glb`** — si existe el archivo, gana. Sirve para
   trabajar sin backend.
2. **signa-api** — devuelve la URL prefirmada del `.glb` que vive en R2:
   - primero `POST /signs/animations` (en lote), que es lo que usa
     `animationPreload.ts` en la app: una sola ida y vuelta para todo el nombre;
   - lo que quede sin resolver se reintenta con
     **`GET /signs/{meaning}/animation`**, que devuelve
     `{signId, animationUrl, expiresInSeconds}` — `animationUrl` ya viene
     firmada y es la URL final del modelo.

   Todo se cachea. Las URLs prefirmadas vencen (15 min por defecto), así que la
   caché se guarda por 10 y se vuelve a pedir.
3. **Placeholder** — tarjeta rayada con la letra, como `SignPlaceholder`.

El `.glb` se muestra con `<model-viewer>`, el mismo componente y la misma
versión que usa `GlbAnimationView` en la app, con el mismo encuadre de cámara
(apunta al pecho, `fieldOfView` 15°).

El API desplegado no corre con perfil `local`, así que `/signs` **pide
autenticación** (devuelve 403 sin token). El servidor se loguea solo con
`POST /auth/login` y renueva el token cuando vence — un 401/403 dispara un
re-login y un reintento, así que una sesión de grabación larga no se corta a los
15 minutos.

```bash
# credenciales por entorno: no quedan en el historial del shell
export SIGNA_API_USER=tu@email
export SIGNA_API_PASSWORD='tu-contraseña'
python demo/server.py

# o con un JWT ya emitido
python demo/server.py --api-token <JWT>

# apuntar a otro servidor / desactivar el API
python demo/server.py --api-url http://localhost:8080
python demo/server.py --api-url ""     # sólo archivos locales
```

Por defecto apunta a `http://161.35.105.45`.

### ¿Qué letras están cargadas?

```bash
python demo/check_signs.py
```

Se loguea, prueba las 27 letras contra las convenciones que conoce y lista
cuáles tienen animación y con qué `meaning`. Si están cargadas con otro nombre,
se ajusta `SignAnimations.MEANING_TEMPLATES` en `demo/server.py`.

**Dos cosas a tener en cuenta:**

- El pedido al API **pasa por este servidor**, no por el navegador. signa-api
  levanta con `.cors(disable)`, así que una página en `localhost:8000` no puede
  llamarlo directo. El `.glb` en sí lo baja el navegador de R2, que sí tiene
  CORS — igual que el WebView en la app.
- Con el perfil `local` el API deja pasar todo; en cualquier otro, `/signs`
  pide autenticación y hace falta `--api-token`.
- El API responde en **snake_case** (`JacksonConfig` fija
  `PropertyNamingStrategies.SNAKE_CASE`), así que el campo es `animation_url`,
  no `animationUrl` como sugiere el record de Java. Los objetos viven en
  `signa-animations/lsa/<meaning>.glb` y la URL viene prefirmada, con 900
  segundos de validez.
- No hay una convención establecida para el «significado» de una letra: el
  contenido del curso usa palabras en minúscula (`hola`, `madre`). El servidor
  prueba `a`, `A` y `letra a`. Usá `demo/check_signs.py` para ver cuál aplica.

---

## Teclas (las tres escenas)

| tecla | acción |
|---|---|
| `H` | mostrar/ocultar el panel de control — **ocultalo antes de grabar** |
| `L` | mostrar/ocultar los landmarks (igual que el toggle «Trackeo») |
| `R` | reiniciar la escena |
| `espacio` | Nombre: dar por acertada la letra actual · Simple y Familia: play/pausa |
| letra | `nombre.html` / `simple.html`: forzar esa detección |
| `1`–`9` | `familia.html`: forzar la seña n-ésima de la lista |

Las teclas de forzado son la red de seguridad: si en la toma algo no coopera,
completás a mano sin cortar la grabación. El panel de control muestra el top-3
con las confianzas, el umbral de la letra pedida, los fps y los ms de
inferencia — sirve para ensayar, y se oculta con `H`.

---

## Consejos para la toma

- **Luz pareja y fondo liso.** MediaPipe pierde la mano con contraluz.
- **Mano completa en cuadro**, sin que se corte la muñeca.
- **En `familia.html` necesitás torso y brazos**, no sólo la mano: el modelo
  dinámico usa Holistic. Alejate un poco de la cámara.
- **Las señas dinámicas necesitan 30 frames** antes de poder decidir. La barrita
  blanca abajo del video muestra cómo se llena el buffer; esperá a que esté
  completa antes de señar. Después de cada acierto se vacía a propósito, para
  que la seña anterior no se vuelva a disparar sola.
- **Volvé a reposo entre seña y seña.** El modelo tiene una clase `reposo` y la
  usa para saber que terminaste una y empieza otra.
- Grabá con el navegador en pantalla completa: la escena mide 390×844 y ya trae
  el marco de teléfono con sombra, así que queda bien tal cual.
