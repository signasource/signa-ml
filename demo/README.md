# Demo de reconocimiento — cómo levantarla y grabarla

Una home y dos ejercicios, hechos a partir de los prototipos de Claude Design,
corriendo con **los mismos modelos, detectores y umbrales que la app nativa**.

| pantalla | archivo | modelo | qué hace |
|---|---|---|---|
| Menú | `index.html` (`/`) | — | elegís el ejercicio; todo vuelve acá |
| Deletreá tu nombre | `nombre.html` | `signa_alphabet_v5.tflite` (abecedario) | escribís tu nombre y te hace señar letra por letra |
| Señas con movimiento | `familia.html` | `signa_model_v9.tflite` (LSTM dinámico) | papá, mamá, hermano, amigo |
| Reconocimiento simple | `simple.html` | el del abecedario | identifica letras sueltas (no está en el menú) |

El flujo es cerrado, pensado para la feria: desde la home se entra a un
ejercicio, la flecha de atrás o `Esc` vuelven al menú, y al terminar un
ejercicio la pantalla final ofrece «Volver al menú» o repetirlo.

---

## Levantarla

```bash
cd ~/Repos/signa/signa-ml
source .venv/bin/activate
python demo/server.py --signs
```

y abrí **http://localhost:8000/**. Vas a ver algo así:

```
Cargando modelo…
  signa_alphabet_v5.tflite · test 79.6%
  letras: A B C D E F G H I J K L M N Ñ O P Q R S T U V W X Y
  umbrales calibrados por letra · más permisivos: …

Cargando modelo de señas dinámicas…
  signa_model_v9.tflite · test 96.9%
  señas: hermano reposo amigo papa mama
  umbrales: hermano 0.78, amigo 0.86, papa 0.41, mama 0.61

  Menú     →  http://localhost:8000/
  Nombre   →  http://localhost:8000/nombre.html
  Familia  →  http://localhost:8000/familia.html
```

Dale permiso de cámara. **Tiene que ser `localhost`** — los navegadores no dan
acceso a la cámara en `http://` salvo en localhost.

Sin `--signs` arranca más rápido, pero la opción de señas con movimiento queda
deshabilitada en el menú.

Los detectores de MediaPipe se leen de `../signa-mobile/assets/mediapipe/`
(`hand_landmarker.task`, `pose_landmarker.task`): hace falta tener clonado
signa-mobile al lado, o pasar la carpeta con `--mediapipe-dir`.

Opciones útiles:

```bash
python demo/server.py --signs --port 8800        # otro puerto
python demo/server.py --signs --mediapipe-dir <carpeta con los .task>
```

Opciones por URL:

- `nombre.html?celebration=fiesta` — final con confeti en vez de "lección completa"
- `familia.html?signs=papa,mama` — practicar sólo algunas señas
- `simple.html?signs=A,B,L,O,V,Y` — qué letras mostrar

---

## Cómo corre por dentro

```
navegador  ──JPEG (POST /predict o /predict_sign)──▶  demo/server.py
            ◀──{letra o seña, confianza, landmarks}──   ├─ HandLandmarker + PoseLandmarker + abecedario
                                                        └─ HandLandmarker + PoseLandmarker + LSTM
```

El navegador se queda con la cámara, el diseño y las animaciones; Python se queda
con los modelos. El servidor usa los mismos runners que los scripts de consola
(`src/inference/alphabet_runner.py` y `sign_runner.py`), así que lo que se ve en
el video sale del mismo código que produjo los números que reportamos — no hay
features reimplementadas en JavaScript que puedan desincronizarse.

**El abecedario usa los mismos detectores que la app.** `HandLandmarker` y
`PoseLandmarker` de MediaPipe Tasks, con los `.task` de
`signa-mobile/assets/mediapipe`, que son también con los que se armó el dataset.
La cara (para la posición de la mano) sale de la pose, igual que en el teléfono.
Si esos archivos están en otro lado: `--mediapipe-dir <carpeta>`. Antes la demo
usaba las soluciones legacy de MediaPipe y un modelo entrenado con ellas: lo que
se veía acá no era lo que veía la app.

**Las señas dinámicas siguen paso por paso a la app** (`Ventana.kt`,
`Reconocedor.kt`, `Confirmador.kt`): mismos detectores, ventana de 2,5 s
remuestreada por tiempo a 30 pasos, cuadros sin manos afuera, pose anulada
antes del modelo, verificación contra las señas que muestra la pantalla con el
umbral calibrado de cada una, piso de movimiento y confirmación por el 60% de
los últimos 700 ms. Antes la demo usaba Holistic legacy, un umbral único de
0.85 y rachas de frames: el v9 recibía otra cosa que en el teléfono.

**Sólo se reconoce lo que se ve en el viewport.** La cámara capta más de lo que
muestra el recuadro (el video está con `object-fit: cover`); lo que queda fuera
de cuadro se manda en negro, así una mano al costado no dispara nada.

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
`models/exports/signa_alphabet_v5_thresholds.json`, que el runner carga solo.

Resultado del v5 sobre fotos de señantes que el modelo nunca vio, **en las dos
orientaciones** (la foto original y espejada, que es como llega la cámara):

|  | identificación (26 clases) | verificación (por letra) |
|---|---|---|
| media | 83.0% | 88.1% |
| **peor letra** | 23% (Q) | **61% (I)** |
| letras por debajo de 70% | 3 (I, P, Q) | **2 (I, Q)** |

La Q se hace con dos manos y el reconocedor mira una. La I comparte la forma de
la mano con la T y sólo se distingue por la altura: en vivo la separa la regla
de ubicación del runner (`LOCATION_PAIRS`), que estos números no incluyen.

Estos números no se comparan uno a uno con los de versiones anteriores: aquellos
se medían sólo sobre las fotos sin espejar, que son más fáciles. La comparación
justa —mismas fotos, mismos folds— está en el informe del v5: con las fotos
espejadas en el entrenamiento, la verificación sobre imagen espejada sube de
83.2% a 87.8% y la T de 40% a 73%.

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

## La seña en 3D y el toggle de trackeo (`nombre.html` y `familia.html`)

Las dos escenas son la misma pantalla —`familia.html` está armada sobre
`nombre.html`— y sólo cambia qué se reconoce: una letra por vez o una seña
por vez. En las señas, la barra blanca del borde inferior del viewport muestra
cómo se llena la ventana de 2,5 s antes de poder decidir.

Dentro del viewport hay dos controles:

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
| `Esc` | volver al menú |
| `espacio` | Nombre y Familia: dar por acertada la letra/seña actual · Simple: play/pausa |
| letra | `nombre.html`: si es la pedida, darla por acertada · `simple.html`: forzar esa detección |

Las teclas de forzado son la red de seguridad: si en la toma algo no coopera,
completás a mano sin cortar la grabación. El panel de control muestra el top-3
con las confianzas, el umbral de la letra pedida, los fps y los ms de
inferencia — sirve para ensayar, y se oculta con `H`.

---

## Consejos para la toma

- **Luz pareja y fondo liso.** MediaPipe pierde la mano con contraluz.
- **Mano completa en cuadro**, sin que se corte la muñeca.
- **En `familia.html` necesitás torso y manos en cuadro**: los hombros son la
  referencia con la que se normaliza la seña. Alejate un poco de la cámara.
- **Las señas dinámicas miran los últimos 2,5 s.** La barrita blanca abajo del
  video muestra cuánto de esa ventana ya está lleno (los cuadros sin manos no
  cuentan); esperá a que esté completa antes de señar. Una seña se confirma
  cuando pasa su umbral la mayor parte de 700 ms, y hace falta movimiento: una
  mano quieta no dispara nada. Es exactamente la lógica de la app.
- **Volvé a reposo entre seña y seña.** El modelo tiene una clase `reposo` y la
  usa para saber que terminaste una y empieza otra.
- Grabá con el navegador en pantalla completa: la escena mide 390×844 y ya trae
  el marco de teléfono con sombra. Se escala sola para ocupar el alto de la
  ventana, así que entra igual en la notebook que en un monitor de 1080.
- En `nombre.html` sólo cuenta la letra pedida: si mientras armás la Ñ el
  modelo pasa por una B, no se marca nada en rojo.
