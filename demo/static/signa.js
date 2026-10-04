/*
 * Base compartida por las dos escenas de la demo.
 *
 * Reparto de trabajo: el navegador tiene la cámara y el dibujo; el servidor
 * local tiene MediaPipe y el .tflite. Acá está el pegamento.
 */

const LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],
               [10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],
               [18,19],[19,20],[0,17]];
const TIPS = [4, 8, 12, 16, 20];

export const COLORS = {
  idle:   "#7857FF",
  hit:    "#4CA65C",
  wrong:  "#E14E22",
};

/* ── Cámara ────────────────────────────────────────────────────────────── */

export async function startCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: 720 }, audio: false,
  });
  video.srcObject = stream;
  video.muted = true;
  await video.play().catch(() => {});
  return stream;
}

/* ── Envío de frames al servidor ───────────────────────────────────────── */

export class FrameSender {
  /*
   * Manda un frame por vez y espera la respuesta antes del siguiente. Eso hace
   * que el ritmo se auto-regule al que el servidor puede sostener: sin cola,
   * sin latencia acumulada, sin frames viejos pisando resultados nuevos.
   */
  constructor(video, { width = 480, quality = 0.7, onResult, endpoint = "/predict" } = {}) {
    this.video = video;
    this.width = width;
    this.quality = quality;
    this.onResult = onResult;
    this.endpoint = endpoint;
    // La escena puede pedir una letra concreta: con eso el servidor pasa a
    // modo verificación ("¿esto es una A?") en vez de identificación.
    this.target = null;
    // Con el reconocimiento en pausa se siguen mandando frames, pero sólo para
    // dibujar el esqueleto: el servidor no clasifica ni acumula nada.
    this.trackOnly = false;
    this.canvas = document.createElement("canvas");
    this.running = false;
    this.fps = 0;
    this._last = 0;
  }

  start() { if (!this.running) { this.running = true; this._tick(); } }
  stop() { this.running = false; }

  async reset() {
    try { await fetch("/reset", { method: "POST" }); } catch (e) {}
  }

  async _tick() {
    while (this.running) {
      const blob = await this._grab();
      if (!blob) { await new Promise((r) => setTimeout(r, 60)); continue; }
      try {
        const q = this.trackOnly ? "track=1"
          : this.target ? `target=${encodeURIComponent(this.target)}` : "";
        const url = q ? `${this.endpoint}?${q}` : this.endpoint;
        const res = await fetch(url, { method: "POST", body: blob });
        const data = await res.json();
        const now = performance.now();
        if (this._last) this.fps = 1000 / (now - this._last);
        this._last = now;
        if (this.onResult) this.onResult(data);
      } catch (e) {
        await new Promise((r) => setTimeout(r, 250));
      }
    }
  }

  _grab() {
    const v = this.video;
    if (!v.videoWidth) return Promise.resolve(null);
    const w = this.width;
    const h = Math.round((v.videoHeight / v.videoWidth) * w);
    if (this.canvas.width !== w) { this.canvas.width = w; this.canvas.height = h; }
    const ctx = this.canvas.getContext("2d");
    // Espejamos acá, igual que cv2.flip en predict_alphabet_realtime.py: así el
    // modelo ve lo mismo que la persona y los landmarks vuelven ya en
    // coordenadas de pantalla, sin tener que invertirlos después.
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0, w, h);
    ctx.restore();

    // Sólo se reconoce lo que se ve. El <video> está con object-fit:cover, así
    // que la cámara capta más de lo que muestra el viewport, y MediaPipe
    // agarraba manos que estaban fuera de cuadro (alguien al lado, la otra mano
    // en el borde). Lo que queda afuera se pinta de negro en vez de recortarlo:
    // el frame conserva su tamaño, así que los landmarks siguen viniendo en
    // coordenadas del video entero y el dibujo y los modelos no cambian.
    const vis = visibleRegion(v);
    if (vis) {
      const x0 = vis.x * w, y0 = vis.y * h, x1 = x0 + vis.w * w, y1 = y0 + vis.h * h;
      ctx.fillStyle = "#000";
      if (x0 > 0) { ctx.fillRect(0, 0, Math.ceil(x0), h); ctx.fillRect(Math.floor(x1), 0, w, h); }
      if (y0 > 0) { ctx.fillRect(0, 0, w, Math.ceil(y0)); ctx.fillRect(0, Math.floor(y1), w, h); }
    }
    return new Promise((r) => this.canvas.toBlob(r, "image/jpeg", this.quality));
  }
}

/*
 * Qué parte del frame de la cámara se ve en el <video> con object-fit:cover,
 * en fracciones del frame. El recorte de cover es centrado, así que da igual
 * que el video esté espejado. null si el elemento todavía no tiene tamaño.
 */
export function visibleRegion(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const cw = video.clientWidth, ch = video.clientHeight;
  if (!vw || !vh || !cw || !ch) return null;
  const scale = Math.max(cw / vw, ch / vh);
  const w = Math.min(1, cw / scale / vw), h = Math.min(1, ch / scale / vh);
  return { x: (1 - w) / 2, y: (1 - h) / 2, w, h };
}

/* ── Dibujo de landmarks ───────────────────────────────────────────────── */

export class LandmarkRenderer {
  /*
   * La inferencia corre a ~15-25 fps pero el overlay se dibuja a 60: cada
   * frame interpolamos hacia los últimos landmarks recibidos. Sin eso la mano
   * dibujada "salta" y se nota feo en video.
   */
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.points = null;
    this.target = null;
    this.visible = false;
    this.accent = COLORS.idle;
    this.enabled = true;
    this._raf = null;
  }

  setLandmarks(pts) {
    if (!pts) { this.visible = false; return; }
    this.target = pts;
    if (!this.points) this.points = pts.map((p) => [p[0], p[1]]);
    this.visible = true;
  }

  start() { if (!this._raf) this._loop(); }
  stop() { cancelAnimationFrame(this._raf); this._raf = null; }

  _loop = () => {
    this._raf = requestAnimationFrame(this._loop);
    const c = this.canvas;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (c.width !== Math.round(w * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.enabled || !this.visible || !this.points || !this.target) return;

    for (let i = 0; i < this.points.length; i++) {
      this.points[i][0] += (this.target[i][0] - this.points[i][0]) * 0.35;
      this.points[i][1] += (this.target[i][1] - this.points[i][1]) * 0.35;
    }

    // El <video> usa object-fit:cover, así que recortamos igual que él para
    // que los puntos caigan sobre la mano y no corridos.
    const vw = this.video.videoWidth || 4, vh = this.video.videoHeight || 3;
    const scale = Math.max(w / vw, h / vh);
    const dw = vw * scale, dh = vh * scale;
    const ox = (w - dw) / 2, oy = (h - dh) / 2;
    const pts = this.points.map((p) => [ox + p[0] * dw, oy + p[1] * dh]);

    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(255,255,255,.55)";
    ctx.lineWidth = 5;
    LINKS.forEach(([a, b]) => {
      ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
    });
    ctx.strokeStyle = this.accent;
    ctx.lineWidth = 2.4;
    LINKS.forEach(([a, b]) => {
      ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
    });
    pts.forEach((p, i) => {
      const r = i === 0 ? 5.5 : TIPS.includes(i) ? 4.6 : 3.4;
      ctx.beginPath(); ctx.arc(p[0], p[1], r + 1.6, 0, 6.2832);
      ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.fill();
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.2832);
      ctx.fillStyle = this.accent; ctx.fill();
    });
  };
}

/* ── Tamaño del teléfono en pantalla ───────────────────────────────────── */

/*
 * La escena mide 390×844, como el prototipo, y se escala entera para ocupar el
 * alto de la ventana. Sin esto en la notebook (810 px lógicos) no entraba, y en
 * un monitor de 1080 quedaba chica con mucho fondo alrededor. Se escala con
 * transform y no cambiando medidas: el layout interno queda idéntico al
 * diseño, sólo se ve más grande o más chico.
 */
const PHONE_W = 390, PHONE_H = 844, PHONE_MARGIN = 0.035;
let phoneScaleValue = 1;

export function phoneScale() { return phoneScaleValue; }

function fitPhone() {
  const phone = document.getElementById("phone");
  if (!phone) return;
  const m = Math.round(window.innerHeight * PHONE_MARGIN);
  phoneScaleValue = Math.min((window.innerHeight - 2 * m) / PHONE_H,
                             (window.innerWidth - 2 * m) / PHONE_W);
  phone.style.transform = `scale(${phoneScaleValue})`;
}

window.addEventListener("resize", fitPhone);
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fitPhone);
else fitPhone();

/* ── Navegación: el menú es la home y todo vuelve a él ─────────────────── */

/*
 * Flujo cerrado para la feria: desde cualquier escena se vuelve al menú con
 * la flecha de atrás, con cualquier elemento .to-menu o con Esc. Cambiar de
 * página apaga la cámara sola, así que no hay que limpiar nada a mano.
 */
export function goHome() { location.href = "/"; }

function wireMenu() {
  document.querySelectorAll(".back, .to-menu").forEach((el) => {
    el.style.cursor = "pointer";
    el.addEventListener("click", goHome);
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && location.pathname !== "/" && !location.pathname.endsWith("index.html")) goHome();
  });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wireMenu);
else wireMenu();

/* ── Utilidades de DOM ─────────────────────────────────────────────────── */

export const $ = (id) => document.getElementById(id);

// Ojo: NO usar style.display para esto. Poner "" borra la declaración inline,
// y si el display del elemento venía de ahí (display:flex en el atributo
// style), al mostrarlo vuelve a `block` y se rompe el layout en silencio.
// El atributo hidden deja que el display lo decida siempre el CSS.
export function show(el, on) { el.hidden = !on; }

export function setText(id, value) {
  const el = $(id);
  if (el && el.textContent !== String(value)) el.textContent = value;
}

export function replay(el, animation) {
  // Reinicia una animación CSS aunque el elemento ya estuviera visible.
  el.style.animation = "none";
  void el.offsetWidth;
  el.style.animation = animation;
}

export async function fetchMeta() {
  try { return await (await fetch("/meta")).json(); } catch (e) { return null; }
}

/* ── Dibujo de pose + manos (señas dinámicas) ──────────────────────────── */

// Sólo torso y brazos: las piernas no aportan a la seña y ensucian el cuadro.
const POSE_LINKS = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];
const POSE_POINTS = [11,12,13,14,15,16,23,24];

export class PoseRenderer {
  /*
   * Equivalente al LandmarkRenderer pero para Holistic: dibuja el torso, los
   * brazos y las dos manos. Mismo criterio de interpolación a 60 fps, que acá
   * importa todavía más porque Holistic corre más lento que Hands.
   */
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;
    this.pose = null; this.poseTarget = null;
    this.hands = {}; this.handsTarget = {};
    this.accent = COLORS.idle;
    this.enabled = true;
    this.visible = false;
    this._raf = null;
  }

  setFrame(pose, hands) {
    this.poseTarget = pose;
    this.handsTarget = hands || {};
    if (pose && !this.pose) this.pose = pose.map((p) => [p[0], p[1]]);
    for (const k of Object.keys(this.handsTarget)) {
      if (!this.hands[k]) this.hands[k] = this.handsTarget[k].map((p) => [p[0], p[1]]);
    }
    this.visible = !!pose;
  }

  start() { if (!this._raf) this._loop(); }
  stop() { cancelAnimationFrame(this._raf); this._raf = null; }

  _loop = () => {
    this._raf = requestAnimationFrame(this._loop);
    const c = this.canvas;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.enabled || !this.visible || !this.pose || !this.poseTarget) return;

    const ease = 0.35;
    for (let i = 0; i < this.pose.length; i++) {
      this.pose[i][0] += (this.poseTarget[i][0] - this.pose[i][0]) * ease;
      this.pose[i][1] += (this.poseTarget[i][1] - this.pose[i][1]) * ease;
    }
    for (const k of Object.keys(this.hands)) {
      const t = this.handsTarget[k];
      if (!t) continue;
      for (let i = 0; i < this.hands[k].length; i++) {
        this.hands[k][i][0] += (t[i][0] - this.hands[k][i][0]) * ease;
        this.hands[k][i][1] += (t[i][1] - this.hands[k][i][1]) * ease;
      }
    }

    const vw = this.video.videoWidth || 4, vh = this.video.videoHeight || 3;
    const scale = Math.max(w / vw, h / vh);
    const dw = vw * scale, dh = vh * scale;
    const ox = (w - dw) / 2, oy = (h - dh) / 2;
    const map = (p) => [ox + p[0] * dw, oy + p[1] * dh];

    ctx.lineCap = "round";

    const stroke = (links, pts, halo, width) => {
      ctx.strokeStyle = halo ? "rgba(255,255,255,.5)" : this.accent;
      ctx.lineWidth = width;
      links.forEach(([a, b]) => {
        if (!pts[a] || !pts[b]) return;
        const p = map(pts[a]), q = map(pts[b]);
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      });
    };

    // Cuerpo más tenue que las manos: la seña se lee en las manos, el cuerpo
    // sólo da contexto de dónde están.
    ctx.globalAlpha = 0.65;
    stroke(POSE_LINKS, this.pose, true, 6);
    stroke(POSE_LINKS, this.pose, false, 2.6);
    POSE_POINTS.forEach((i) => {
      if (!this.pose[i]) return;
      const p = map(this.pose[i]);
      ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, 6.2832);
      ctx.fillStyle = "rgba(255,255,255,.85)"; ctx.fill();
      ctx.beginPath(); ctx.arc(p[0], p[1], 3.4, 0, 6.2832);
      ctx.fillStyle = this.accent; ctx.fill();
    });

    ctx.globalAlpha = 1;
    for (const k of Object.keys(this.hands)) {
      if (!this.handsTarget[k]) continue;
      const pts = this.hands[k];
      stroke(LINKS, pts, true, 5);
      stroke(LINKS, pts, false, 2.4);
      pts.forEach((p0, i) => {
        const p = map(p0);
        const r = i === 0 ? 5 : TIPS.includes(i) ? 4.2 : 3;
        ctx.beginPath(); ctx.arc(p[0], p[1], r + 1.5, 0, 6.2832);
        ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.fill();
        ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 6.2832);
        ctx.fillStyle = this.accent; ctx.fill();
      });
    }
  };
}

/* Etiquetas lindas para las clases del modelo dinámico (que vienen sin tildes). */
export const SIGN_LABELS = {
  papa: "Papá", mama: "Mamá", hermano: "Hermano", hermanos: "Hermanos", amigo: "Amigo",
  casa: "Casa",
  gracias: "Gracias", nombre: "Nombre", estudiar: "Estudiar",
  entender: "Entender", repetir: "Repetir", gato: "Gato",
  computadora: "Computadora", lengua_de_senas: "Lengua de señas",
  reposo: "Reposo",
};
export const pretty = (s) => SIGN_LABELS[s] || (s ? s[0].toUpperCase() + s.slice(1) : "");
