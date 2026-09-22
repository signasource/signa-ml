#!/usr/bin/env node
/**
 * Deja una carpeta de .glb lista para el motor 3D de la app.
 *
 * Es un solo archivo, no necesita el repositorio ni Python: se copia a
 * cualquier carpeta, se instalan las dependencias con una línea y se corre.
 *
 * ─── Instrucciones ──────────────────────────────────────────────────────────
 *
 *   1. Instalar Node.js (https://nodejs.org), versión 18 o más nueva.
 *
 *   2. Poner este archivo en una carpeta vacía y, dentro de esa carpeta:
 *
 *        npm init -y
 *        npm install @gltf-transform/core @gltf-transform/extensions \
 *                    @gltf-transform/functions draco3dgltf sharp basisu
 *
 *   3. Correrlo:
 *
 *        node preparar-avatares.mjs <carpeta-con-glb> <carpeta-de-salida>
 *
 *      Por ejemplo:
 *
 *        node preparar-avatares.mjs ./originales ./listos
 *
 * Lee todos los .glb de la primera carpeta, escribe los preparados en la
 * segunda con el mismo nombre, y no toca los originales.
 *
 * Eso alcanza: el compresor de texturas de Khronos viene en el paquete
 * `basisu`, así que no hay que bajar nada a mano.
 *
 * ─── Qué arregla, y por qué ─────────────────────────────────────────────────
 *
 * Los avatares tal como salen del exportador traen cuatro cosas:
 *
 *   1. Texturas en webp, que el motor de la app no sabe decodificar: el modelo
 *      carga entero y se ve NEGRO, sin ningún error a la vista.
 *   2. Un canal de animación —el de los ojos— que apunta a datos que no
 *      existen. El motor descarta por eso la animación ENTERA y el avatar se
 *      queda quieto en la primera pose.
 *   3. Esqueletos de más de 256 huesos. Las letras M y N traen 574 porque se
 *      exportaron con el rig de control completo; el resto tiene 198. Pasarse
 *      de ese número no da error: CIERRA la app.
 *   4. Canales que mueven huesos que no deforman nada, llaves de animación
 *      repetidas, vértices duplicados y nodos sueltos. No rompen nada, pero se
 *      calculan en cada cuadro.
 *
 * Nada de lo que hace este script cambia cómo se ve la seña: no simplifica las
 * mallas ni toca el movimiento.
 */
import { execFileSync } from 'node:child_process';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';

const FALTAN = [];
let core, extensions, functions, draco3d, sharp;
for (const [nombre, destino] of [
  ['@gltf-transform/core', (m) => (core = m)],
  ['@gltf-transform/extensions', (m) => (extensions = m)],
  ['@gltf-transform/functions', (m) => (functions = m)],
  ['draco3dgltf', (m) => (draco3d = m)],
  ['sharp', (m) => (sharp = m)],
]) {
  try {
    destino(await import(nombre));
  } catch {
    FALTAN.push(nombre);
  }
}

if (FALTAN.length) {
  console.error('Faltan dependencias. Dentro de esta carpeta, correr:\n');
  console.error('  npm init -y');
  console.error(`  npm install ${FALTAN.join(' ')}\n`);
  process.exit(1);
}

const [entrada, salida] = process.argv.slice(2);
if (!entrada || !salida) {
  console.error('Uso: node preparar-avatares.mjs <carpeta-con-glb> <carpeta-de-salida>');
  process.exit(1);
}

/** Máximo que admite el motor de la app. Pasarse cierra la aplicación. */
const MAX_HUESOS = 256;

/** Arriba de esto la textura no se nota en pantalla y sí en el peso. */
const LADO_TEXTURA = 1024;

/**
 * El compresor de texturas de Khronos que viene en el paquete `basisu`.
 *
 * Se busca el binario a mano en vez de usar el atajo que trae el paquete:
 * ese atajo depende de otro módulo nativo que no siempre compila, y acá sólo
 * hace falta el ejecutable.
 */
const basisu = (() => {
  const plataformas = { linux: 'linux', darwin: 'darwin', win32: 'win' };
  const carpeta = plataformas[process.platform];
  if (!carpeta) return null;
  const arco = process.arch === 'arm64' ? 'arm64' : 'x64';
  const exe = process.platform === 'win32' ? 'basisu.exe' : 'basisu';
  for (const variante of [arco, `${arco}_sse`]) {
    const ruta = join(process.cwd(), 'node_modules', 'basisu', 'bin', carpeta, variante, exe);
    if (!existsSync(ruta)) continue;
    try {
      if (process.platform !== 'win32') chmodSync(ruta, 0o755);
      execFileSync(ruta, ['-version'], { stdio: 'ignore' });
      return ruta;
    } catch {
      // Probar la otra variante.
    }
  }
  return null;
})();

const io = new core.NodeIO()
  .registerExtensions(extensions.ALL_EXTENSIONS)
  .registerDependencies({
    'draco3d.decoder': await draco3d.createDecoderModule(),
    'draco3d.encoder': await draco3d.createEncoderModule(),
  });

/**
 * Quita del esqueleto los huesos que la malla no usa y renumera los índices.
 *
 * Un rig tiene huesos que deforman la malla y huesos de control, que el
 * animador manipula y que no deforman nada. Si al exportar no se marca "sólo
 * deformación", los de control viajan igual.
 */
function podarHuesos(doc) {
  let quitados = 0;
  for (const skin of doc.getRoot().listSkins()) {
    const juntas = skin.listJoints();
    if (juntas.length <= MAX_HUESOS) continue;

    const usados = new Set();
    const atributos = [];
    for (const malla of doc.getRoot().listMeshes()) {
      for (const prim of malla.listPrimitives()) {
        const j = prim.getAttribute('JOINTS_0');
        if (!j) continue;
        atributos.push(j);
        for (const v of j.getArray()) usados.add(v);
      }
    }
    if (!usados.size) continue;

    const quedan = [...usados].sort((a, b) => a - b);
    const mapa = new Map(quedan.map((viejo, nuevo) => [viejo, nuevo]));

    // Las matrices de bind van una por hueso y en el mismo orden.
    const ibm = skin.getInverseBindMatrices();
    if (ibm) {
      const datos = ibm.getArray();
      const nuevos = new Float32Array(quedan.length * 16);
      quedan.forEach((viejo, i) => nuevos.set(datos.slice(viejo * 16, viejo * 16 + 16), i * 16));
      ibm.setArray(nuevos);
    }

    juntas.forEach((hueso, i) => {
      if (!mapa.has(i)) skin.removeJoint(hueso);
    });
    for (const atributo of atributos) {
      const datos = atributo.getArray();
      for (let i = 0; i < datos.length; i++) datos[i] = mapa.get(datos[i]) ?? 0;
      atributo.setArray(datos);
    }
    quitados += juntas.length - skin.listJoints().length;
  }
  return quitados;
}

/** Canales que mueven huesos que ya no deforman nada ni son padres de uno. */
function podarCanales(doc) {
  const raiz = doc.getRoot();
  const importan = new Set();
  for (const skin of raiz.listSkins()) {
    for (const hueso of skin.listJoints()) {
      let n = hueso;
      while (n && !importan.has(n)) {
        importan.add(n);
        n = typeof n.getParentNode === 'function' ? n.getParentNode() : null;
      }
    }
  }
  for (const nodo of raiz.listNodes()) if (nodo.getMesh()) importan.add(nodo);

  let quitados = 0;
  for (const anim of raiz.listAnimations()) {
    for (const canal of anim.listChannels()) {
      const destino = canal.getTargetNode();
      if (destino && !importan.has(destino)) {
        canal.dispose();
        quitados++;
      }
    }
  }
  return quitados;
}

let contador = 0;
const resumen = [];

const archivos = readdirSync(entrada).filter((n) => n.toLowerCase().endsWith('.glb'));
if (!archivos.length) {
  console.error(`No hay archivos .glb en ${resolve(entrada)}`);
  process.exit(1);
}

mkdirSync(salida, { recursive: true });
console.log(
  basisu
    ? 'Texturas: KTX2, comprimidas para la placa de video'
    : 'Texturas: JPEG y PNG (no se encontró el compresor de Khronos para esta plataforma)',
);
console.log('');

for (const nombre of archivos) {
  const origen = join(entrada, nombre);
  const destino = join(salida, nombre);
  const antes = statSync(origen).size;

  const doc = await io.read(origen);
  const huesos = podarHuesos(doc);
  const canales = podarCanales(doc);
  let usaKtx2 = false;

  // Texturas: a un formato que el motor entienda y al tamaño que se usa en
  // pantalla. JPEG salvo que la textura tenga transparencia —el pelo la usa—,
  // porque en PNG los mismos 1024x1024 pesan varias veces más.
  const sh = sharp.default ?? sharp;
  for (const textura of doc.getRoot().listTextures()) {
    const datos = textura.getImage();
    if (!datos) continue;
    const imagen = sh(Buffer.from(datos));
    const meta = await imagen.metadata();

    // Los lados se redondean a múltiplos de 4: la compresión para GPU trabaja
    // en bloques de 4x4 píxeles y una textura que no encaja se rechaza, con un
    // "Failed" sin explicación.
    const escala = Math.min(1, LADO_TEXTURA / Math.max(meta.width ?? 1, meta.height ?? 1));
    const aMultiploDe4 = (n) => Math.max(4, Math.round((n * escala) / 4) * 4);
    const ancho = aMultiploDe4(meta.width ?? LADO_TEXTURA);
    const alto = aMultiploDe4(meta.height ?? LADO_TEXTURA);
    const escalada = imagen.resize(ancho, alto, { fit: 'fill', kernel: 'lanczos3' });

    const conAlfa = Boolean(meta.hasAlpha);
    const intermedio = conAlfa
      ? await escalada.png().toBuffer()
      : await escalada.jpeg({ quality: 92 }).toBuffer();

    if (!basisu) {
      textura
        .setImage(intermedio)
        .setMimeType(conAlfa ? 'image/png' : 'image/jpeg');
      continue;
    }

    // KTX2: la textura queda comprimida también mientras se dibuja, así que la
    // placa de video la lee tal cual, sin desarmarla y sin ocupar de más.
    const entradaTmp = join(tmpdir(), `signa-${process.pid}-${contador++}.${conAlfa ? 'png' : 'jpg'}`);
    const salidaTmp = `${entradaTmp}.ktx2`;
    writeFileSync(entradaTmp, intermedio);
    try {
      execFileSync(
        basisu,
        ['-ktx2', '-mipmap', '-q', '190', ...(conAlfa ? ['-force_alpha'] : []), '-output_file', salidaTmp, entradaTmp],
        { stdio: 'ignore' },
      );
      textura.setImage(readFileSync(salidaTmp)).setMimeType('image/ktx2');
      usaKtx2 = true;
    } catch {
      textura.setImage(intermedio).setMimeType(conAlfa ? 'image/png' : 'image/jpeg');
      console.warn(`  (${nombre}: una textura no se pudo comprimir, queda sin comprimir)`);
    }
    for (const tmp of [entradaTmp, salidaTmp]) if (existsSync(tmp)) rmSync(tmp);
  }

  // Declarar KTX2 en el archivo: sin esto, un visor que lo abra no sabe que
  // tiene que pedirle a la placa de video que lo descomprima.
  if (usaKtx2) doc.createExtension(extensions.KHRTextureBasisu).setRequired(true);

  // La extensión de webp queda declarada aunque ya no haya ninguna textura en
  // ese formato, y un cargador estricto puede exigirla igual.
  for (const ext of doc.getRoot().listExtensionsUsed()) {
    if (ext.extensionName === 'EXT_texture_webp') ext.dispose();
  }

  await doc.transform(
    functions.resample(),
    functions.weld(),
    functions.dedup(),
    functions.prune(),
  );

  // Las mallas se vuelven a comprimir con Draco al escribir: el archivo ya
  // traía esa extensión y se conserva.
  await io.write(destino, doc);

  const despues = statSync(destino).size;
  const detalle = [
    huesos ? `${huesos} huesos de más` : null,
    canales ? `${canales} canales que no movían nada` : null,
  ].filter(Boolean);
  console.log(
    `  ${basename(nombre).padEnd(16)} ${(antes / 1048576).toFixed(2)} MB → ` +
      `${(despues / 1048576).toFixed(2)} MB${detalle.length ? `  (sacados: ${detalle.join(', ')})` : ''}`,
  );
  resumen.push({ nombre, antes, despues, huesos, canales, ktx2: usaKtx2 });
}

// Un resumen al lado de los archivos: quien los sube no es necesariamente
// quien los preparó, y conviene que pueda ver qué se tocó sin preguntar.
const lineas = [
  'Avatares preparados para el motor 3D de la app.',
  '',
  'archivo'.padEnd(20) + 'antes'.padStart(10) + 'después'.padStart(10) + '  texturas',
  ''.padEnd(52, '-'),
  ...resumen.map(
    (r) =>
      r.nombre.padEnd(20) +
      `${(r.antes / 1048576).toFixed(2)} MB`.padStart(10) +
      `${(r.despues / 1048576).toFixed(2)} MB`.padStart(10) +
      `  ${r.ktx2 ? 'KTX2' : 'JPEG/PNG'}` +
      (r.huesos ? `  · ${r.huesos} huesos de más quitados` : '') +
      (r.canales ? `  · ${r.canales} canales sin efecto quitados` : ''),
  ),
  '',
  'Qué se hizo en cada uno, y por qué:',
  '  · huesos que la malla no usa: arriba de 256 el motor de la app se cierra',
  '  · canales de animación que mueven huesos que ya no deforman nada',
  '  · nodos sueltos, llaves de animación repetidas y vértices duplicados',
  '  · texturas a 1024 px y comprimidas; el canal de animación sin datos, arreglado',
  '  · mallas recomprimidas con Draco',
  '',
  'Nada de esto cambia cómo se ve la seña: no se simplificaron las mallas ni se',
  'tocó el movimiento.',
];
writeFileSync(join(salida, 'resumen.txt'), lineas.join('\n') + '\n');

console.log(`\nListo: ${archivos.length} archivo(s) en ${resolve(salida)}`);
console.log('');
console.log('Para subirlos: el contenido de esa carpeta reemplaza a los archivos del');
console.log('mismo nombre que hay hoy en el servidor. No hay que renombrar nada ni');
console.log('cambiar la estructura; la app los pide por el nombre de la seña.');
console.log('');
console.log(`Queda también ${join(salida, 'resumen.txt')} con el detalle de cada archivo.`);
