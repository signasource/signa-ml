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
 *                    @gltf-transform/functions draco3dgltf sharp
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
 * ─── Opcional: texturas KTX2 ────────────────────────────────────────────────
 *
 * Si además está instalada la herramienta `ktx` de Khronos, las texturas se
 * comprimen en un formato que la placa de video lee sin desarmar: carga más
 * rápido y ocupa menos memoria. Sin ella, el script usa JPEG y PNG, que andan
 * igual de bien aunque un poco más lentos.
 *
 *   https://github.com/KhronosGroup/KTX-Software/releases
 *
 * OJO con la versión: hace falta 4.4 o más nueva. Con la 4.3 la compresión
 * falla textura por textura y el archivo sale igual pero en JPEG, sin que nada
 * lo avise. El script lo comprueba y lo dice al empezar.
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
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from 'node:fs';
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
 * ¿Está la herramienta de Khronos, y sirve?
 *
 * La 4.3 no entiende las opciones que le pasa gltf-transform y falla textura
 * por textura, en silencio: el archivo sale igual, pero con las texturas en
 * JPEG en vez de comprimidas. Por eso se mira la versión y no sólo que exista.
 */
const hayKtx = (() => {
  try {
    const salida = execFileSync('ktx', ['--version'], { encoding: 'utf8' });
    const version = salida.match(/v?(\d+)\.(\d+)/);
    if (!version) return false;
    const sirve = Number(version[1]) > 4 || (Number(version[1]) === 4 && Number(version[2]) >= 4);
    if (!sirve) {
      console.warn(`La herramienta ktx es la ${version[0]} y hace falta 4.4 o más nueva.`);
      console.warn('Se usan JPEG y PNG, que andan igual aunque cargan un poco más lento.\n');
    }
    return sirve;
  } catch {
    return false;
  }
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

const archivos = readdirSync(entrada).filter((n) => n.toLowerCase().endsWith('.glb'));
if (!archivos.length) {
  console.error(`No hay archivos .glb en ${resolve(entrada)}`);
  process.exit(1);
}

mkdirSync(salida, { recursive: true });
console.log(hayKtx ? 'Texturas: KTX2 (la mejor opción)' : 'Texturas: JPEG/PNG (sin la herramienta ktx)');
console.log('');

for (const nombre of archivos) {
  const origen = join(entrada, nombre);
  const destino = join(salida, nombre);
  const antes = statSync(origen).size;

  const doc = await io.read(origen);
  const huesos = podarHuesos(doc);
  const canales = podarCanales(doc);

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

    if (meta.hasAlpha) {
      textura.setImage(await escalada.png().toBuffer()).setMimeType('image/png');
    } else {
      textura.setImage(await escalada.jpeg({ quality: 90 }).toBuffer()).setMimeType('image/jpeg');
    }
  }

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

  // Los dos últimos pasos necesitan binarios aparte, así que van por la línea
  // de comandos. Se encadenan con archivos temporales: escribir sobre el mismo
  // archivo que se está leyendo no funciona y se saltea en silencio.
  const paso1 = `${destino}.paso1`;
  const paso2 = `${destino}.paso2`;
  await io.write(paso1, doc);

  const cli = (comando, ent, sal) => {
    execFileSync('npx', ['--yes', '@gltf-transform/cli@latest', comando, ent, sal], {
      stdio: 'ignore',
    });
  };

  let actual = paso1;
  if (hayKtx) {
    try {
      cli('etc1s', actual, paso2);
      actual = paso2;
    } catch {
      console.warn(`  (${nombre}: no se pudo comprimir a KTX2, sigue en JPEG)`);
    }
  }
  try {
    cli('draco', actual, destino);
  } catch {
    renameSync(actual, destino);
    console.warn(`  (${nombre}: no se pudo comprimir la malla con Draco)`);
  }
  for (const temporal of [paso1, paso2]) if (existsSync(temporal)) rmSync(temporal);

  const despues = statSync(destino).size;
  const detalle = [
    huesos ? `${huesos} huesos de más` : null,
    canales ? `${canales} canales que no movían nada` : null,
  ].filter(Boolean);
  console.log(
    `  ${basename(nombre).padEnd(16)} ${(antes / 1048576).toFixed(2)} MB → ` +
      `${(despues / 1048576).toFixed(2)} MB${detalle.length ? `  (sacados: ${detalle.join(', ')})` : ''}`,
  );
}

console.log(`\nListo: ${archivos.length} archivo(s) en ${resolve(salida)}`);
console.log('Para revisar uno antes de subirlo:');
console.log(`  npx @gltf-transform/cli inspect ${join(salida, archivos[0])}`);
