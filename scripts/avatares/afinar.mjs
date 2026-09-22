/**
 * Saca del .glb lo que no se ve, sin tocar cómo se ve.
 *
 * - Canales de animación que mueven huesos que no deforman nada ni son padres
 *   de alguno que sí: en M y N el rig de control quedó animado aunque ya no
 *   esté en el esqueleto, y eso son cuentas por cuadro que no mueven un píxel.
 * - Vértices duplicados, datos repetidos y nodos sueltos.
 * - Llaves de animación repetidas: están horneadas cuadro por cuadro.
 */
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, prune, resample, weld } from '@gltf-transform/functions';
import draco3d from 'draco3dgltf';

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'draco3d.decoder': await draco3d.createDecoderModule(),
  'draco3d.encoder': await draco3d.createEncoderModule(),
});
const [entrada, salida] = process.argv.slice(2);
const doc = await io.read(entrada);
const raiz = doc.getRoot();
const antes = raiz.listAnimations().reduce((n, a) => n + a.listChannels().length, 0);

// Qué nodos importan: los huesos y todos sus ancestros, porque las
// transformaciones se componen bajando por el árbol.
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

for (const anim of raiz.listAnimations()) {
  for (const canal of anim.listChannels()) {
    const destino = canal.getTargetNode();
    if (destino && !importan.has(destino)) canal.dispose();
  }
}

await doc.transform(resample(), weld(), dedup(), prune());
const despues = raiz.listAnimations().reduce((n, a) => n + a.listChannels().length, 0);
console.log(`  canales: ${antes} → ${despues}`);
await io.write(salida, doc);
