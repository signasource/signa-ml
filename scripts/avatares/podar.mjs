/** Quita del esqueleto los huesos que la malla no usa. Ver avatar-3d.md. */
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import draco3d from 'draco3dgltf';

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'draco3d.decoder': await draco3d.createDecoderModule(),
  'draco3d.encoder': await draco3d.createEncoderModule(),
});
const [entrada, salida] = process.argv.slice(2);
const doc = await io.read(entrada);

for (const skin of doc.getRoot().listSkins()) {
  const juntas = skin.listJoints();
  const usados = new Set();
  const atributos = [];
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const j = prim.getAttribute('JOINTS_0');
      if (!j) continue;
      atributos.push(j);
      for (const v of j.getArray()) usados.add(v);
    }
  }
  if (!usados.size || juntas.length <= 256) continue;
  const quedan = [...usados].sort((a, b) => a - b);
  const mapa = new Map(quedan.map((viejo, nuevo) => [viejo, nuevo]));
  const ibm = skin.getInverseBindMatrices();
  if (ibm) {
    const datos = ibm.getArray();
    const nuevos = new Float32Array(quedan.length * 16);
    quedan.forEach((viejo, i) => nuevos.set(datos.slice(viejo * 16, viejo * 16 + 16), i * 16));
    ibm.setArray(nuevos);
  }
  juntas.forEach((j, i) => { if (!mapa.has(i)) skin.removeJoint(j); });
  for (const atributo of atributos) {
    const datos = atributo.getArray();
    for (let i = 0; i < datos.length; i++) datos[i] = mapa.get(datos[i]) ?? 0;
    atributo.setArray(datos);
  }
  console.log(`  huesos: ${juntas.length} → ${skin.listJoints().length}`);
}
await io.write(salida, doc);
