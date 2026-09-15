# Animaciones 3D locales

Dejá acá archivos `.glb` con el nombre de la letra: `A.glb`, `B.glb`, `Ñ.glb`…

El servidor los busca **antes** de preguntarle a signa-api, así que sirven para
tener el picture-in-picture andando sin backend, o para pisar una animación
puntual mientras se prueba.

Se cargan con `<model-viewer>`, el mismo componente que usa `GlbAnimationView`
en signa-mobile, así que cualquier `.glb` que funcione en la app funciona acá.

Si no hay archivo local ni el API devuelve nada, la letra muestra una tarjeta
rayada con la letra grande — el mismo fallback que `SignPlaceholder` en la app.
