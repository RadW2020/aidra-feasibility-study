-- 016: marca metodologica de la era pre-mascara-afin. Marcar, no borrar (§8).
--
-- Hasta el deploy de edd09e3 (en produccion via 70ee733, primer run
-- 2026-08-16 10:33Z) la mascara de mar de CFAR muestreaba un rectangulo
-- lat/lon alineado al norte sobre escenas rotadas ~13 grados: la tasa de
-- detecciones en tierra de CFAR quedo estancada en ~55% (88% antes de
-- 2026-05-07). Los recuentos de deteccion de esa era no son comparables
-- con los posteriores (ej. escena completa ground: ~904 det/run antes,
-- 548 despues, con output_hash identico entre 5 perfiles).
--
-- Mismo mecanismo que el marcador 'methodology:pre-321de6b' ya usado por
-- los dashboards: se anota la fila y las queries de agregados excluyen o
-- desglosan. La evidencia se conserva integra — es la serie que documenta
-- la mejora 88% -> 55% -> 2.2% para el D4.
--
-- Corte temporal: 2026-08-16T10:00:00Z (ultimo run pre-mascara 07:11Z,
-- primero post-mascara 10:33Z). Idempotente via NOT LIKE.

UPDATE execution_log
SET notes = COALESCE(NULLIF(notes, '') || ' ', '') || 'methodology:pre-affine-mask'
WHERE created_at < '2026-08-16T10:00:00Z'::timestamptz
  AND (notes IS NULL OR notes NOT LIKE '%methodology:pre-affine-mask%');
