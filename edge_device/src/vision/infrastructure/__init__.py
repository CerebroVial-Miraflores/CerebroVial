"""Infrastructure layer for the Computer Vision module.

Concrete adapters live in their own submodules and are imported directly
(`from ...infrastructure.zones.zone_counter import ZoneCounter`). This barrel
is intentionally empty: an eager re-export here forces heavy optional deps
(ultralytics, supervision, cv2) to load whenever *any* infra submodule is
imported, which broke dependency-light tests. Re-exports return per component
as the layer is rewritten (TTH-08 Fase 4+).
"""
