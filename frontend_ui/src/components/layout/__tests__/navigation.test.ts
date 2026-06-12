import { describe, expect, it } from 'vitest';

import { TABS_BY_ROLE, type Tab } from '../../../auth/roles';
import type { Role } from '../../../auth/types';
import { NAV_DESCRIPTORS, PATH_BY_TAB, pathForTab, rolesForTab, tabForPath } from '../navigation';

const ALL_TABS = Object.keys(PATH_BY_TAB) as Tab[];

describe('navigation — mapeo tab ↔ ruta', () => {
  it('round-trip: tabForPath(pathForTab(tab)) === tab para los 5 tabs', () => {
    for (const tab of ALL_TABS) {
      expect(tabForPath(pathForTab(tab))).toBe(tab);
    }
  });

  it("'/' resuelve a dashboard y las rutas desconocidas a null", () => {
    expect(tabForPath('/')).toBe('dashboard');
    expect(tabForPath('/no-existe')).toBeNull();
  });

  // FASE 3: rutas legacy fuera del union → null (las maneja el redirect del
  // router, D3.1a); el puente /camara/:id mantiene activo el tab del comando.
  it('legacy /alertas y /congestion ya no mapean a tab; /camara alias a dashboard', () => {
    expect(tabForPath('/alertas')).toBeNull();
    expect(tabForPath('/congestion')).toBeNull();
    expect(tabForPath('/camara/cam_larco_benavides')).toBe('dashboard');
  });

  it('resuelve por primer segmento (subrutas futuras no rompen el tab activo)', () => {
    expect(tabForPath('/control/detalle')).toBe('control');
  });

  it('rolesForTab es consistente con TABS_BY_ROLE', () => {
    for (const tab of ALL_TABS) {
      const expected = (Object.keys(TABS_BY_ROLE) as Role[]).filter((role) =>
        TABS_BY_ROLE[role].includes(tab),
      );
      expect(rolesForTab(tab)).toEqual(expected);
    }
  });

  it('NAV_DESCRIPTORS cubre todos los tabs exactamente una vez', () => {
    expect(NAV_DESCRIPTORS.map((d) => d.tab).sort()).toEqual([...ALL_TABS].sort());
  });
});
