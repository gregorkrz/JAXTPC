"""
Random boundary muon track specs for visualization and landscape jobs.

See ``generate_random_boundary_tracks`` and ``filter_track_inside_volumes``.
"""
from __future__ import annotations

import numpy as np

# Random x-face muons count; ``generate_random_boundary_tracks`` appends one fixed
# diagonal cross-detector muon by default (total default tracks = this + 1).
N_DEFAULT_BOUNDARY_MUONS = 12
_MIN_INWARD_DIR_DX = 0.12

# Full-volume diagonal (1000 MeV): start at +++ corner mm, toward --- corner; direction (−1,−1,−1) normalized.
_DIAG_CROSS_START_MM = (2000.0, 2000.0, 2000.0)
_DIAG_CROSS_END_MM = (-2000.0, -2000.0, -2000.0)

_MUON_ENERGIES_MEV = (100, 500, 1000)


def _inside_any_volume_mask(positions_mm, volumes) -> np.ndarray:
    """True where a point lies in at least one active volume (same half-open rules as ``build_deposit_data``)."""
    pos = np.asarray(positions_mm, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f'Expected positions (N, 3), got {pos.shape}')
    if len(pos) == 0:
        return np.zeros(0, dtype=bool)
    x_cm = pos[:, 0] / 10.0
    y_cm = pos[:, 1] / 10.0
    z_cm = pos[:, 2] / 10.0
    inside = np.zeros(len(pos), dtype=bool)
    for vol in volumes:
        x0, x1 = vol.ranges_cm[0]
        y0, y1 = vol.ranges_cm[1]
        z0, z1 = vol.ranges_cm[2]
        inside |= (
            (x_cm >= x0) & (x_cm < x1)
            & (y_cm >= y0) & (y_cm < y1)
            & (z_cm >= z0) & (z_cm < z1)
        )
    return inside


def filter_track_inside_volumes(track: dict, volumes) -> dict:
    """Drop steps whose centroid is outside all active volumes (same bounds as ``build_deposit_data``)."""
    pos = np.asarray(track['position'])
    if pos.size == 0:
        return track
    mask = _inside_any_volume_mask(pos, volumes)
    if bool(mask.all()):
        return track
    if not bool(mask.any()):
        n = 0
        return {
            'position': np.zeros((n, 3), dtype=np.float32),
            'x': np.zeros(n, dtype=np.float32),
            'y': np.zeros(n, dtype=np.float32),
            'z': np.zeros(n, dtype=np.float32),
            'de': np.zeros(n, dtype=np.float32),
            'dx': np.zeros(n, dtype=np.float32),
            'theta': np.zeros(n, dtype=np.float32),
            'phi': np.zeros(n, dtype=np.float32),
            'track_id': np.zeros(n, dtype=np.int32),
        }

    def _sl(k):
        return np.asarray(track[k])[mask]

    positions_arr = np.asarray(_sl('position'), dtype=np.float32)
    out = dict(track)
    out['position'] = positions_arr
    out['x'] = positions_arr[:, 0]
    out['y'] = positions_arr[:, 1]
    out['z'] = positions_arr[:, 2]
    out['de'] = np.asarray(_sl('de'), dtype=np.float32)
    out['dx'] = np.asarray(_sl('dx'), dtype=np.float32)
    out['theta'] = np.asarray(_sl('theta'), dtype=np.float32)
    out['phi'] = np.asarray(_sl('phi'), dtype=np.float32)
    out['track_id'] = np.asarray(_sl('track_id'), dtype=np.int32)
    return out


def _random_inward_direction(rng, start_side: str):
    """Unit vector into the TPC: East needs ``dx >= min``; West needs ``dx <= -min`` (after normalization)."""
    m = _MIN_INWARD_DIR_DX
    for _ in range(4096):
        v = rng.normal(size=3)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-9:
            continue
        v = v / nrm
        if start_side == 'east':
            if v[0] <= 0:
                v[0] = -v[0]
            if v[0] < m:
                continue
        else:
            if v[0] >= 0:
                v[0] = -v[0]
            if v[0] > -m:
                continue
        return (float(v[0]), float(v[1]), float(v[2]))
    raise RuntimeError('Failed to sample inward direction')


def _random_vertex_outer_x_face_mm(rng, volumes, start_side: str):
    """Uniform ``y,z`` on active bounds; ``x`` on East or West outer x face (mm)."""
    if len(volumes) < 2:
        raise ValueError('Expected at least 2 TPC volumes')
    east, west = volumes[0], volumes[1]
    y0 = east.ranges_cm[1][0] * 10.0
    y1 = east.ranges_cm[1][1] * 10.0
    z0 = east.ranges_cm[2][0] * 10.0
    z1 = east.ranges_cm[2][1] * 10.0
    y = float(rng.uniform(y0, y1))
    z = float(rng.uniform(z0, z1))
    if start_side == 'east':
        x = float(east.ranges_cm[0][0] * 10.0)
    else:
        x = float(west.ranges_cm[0][1] * 10.0)
    return (x, y, z)


def _diagonal_cross_detector_track():
    """1000 MeV, unit direction from start toward (-2000,-2000,-2000) mm (body diagonal)."""
    a = np.asarray(_DIAG_CROSS_START_MM, dtype=np.float64)
    b = np.asarray(_DIAG_CROSS_END_MM, dtype=np.float64)
    v = b - a
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-9:
        raise ValueError('Diagonal cross track: zero length')
    u = v / nrm
    return dict(
        name='Muon_diagCross_1000MeV',
        direction=tuple(float(x) for x in u),
        momentum_mev=1000.0,
        start_position_mm=tuple(float(x) for x in a),
    )


def generate_random_boundary_tracks(
    volumes,
    n=N_DEFAULT_BOUNDARY_MUONS,
    seed=42,
    *,
    include_diagonal_cross_muon: bool = True,
):
    """Generate ``n`` random boundary-start muon specs, optionally plus one fixed diagonal.

    The optional extra muon has T=1000 MeV, starts at (2000, 2000, 2000) mm and
    travels along the body diagonal toward (-2000, -2000, -2000) mm (unit direction).

    Returns list[dict] with keys:
      - name, direction, momentum_mev, start_position_mm
    """
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(1, n + 1):
        t_mev = int(rng.choice(_MUON_ENERGIES_MEV))
        start_side = 'east' if rng.integers(0, 2) == 0 else 'west'
        start_mm = _random_vertex_outer_x_face_mm(rng, volumes, start_side)
        direction = _random_inward_direction(rng, start_side)
        specs.append(dict(
            name=f'Muon{i}_{t_mev}MeV',
            direction=direction,
            momentum_mev=float(t_mev),
            start_position_mm=start_mm,
        ))
    if include_diagonal_cross_muon:
        specs.append(_diagonal_cross_detector_track())
    return specs
