"""
Random boundary muon track specs for visualization and landscape jobs.

See ``generate_random_boundary_tracks`` and ``filter_track_inside_volumes``.
"""
from __future__ import annotations

import numpy as np

# Random x-face muons count. By default ``generate_random_boundary_tracks`` also adds
# three fixed 1000 MeV chords through East+West (one body diagonal + two skew chords).
N_DEFAULT_BOUNDARY_MUONS = 12
_MIN_INWARD_DIR_DX = 0.12

# Full-volume body diagonal (1000 MeV).
_DIAG_CROSS_START_MM = (2000.0, 2000.0, 2000.0)
_DIAG_CROSS_END_MM = (-2000.0, -2000.0, -2000.0)

# Two oblique chords: start mm, target mm defining unit velocity (cross cathode × both volumes).
_FIXED_THROUGH_BOTH_CHORDS = (
    ('Muon_throughEw_skew02_1000MeV',
     (-2100.0, 750.0, -550.0),
     (2100.0, -520.0, 420.0)),
    ('Muon_throughWe_skew03_1000MeV',
     (2100.0, -620.0, 480.0),
     (-2100.0, 580.0, -490.0)),
)

_MUON_ENERGIES_MEV = (100, 500, 1000)

_NICE_TRACK_X_RANGE_MM  = 1000.0  # |x| < this for near-cathode face entries
_NICE_TRACK_THETA_MIN_DEG = 30.0  # min polar angle from x-axis (drift direction)


def _balanced_boundary_energies_mev(rng, n: int) -> list[int]:
    """Return ``n`` energies spread as evenly as possible across ``_MUON_ENERGIES_MEV``."""
    if n < 0:
        raise ValueError('n must be non-negative')
    if n == 0:
        return []
    per_energy, remainder = divmod(n, len(_MUON_ENERGIES_MEV))
    energies: list[int] = []
    for i, t_mev in enumerate(sorted(_MUON_ENERGIES_MEV, reverse=True)):
        count = per_energy + (1 if i < remainder else 0)
        energies.extend([int(t_mev)] * count)
    rng.shuffle(energies)
    return energies


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


def _fixed_chord_muon_track(name: str, start_mm, toward_mm, momentum_mev: float = 1000.0):
    """Fixed T; unit ``direction`` points from ``start_mm`` toward ``toward_mm`` (mm)."""
    a = np.asarray(start_mm, dtype=np.float64)
    b = np.asarray(toward_mm, dtype=np.float64)
    v = b - a
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-9:
        raise ValueError(f'Track {name!r}: zero chord length')
    u = v / nrm
    return dict(
        name=name,
        direction=tuple(float(x) for x in u),
        momentum_mev=float(momentum_mev),
        start_position_mm=tuple(float(x) for x in a),
    )


def _diagonal_cross_detector_track():
    """1000 MeV body-diagonal chord (see ``_DIAG_CROSS_*`` constants)."""
    return _fixed_chord_muon_track(
        'Muon_diagCross_1000MeV',
        _DIAG_CROSS_START_MM,
        _DIAG_CROSS_END_MM,
        momentum_mev=1000.0,
    )


def _oblique_through_both_volume_muons():
    """Two 1000 MeV skew chords, East↔West, distinct from body diagonal."""
    return [
        _fixed_chord_muon_track(name, a, b, momentum_mev=1000.0)
        for name, a, b in _FIXED_THROUGH_BOTH_CHORDS
    ]


def _sample_nice_direction(rng):
    """Unit direction with polar angle θ from x-axis in [30°, 150°].

    Samples cos θ uniformly in [−cos 30°, +cos 30°] and φ uniformly in [0, 2π).
    Negating the result maps θ → π−θ, which stays in the same range, so the
    caller can safely negate to enforce the inward-facing constraint.
    """
    cos_max = float(np.cos(np.radians(_NICE_TRACK_THETA_MIN_DEG)))  # ≈ 0.866
    u = float(rng.uniform(-cos_max, cos_max))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    sin_theta = float(np.sqrt(1.0 - u * u))
    return (u, sin_theta * float(np.cos(phi)), sin_theta * float(np.sin(phi)))


def _random_nice_face_entry_mm(rng, volumes):
    """Random point on a y/z face of the combined volume with |x| < _NICE_TRACK_X_RANGE_MM.

    Faces are weighted by area. Returns (start_mm, outward_normal).
    """
    east = volumes[0]
    y0 = east.ranges_cm[1][0] * 10.0
    y1 = east.ranges_cm[1][1] * 10.0
    z0 = east.ranges_cm[2][0] * 10.0
    z1 = east.ranges_cm[2][1] * 10.0
    x_lo, x_hi = -_NICE_TRACK_X_RANGE_MM, _NICE_TRACK_X_RANGE_MM

    x_span = x_hi - x_lo
    a_tb = x_span * (z1 - z0)  # top / bottom face area
    a_fb = x_span * (y1 - y0)  # front / back face area
    total = 2.0 * a_tb + 2.0 * a_fb

    r = float(rng.uniform(0.0, total))
    x = float(rng.uniform(x_lo, x_hi))
    if r < a_tb:
        return (x, y1, float(rng.uniform(z0, z1))), (0.0,  1.0,  0.0)
    elif r < 2.0 * a_tb:
        return (x, y0, float(rng.uniform(z0, z1))), (0.0, -1.0,  0.0)
    elif r < 2.0 * a_tb + a_fb:
        return (x, float(rng.uniform(y0, y1)), z1), (0.0,  0.0,  1.0)
    else:
        return (x, float(rng.uniform(y0, y1)), z0), (0.0,  0.0, -1.0)


def generate_random_nice_tracks(volumes, n=10, seed=7):
    """Generate n near-cathode muon tracks entering through y/z faces with |x| < 1000 mm.

    Entry points are sampled uniformly on the four y/z faces of the combined TPC
    volume (weighted by area), restricted to |x| < 1000 mm so tracks sample the
    near-cathode region rather than the anode region.

    Direction: polar angle from the x-axis (drift direction) drawn uniformly in
    [30°, 150°] by sampling cos θ ∈ [−cos 30°, cos 30°] and φ ∈ [0, 2π). If the
    sampled direction points outward from the chosen face it is negated — valid
    because the angle range is symmetric around 90°, so negation maps θ → π−θ,
    which stays in [30°, 150°].

    Energy is drawn uniformly (continuous float) from [100, 1000] MeV.

    Returns list[dict] with keys: name, direction, momentum_mev, start_position_mm.
    """
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(1, n + 1):
        start_mm, outward_normal = _random_nice_face_entry_mm(rng, volumes)
        d = _sample_nice_direction(rng)
        dot = d[0]*outward_normal[0] + d[1]*outward_normal[1] + d[2]*outward_normal[2]
        if dot > 0.0:
            d = (-d[0], -d[1], -d[2])
        t_mev = float(rng.uniform(100.0, 1000.0))
        specs.append(dict(
            name=f'NiceMuon{i}_{int(round(t_mev))}MeV',
            direction=d,
            momentum_mev=t_mev,
            start_position_mm=start_mm,
        ))
    return specs


def generate_random_boundary_track(volumes, seed, *, min_x_mm: float = 1000.0):
    """Generate a single random boundary-track starting on a face of the inner box.

    The inner box spans x ∈ [−min_x_mm, +min_x_mm] and y/z over both active volumes.
    One of the 6 faces is chosen with probability proportional to its area, then a
    uniform point on that face is picked as the start position.

    Direction: theta (polar angle from x-axis / drift direction) drawn uniformly in
    [0, π], alpha (azimuthal around x-axis) uniformly in [0, 2π).  The resulting
    vector is flipped when it points outward from the chosen face so it always aims
    into the volume.

    Returns:
        (direction, start_position_mm) — each a tuple of 3 floats.
    """
    rng = np.random.default_rng(seed)
    east = volumes[0]
    y0, y1 = east.ranges_cm[1][0] * 10.0, east.ranges_cm[1][1] * 10.0
    z0, z1 = east.ranges_cm[2][0] * 10.0, east.ranges_cm[2][1] * 10.0
    y_span, z_span = y1 - y0, z1 - z0
    x_span = 2.0 * min_x_mm

    # 6 faces: 0=x−, 1=x+, 2=y−, 3=y+, 4=z−, 5=z+
    areas = np.array([
        y_span * z_span,   # x− face
        y_span * z_span,   # x+ face
        x_span * z_span,   # y− face
        x_span * z_span,   # y+ face
        x_span * y_span,   # z− face
        x_span * y_span,   # z+ face
    ])
    face = int(rng.choice(6, p=areas / areas.sum()))

    # Outward unit normals for each face
    normals = ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))

    if face == 0:   # x = -min_x_mm
        sx = -min_x_mm
        sy = float(rng.uniform(y0, y1))
        sz = float(rng.uniform(z0, z1))
    elif face == 1: # x = +min_x_mm
        sx = min_x_mm
        sy = float(rng.uniform(y0, y1))
        sz = float(rng.uniform(z0, z1))
    elif face == 2: # y = y0
        sx = float(rng.uniform(-min_x_mm, min_x_mm))
        sy = y0
        sz = float(rng.uniform(z0, z1))
    elif face == 3: # y = y1
        sx = float(rng.uniform(-min_x_mm, min_x_mm))
        sy = y1
        sz = float(rng.uniform(z0, z1))
    elif face == 4: # z = z0
        sx = float(rng.uniform(-min_x_mm, min_x_mm))
        sy = float(rng.uniform(y0, y1))
        sz = z0
    else:           # z = z1
        sx = float(rng.uniform(-min_x_mm, min_x_mm))
        sy = float(rng.uniform(y0, y1))
        sz = z1

    theta = float(rng.uniform(np.radians(25.0), np.pi))
    alpha = float(rng.uniform(0.0, 2.0 * np.pi))
    dx = float(np.cos(theta))
    dy = float(np.sin(theta) * np.cos(alpha))
    dz = float(np.sin(theta) * np.sin(alpha))

    nx, ny, nz = normals[face]
    if dx * nx + dy * ny + dz * nz > 0:
        dx, dy, dz = -dx, -dy, -dz

    return (dx, dy, dz), (sx, sy, sz)


def generate_random_boundary_tracks(
    volumes,
    n=N_DEFAULT_BOUNDARY_MUONS,
    seed=42,
    *,
    include_diagonal_cross_muon: bool = True,
):
    """Generate ``n`` random boundary-start muon specs, optionally plus three fixed chords.

    When ``include_diagonal_cross_muon`` is True, appends (1000 MeV each):

      * ``Muon_diagCross_1000MeV``: (2000,2000,2000) mm toward (−2000,−2000,−2000) mm;
      * ``Muon_throughEw_skew02_1000MeV`` — oblique chord from East side toward West;
      * ``Muon_throughWe_skew03_1000MeV`` — oblique chord from West side toward East.

    Each chord samples both drift volumes before energy loss terminates the propagated track
    inside active LAr (see ``particle_generator.generate_muon_track`` + filtering).

    Returns list[dict] with keys:
      - name, direction, momentum_mev, start_position_mm

    Random-track energies are balanced across ``(1000, 500, 100)`` as evenly as
    possible (default ``n=12`` gives ``4`` tracks per energy).
    """
    rng = np.random.default_rng(seed)
    specs = []
    random_track_energies = _balanced_boundary_energies_mev(rng, n)
    for i, t_mev in enumerate(random_track_energies, start=1):
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
        specs.extend(_oblique_through_both_volume_muons())
    return specs
