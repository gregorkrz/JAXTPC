"""Read/write event files of per-step particle deposits (muon.h5-compatible).

Storage format mirrors the ``pstep/lar_vol`` layout found in GEANT4-derived
files like ``muon.h5``: a ``(n_events,)`` object-dtype dataset where each
element is a structured array with one row per deposit step. Each event may
contain any number of tracks (distinguished by ``track_id``) — there is no
assumption of a single track per event, so files written by
``generate_muon_tracks.py`` (single-track events) and real multi-track GEANT4
output can be read by the same loader.

Fields read (all per-step, same length): x, y, z, theta, phi, de, dx,
track_id. ``ancestor_track_id`` and ``pdg`` are read if present and passed
through to ``build_deposit_data``; otherwise they default to zeros there.

An optional top-level ``event_names`` dataset (variable-length UTF-8 strings,
shape ``(n_events,)``) gives each event a human-readable name; real GEANT4
files lacking it fall back to ``event_<idx>``.
"""

import h5py
import numpy as np

_STEP_FIELDS = ('x', 'y', 'z', 'theta', 'phi', 'de', 'dx', 'track_id')
_OPTIONAL_STEP_FIELDS = ('ancestor_track_id', 'pdg')


def save_events_h5(path, events):
    """Write a list of single/multi-track events to an HDF5 file.

    events : list of dict, each with keys: name (str), and per-step arrays
        x, y, z, theta, phi, de, dx, track_id (all shape (N_i,), N_i may
        differ per event). Optional: ancestor_track_id, pdg.
    """
    dtype_fields = [(f, '<f4') for f in ('x', 'y', 'z', 'theta', 'phi', 'de', 'dx')]
    dtype_fields += [('track_id', '<i4')]
    for f in _OPTIONAL_STEP_FIELDS:
        if any(f in e for e in events):
            dtype_fields.append((f, '<i4'))
    dtype = np.dtype(dtype_fields)

    with h5py.File(path, 'w') as f:
        grp = f.create_group('pstep')
        ds = grp.create_dataset('lar_vol', shape=(len(events),), dtype=h5py.special_dtype(vlen=dtype))
        for i, ev in enumerate(events):
            n = len(ev['de'])
            rec = np.zeros(n, dtype=dtype)
            for name in dtype.names:
                rec[name] = ev.get(name, np.zeros(n))
            ds[i] = rec
        names_ds = f.create_dataset('event_names', shape=(len(events),),
                                    dtype=h5py.string_dtype(encoding='utf-8'))
        for i, ev in enumerate(events):
            names_ds[i] = ev.get('name', f'event_{i}')


def load_events_h5(path):
    """Read events back as a list of dicts compatible with build_deposit_data.

    Returns list of dict with keys: name, position (N,3) float32, de, dx,
    theta, phi, track_id, and (if present in the file) ancestor_track_id, pdg.
    """
    events = []
    with h5py.File(path, 'r') as f:
        steps = f['pstep/lar_vol']
        n_events = steps.shape[0]
        names = f['event_names'][:] if 'event_names' in f else None
        for i in range(n_events):
            rec = steps[i]
            name_i = names[i] if names is not None else f'event_{i}'
            if isinstance(name_i, bytes):
                name_i = name_i.decode('utf-8')
            ev = dict(
                name=name_i,
                position=np.stack([rec['x'], rec['y'], rec['z']], axis=1).astype(np.float32),
                de=rec['de'].astype(np.float32),
                dx=rec['dx'].astype(np.float32),
                theta=rec['theta'].astype(np.float32),
                phi=rec['phi'].astype(np.float32),
                track_id=rec['track_id'].astype(np.int32),
            )
            for opt in _OPTIONAL_STEP_FIELDS:
                if opt in rec.dtype.names:
                    ev[opt] = rec[opt]
            events.append(ev)
    return events
