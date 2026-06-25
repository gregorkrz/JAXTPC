"""CLI argument parsing and spec parsers.

Extracted verbatim from ``src/opt/run_optimization.py`` (no logic changes), except
``parse_args`` now takes the help ``doc`` explicitly (its description used to be the
driver module's ``__doc__``).
"""

import argparse

from optlib.constants import (
    VALID_PARAMS, VALID_LOSSES, VALID_OPTIMIZERS, TRACK_PRESETS, PLANE_NAME_MAP,
    ADAM_BETA2, GT_LIFETIME_US, _RESULTS_DIR,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(doc=None):
    p = argparse.ArgumentParser(description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--params', required=True,
                   help='Comma-separated params to optimize, e.g. recomb_alpha,recomb_beta_90')
    p.add_argument('--range', type=float, nargs='+', metavar='VAL', default=(0.95, 1.05),
                   help='One or more lo/hi pairs defining relative-factor intervals for random '
                        'starting points. Must be an even count. A single pair "LO HI" draws '
                        'uniformly from [LO, HI]. Multiple pairs e.g. "0.8 0.9 1.1 1.2" draw '
                        'from the union [0.8,0.9]∪[1.1,1.2] (default: 0.95 1.05)')
    # ── Efield (MLP E-field distortion) options ──
    p.add_argument('--electric-dist-path', default=None,
                   help='Path to the ground-truth SCE distortion map (.npz from '
                        'tools.efield_distortions, or .h5). Required when "Efield" is in --params: '
                        'the GT simulator runs WITH this distortion and the MLP learns to match it.')
    p.add_argument('--efield-mode', default='potential',
                   choices=('potential', 'efield', 'correction'),
                   help='MLP parameterization for the Efield model (default: potential, '
                        'i.e. conservative E=-grad(phi)).')
    p.add_argument('--efield-hidden', type=int, nargs='+', default=[64, 64, 64],
                   help='Hidden layer widths for the Efield MLP (default: 64 64 64).')
    p.add_argument('--efield-lr-mult', type=float, default=1.0,
                   help='LR multiplier applied to all Efield MLP weights (default: 1.0). '
                        'MLP weights typically need a different step size than physics scalars.')
    p.add_argument('--efield-per-volume', action='store_true', default=False,
                   help='Use separate MLP weights for each drift volume (east + west) '
                        'instead of one shared model. Doubles the MLP parameter count.')
    p.add_argument('--mlp-snapshot-interval', type=int, default=0,
                   help='Save the full MLP weight vector every N steps into mlp_trajectory '
                        '(0 = disabled, only final_p is saved). Useful for visualising '
                        'how the learned field evolves during training.')
    p.add_argument('--tracks', default='diagonal',
                   help='"+"-separated track presets or name:dx,dy,dz:mom_mev specs '
                        '("+" separates tracks, "," separates direction components). '
                        f'Default: diagonal.  Presets: {", ".join(TRACK_PRESETS)}')
    p.add_argument('--N-random-tracks', type=int, default=0, metavar='N',
                   help='If > 0, ignore --tracks and generate N random near-cathode tracks '
                        'via generate_random_nice_tracks: y/z-face entries with |x| < 1000 mm, '
                        'polar angle from x-axis in [30°, 150°], T ~ U[100, 1000] MeV.')
    p.add_argument('--tracks-random-seed', type=int, default=7, metavar='SEED',
                   help='RNG seed for --N-random-tracks (default: 7).')
    p.add_argument('--events-file', default=None, metavar='PATH',
                   help='Load events from an HDF5 file (muon.h5-compatible pstep/lar_vol '
                        'format; see tools/event_io.py and generate_muon_tracks.py) instead '
                        'of generating tracks from --tracks/--N-random-tracks. Each event\'s '
                        'deposits are used as-is (already at a fixed step size) for both the '
                        'GT and all forward phases — --gt-step-size and per-phase step sizes '
                        'are ignored. Mutually exclusive with --tracks/--N-random-tracks.')
    p.add_argument('--loss', default='sobolev_loss_geomean_log1p', choices=VALID_LOSSES,
                   help='Loss function (default: sobolev_loss_geomean_log1p)')
    p.add_argument('--optimizer', default='adam', choices=VALID_OPTIMIZERS,
                   help='Optimizer (default: adam)')
    p.add_argument('--lr', type=float, default=0.01,
                   help='Peak learning rate (default: 0.01)')
    p.add_argument('--lr-schedule', default='cosine', choices=('constant', 'cosine'),
                   help='LR schedule: constant or cosine decay over --max-steps (default: constant)')
    p.add_argument('--max-steps', type=int, default=200,
                   help='Max gradient steps per trial (default: 200)')
    p.add_argument('--tol', type=float, default=1e-5,
                   help='Early-stop relative tolerance on p_n norm (default: 1e-5)')
    p.add_argument('--patience', type=int, default=20,
                   help='Steps over which relative change is checked (default: 20)')
    p.add_argument('--tol-per-param', type=float, default=None,
                   metavar='TOL',
                   help='With --patience-per-param: freeze when relative change from t-W to '
                        'now and every step-to-step change in the window are all < TOL '
                        '(default: disabled).')
    p.add_argument('--patience-per-param', type=int, default=None,
                   metavar='STEPS',
                   help='Window length W for --tol-per-param: compare to t-W and check each '
                        'of the W consecutive updates (default: disabled). '
                        'Set both flags together to enable per-parameter freezing.')
    p.add_argument('--phase2-params', default=None,
                   help='Comma-separated subset of --params. Before --phase2-start-step, '
                        'these params are frozen (zero gradient) and all other --params are '
                        'optimized normally. From --phase2-start-step onward, only these '
                        'params receive gradient updates and all other --params are frozen. '
                        'Must be a strict, non-empty subset of --params. '
                        'Set together with --phase2-start-step (default: disabled, all '
                        'params optimized throughout). Not compatible with --optimizer newton.')
    p.add_argument('--phase2-start-step', type=int, default=None,
                   metavar='STEP',
                   help='Step at which to switch to optimizing only --phase2-params '
                        '(required together with --phase2-params).')
    p.add_argument('--N', type=int, default=25,
                   help='Number of random trials (default: 25)')
    p.add_argument('--results-base', default=_RESULTS_DIR,
                   help='Base directory; output goes to <results-base>/<folder>/ (default: $RESULTS_DIR or results)')
    p.add_argument('--seed', type=int, default=None,
                   help='Master RNG seed (default: random). Seeds everything: '
                        'trial starting points and GT noise draw. '
                        'The resolved seed is printed and stored in the pkl.')
    p.add_argument('--noise-scale', type=float, default=0.0,
                   help='Noise amplitude as a multiple of the calibrated detector noise '
                        '(MicroBooNE model, converted to signal units via electrons_per_adc). '
                        '0.0 = no noise (default), 1.0 = realistic noise. '
                        'Signal and noise RMS are printed at startup for reference.')
    p.add_argument('--rotate-noise-seeds', type=int, default=-1,
                   metavar='N',
                   help='Cycle through N distinct noise realisations across optimizer steps. '
                        'At step s the noise seed index is s %% N, so each seed recurs every '
                        'N steps. -1 (default) disables rotation (same seed every step). '
                        'Has no effect when --noise-scale 0.')
    p.add_argument('--warmup-steps', type=int, default=100,
                   help='Linear LR warmup from 0 to --lr over this many steps '
                        '(default: 100, set to 0 to disable).')
    p.add_argument('--clip-grad-norm', type=float, default=10.0,
                   help='If > 0, rescale the full gradient vector so its L2 norm is at most '
                        'this value (global norm clip; default: 10.0). Set to 0 to disable.')
    p.add_argument('--lr-multipliers', default=None,
                   help='Per-parameter LR multipliers as comma-separated name:factor pairs, '
                        'e.g. "velocity_cm_us:0.01,lifetime_us:0.1". '
                        'Unlisted parameters keep multiplier 1.0. '
                        'Use "auto" to set each multiplier once from |dL/dp| (median-scaled, '
                        'clipped to [0.01, 10]); see --lr-mult-auto-burn-in-steps. '
                        'Values are stored in the result pickle for resume.')
    p.add_argument('--lr-mult-auto-burn-in-steps', type=int, default=100,
                   help='With --lr-multipliers auto: run this many optimizer steps first '
                        '(same LR/clip/warmup/schedule as trials, but no per-param grad scaling) '
                        'and set each multiplier from the mean |dL/dp_i| over those steps, so '
                        'multi-phase schedules are exercised by step index. '
                        '0 = use a single summed grad at trial start (step 0). Default: 100.')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Number of tracks processed together on GPU per grad call (default: 1). '
                        'Larger values use vmap to parallelize tracks; try 2–4.')
    p.add_argument('--effective-batch-size', type=int, default=1,
                   help='Number of consecutive micro-batches to accumulate before one optimizer '
                        'update (default: 1). This increases effective batch size without '
                        'holding all tracks in memory at once.')
    p.add_argument('--step-size', type=float, default=0.1,
                   help='Muon track step size in mm (default: 0.1). '
                        'Larger values reduce deposit count and memory use.')
    p.add_argument('--max-num-deposits', type=int, default=50_000,
                   help='Static deposit buffer size passed to the differentiable simulator '
                        'as n_segments (default: 50000). Must be >= actual deposits per track.')
    p.add_argument('--num-buckets', type=int, default=1000,
                   help='Max active buckets for non-differentiable bucketed accumulation '
                        '(default: 1000). Increase if you see bucket overflow warnings.')
    p.add_argument('--schedule-steps', default=None,
                   help='Comma-separated step thresholds that divide optimization into phases '
                        '(e.g. "1000" → 2 phases; "1000,5000" → 3 phases).')
    p.add_argument('--schedule-step-sizes', default=None,
                   help='Comma-separated step sizes in mm, one per phase (e.g. "1.0,0.1").')
    p.add_argument('--schedule-deposits', default=None,
                   help='Comma-separated max-num-deposits, one per phase (e.g. "5000,50000").')
    p.add_argument('--schedule-batch-sizes', default=None,
                   help='Comma-separated batch sizes, one per phase (e.g. "5,1").')
    p.add_argument('--gt-step-size', type=float, default=0.1,
                   help='Step size in mm used to generate GT signals (default: 0.1). '
                        'Independent of the forward simulation schedule.')
    p.add_argument('--gt-max-deposits', type=int, default=50_000,
                   help='Static deposit buffer for the GT simulator (default: 50000). '
                        'Must be >= actual deposits per track at --gt-step-size.')
    p.add_argument('--no-wandb', action='store_true',
                   help='Disable Weights & Biases logging (enabled by default).')
    p.add_argument('--wandb-project', default='jaxtpc-optimization',
                   help='W&B project name (default: jaxtpc-optimization).')
    p.add_argument('--wandb-tags', default=None,
                   help='Comma-separated W&B run tags (e.g. "sched_v2,fine_stage").')
    p.add_argument('--log-interval', type=int, default=50,
                   help='Log to W&B every this many steps (default: 50).')
    p.add_argument('--newton-damping', type=float, default=1e-3,
                   help='Damping for Newton optimizer (lambda in H + lambda*I). Default 1e-3.')
    p.add_argument('--adam-beta2', type=float, default=ADAM_BETA2,
                   help=f'Adam beta2 (second-moment decay). Default {ADAM_BETA2}.')
    p.add_argument('--init-from-wandb-run', default=None, metavar='RUN_ID',
                   help='Start trial 0 from param values of an existing W&B run '
                        '(fetches params/<name>_physical). '
                        'Remaining trials use random starts as usual.')
    p.add_argument('--init-from-wandb-step', type=int, default=-1, metavar='STEP',
                   help='Step to read from --init-from-wandb-run. '
                        '-1 (default) uses the run summary (last logged value). '
                        'A non-negative value fetches that exact logged step.')
    p.add_argument('--gt-param-multiplier', type=float, default=1.0,
                   help='Multiply all optimized GT parameter values by this factor before '
                        'generating the reference signal (default: 1.0, i.e. no shift). '
                        'Use 1.2 to shift the true parameters 20%% upward.')
    p.add_argument('--gt-lifetime-us', type=float, default=None,
                   help='Override the GT electron lifetime in μs (default: use GT_LIFETIME_US '
                        f'= {GT_LIFETIME_US:.0f} μs). E.g. 6000 for 6 ms.')
    p.add_argument('--sobolev-exponent', type=float, default=2.0,
                   help='Sobolev exponent s passed to make_sobolev_weight (default: 2.0). '
                        'Higher values penalise high-frequency components more strongly.')
    p.add_argument('--freq-cutoff', type=float, default=None,
                   help='Hard high-frequency cutoff for Sobolev weights, in normalised units '
                        '(0, 0.5]. Frequencies with |f| > cutoff (L2 norm) are zeroed. '
                        'None disables (default).')
    p.add_argument('--freq-cutoff-per-param', default=None,
                   help='Per-parameter freq cutoff: "param1:cutoff1,param2:cutoff2". '
                        'Params not listed use --freq-cutoff (default None = no cutoff). '
                        'Not compatible with --optimizer newton.')
    p.add_argument('--fourier-cutoff', type=float, default=0.0,
                   help='Signal-power Fourier cutoff (ADC²): zero spectral bins where '
                        '|FFT(gt)|^2/N < cutoff. Same convention as 1d_gradients.py. '
                        '0 disables (default).')
    p.add_argument('--fourier-cutoff-per-param', default=None,
                   help='Per-parameter Fourier power cutoff: "param1:cutoff1,param2:cutoff2". '
                        'Params not listed use --fourier-cutoff (default 0 = no cutoff). '
                        'Not compatible with --optimizer newton.')
    p.add_argument('--sobolev-loss-cutoff', type=float, default=0.0,
                   help='ADC cutoff: zero out pixels where |gt| < cutoff before computing loss '
                        '(default: 0.0, no cutoff). Applied to both sim and gt signals.')
    p.add_argument('--sobolev-loss-cutoff-per-param', default=None,
                   help='Per-parameter ADC cutoff: "param1:cutoff1,param2:cutoff2". '
                        'Params not listed use --sobolev-loss-cutoff (default 0.0). '
                        'Each param\'s gradient is computed from a loss masked with its own cutoff. '
                        'Not compatible with --optimizer newton.')
    p.add_argument('--planes-per-param', default=None,
                   help='Per-parameter active planes: "param1:Y1,Y2%%param2:U1,V1". '
                        'Uses "%%" as outer separator (between param entries) and ":" between '
                        'param name and plane spec. Planes spec uses the same format as --planes. '
                        'Params not listed use --planes (default: all planes). '
                        'Not compatible with --optimizer newton.')
    p.add_argument('--start-position-mm', default='0,0,0',
                   help='Track start position as "x,y,z" in mm, applied to all tracks '
                        '(default: 0,0,0)')
    p.add_argument('--planes', default=None,
                   help='Comma-separated plane names or indices to include in the loss '
                        '(default: all planes). Names: U1,V1,Y1 (vol 0) / U2,V2,Y2 (vol 1); '
                        'U,V,Y selects both volumes. Integer indices 0-5 also accepted. '
                        'Example: --planes V1,V2')
    return p.parse_args()


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_params(params_str):
    names = [n.strip() for n in params_str.split(',') if n.strip()]
    if not names:
        raise ValueError('--params is empty')
    for name in names:
        if name not in VALID_PARAMS:
            raise ValueError(f'Unknown param {name!r}. Choose from: {VALID_PARAMS}')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate param names in --params')
    return names


def parse_tracks(tracks_str):
    """Parse '+'-separated preset names or name:dx,dy,dz:mom_mev[:x,y,z] specs.

    '+' separates tracks; ',' is used only inside direction/position components.
    Mixed input is supported, e.g. 'diagonal+mytrack:0.1,0.2,0.9:500'.
    Optional 4th field 'x,y,z' sets a per-track start position in mm.
    Returns list of dicts: {name, direction (tuple), momentum_mev, start_position_mm (tuple or None)}.
    """
    specs = []
    for item in tracks_str.split('+'):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            # Full spec: name:dx,dy,dz:momentum_mev[:x,y,z]
            parts = item.split(':')
            if len(parts) not in (3, 4):
                raise ValueError(
                    f'Custom track must be name:dx,dy,dz:momentum_mev or '
                    f'name:dx,dy,dz:momentum_mev:x,y,z, got {item!r}')
            name = parts[0].strip()
            try:
                direction = tuple(float(x) for x in parts[1].split(','))
            except ValueError:
                raise ValueError(f'Bad direction in track spec {item!r}')
            if len(direction) != 3:
                raise ValueError(f'Direction must have 3 components in {item!r}')
            try:
                momentum_mev = float(parts[2])
            except ValueError:
                raise ValueError(f'Bad momentum in track spec {item!r}')
            start_position_mm = None
            if len(parts) == 4:
                try:
                    start_position_mm = tuple(float(x) for x in parts[3].split(','))
                except ValueError:
                    raise ValueError(f'Bad start position in track spec {item!r}')
                if len(start_position_mm) != 3:
                    raise ValueError(f'Start position must have 3 components in {item!r}')
            specs.append(dict(name=name, direction=direction, momentum_mev=momentum_mev,
                               start_position_mm=start_position_mm))
        else:
            # Preset name
            if item not in TRACK_PRESETS:
                raise ValueError(
                    f'Unknown track preset {item!r}. '
                    f'Known: {list(TRACK_PRESETS)}. '
                    f'Use name:dx,dy,dz:mom_mev for custom tracks.')
            direction, momentum_mev = TRACK_PRESETS[item]
            specs.append(dict(name=item, direction=direction, momentum_mev=momentum_mev,
                               start_position_mm=None))
    if not specs:
        raise ValueError('--tracks produced no entries')
    return specs


def parse_planes(planes_str, n_planes=6):
    """Parse comma-separated plane names/indices into a sorted tuple of global plane indices.

    Names: U1,V1,Y1 (volume 0) / U2,V2,Y2 (volume 1); U,V,Y expand to both volumes.
    Integer indices 0-5 are also accepted.  None/empty string → all planes.
    """
    if not planes_str:
        return tuple(range(n_planes))
    indices: set = set()
    for tok in planes_str.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok in PLANE_NAME_MAP:
            indices.update(PLANE_NAME_MAP[tok])
        elif tok.lstrip('-').isdigit():
            idx = int(tok)
            if not (0 <= idx < n_planes):
                raise ValueError(f'Plane index {idx} out of range (0-{n_planes - 1})')
            indices.add(idx)
        else:
            raise ValueError(
                f'Unknown plane {tok!r}. Known names: {sorted(PLANE_NAME_MAP)}, '
                f'or use integer indices 0-{n_planes - 1}.')
    if not indices:
        raise ValueError('--planes produced no entries')
    return tuple(sorted(indices))


def parse_lr_multipliers(spec, param_names):
    """Parse 'name:factor,...' string into a per-parameter scale vector (length = len(param_names)).

    Unlisted parameters default to 1.0.
    """
    scales = [1.0] * len(param_names)
    if not spec:
        return scales
    if spec.strip().lower() == 'auto':
        raise ValueError('parse_lr_multipliers: use resolve path for "auto" (caller handles this)')
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, factor = item.split(':')
        name = name.strip()
        if name not in param_names:
            raise ValueError(f'--lr-multipliers: unknown param {name!r}. Known: {param_names}')
        scales[param_names.index(name)] = float(factor)
    return scales


def parse_cutoff_per_param(spec, param_names, default_cutoff=0.0):
    """Parse 'name:cutoff,...' into a per-parameter cutoff list (length = len(param_names)).

    Unlisted parameters default to default_cutoff.
    """
    cutoffs = [default_cutoff] * len(param_names)
    if not spec:
        return cutoffs
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split(':')
        if len(parts) != 2:
            raise ValueError(
                f'--sobolev-loss-cutoff-per-param: bad spec {item!r} (expected name:cutoff)')
        name, val = parts[0].strip(), parts[1].strip()
        if name not in param_names:
            raise ValueError(
                f'--sobolev-loss-cutoff-per-param: unknown param {name!r}. Known: {param_names}')
        cutoffs[param_names.index(name)] = float(val)
    return cutoffs


def parse_planes_per_param(spec, param_names, n_planes=6, default_planes=None):
    """Parse 'param1:Y1,Y2%param2:U1,V1' into per-parameter active_planes tuples.

    Uses '%' as outer separator (between param entries) and ':' between name and plane spec.
    The plane spec itself uses ',' (same format as --planes).
    Params not listed default to default_planes (None = all planes).
    Returns a list of (sorted tuple of int) or None, one entry per param.
    """
    result = [default_planes] * len(param_names)
    if not spec:
        return result
    for entry in spec.split('%'):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError(
                f'--planes-per-param: bad entry {entry!r} (expected param_name:plane_spec)')
        name, planes_str = entry.split(':', 1)
        name = name.strip()
        if name not in param_names:
            raise ValueError(
                f'--planes-per-param: unknown param {name!r}. Known: {param_names}')
        result[param_names.index(name)] = parse_planes(planes_str.strip(), n_planes=n_planes)
    return result


def parse_schedule(args):
    """Return a list of phase dicts: {step_size, max_num_deposits, batch_size, until_step}.

    Single-phase (no --schedule-steps) returns a one-element list using the
    top-level --step-size / --max-num-deposits / --batch-size values.
    """
    if args.schedule_steps is None:
        return [dict(step_size=args.step_size,
                     max_num_deposits=args.max_num_deposits,
                     batch_size=args.batch_size,
                     until_step=args.max_steps)]

    thresholds = [int(x.strip()) for x in args.schedule_steps.split(',')]
    n_phases   = len(thresholds) + 1

    def _csv(s, typ, default):
        if s is None:
            return [typ(default)] * n_phases
        vals = [typ(x.strip()) for x in s.split(',')]
        if len(vals) != n_phases:
            raise ValueError(
                f'Expected {n_phases} comma-separated values (got {len(vals)}): {s!r}')
        return vals

    step_sizes  = _csv(args.schedule_step_sizes,  float, args.step_size)
    deposits    = _csv(args.schedule_deposits,     int,   args.max_num_deposits)
    batch_sizes = _csv(args.schedule_batch_sizes,  int,   args.batch_size)

    until_steps = thresholds + [args.max_steps]
    return [dict(step_size=ss, max_num_deposits=dep, batch_size=bs, until_step=us)
            for ss, dep, bs, us in zip(step_sizes, deposits, batch_sizes, until_steps)]
