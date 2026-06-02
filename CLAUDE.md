## High-level repo description
This is a differentiable LArTPC simulation repo written in JAX. We experiment with simulation parameter calibration using src/scripts/run_optimization.py. The loss landscape is generated using scripts/jobs_Landscape.sh.
Simulation calibration jobs are submitted to slurm using src/jobs/submit_jobs.py.


## Running code
Locally, use the Python interpreter at .venv/bin/python. If the command should run on S3DF, just print it using the "/sdf/home/g/gregork/envs/base_env/bin/python" interpreter. Add important things you learned about the project to this file, so that you can use this context later.
If you run the sync_results_to_remote.sh script, the plots/<...> folders will be available for access by external collaborators on URL https://d3jk0djzcq11zh.cloudfront.net/<...>.

## Important insights

Add important insights about the repo that you learn here.

Track ensemble: ``tools/random_boundary_tracks.py`` — ``generate_random_boundary_tracks`` builds N random x-face muons plus **three** fixed 1000 MeV chords that span East+West: ``Muon_diagCross_1000MeV``, ``Muon_throughEw_skew02_1000MeV``, ``Muon_throughWe_skew03_1000MeV`` (see that file for start/toward mm). Random-track energies are now balanced as evenly as possible across ``(1000, 500, 100)``; default ``N=12`` gives exactly ``4`` tracks at each energy, so total default tracks remain ``15`` with the 3 fixed chords. ``launch_2d_landscape_pairs.py`` uses the same list and passes ``--start-position-mm`` into ``2d_loss_landscape.py``. In ``plot_mixed_tracks_edep_wireplanes.py``, out-of-volume steps are dropped before ``build_deposit_data``; Plotly ``Scatter3d`` uses ``marker.line.width=0`` to reduce white halos.

E-field distortions (SCE): ``tools/efield_distortions.py`` is the **non-differentiable** generator (analytic toy field + SciPy-Euler drift integration); its ``__main__`` writes a per-side ``.npz`` (``east_*``/``west_*`` keys). ``tools/nonlocal_efield.py`` is the **differentiable** MLP counterpart — ``mode='potential'`` learns a scalar potential and returns ``E = E_bg − ∇φ`` (conservative by construction); also ``efield``/``correction`` modes. Helpers: ``sce_outputs`` (→ ``SCEOutputs``-style ``efield_correction``/``drift_corr_cm``), ``nominal_start_params`` (random hidden + **zeroed output layer** → starts at nominal field yet has nonzero gradient; an all-zero MLP is dead — zero gradient everywhere), ``flatten_params``/``zero_params``. The MLP field lives in ``SimParams.sce_models`` (previously a declared-but-unused field). ``tools/simulation.py`` ``DetectorSimulator(..., efield_model=FieldConfig)`` builds an MLP SCE factory that reads weights from ``sim_params.sce_models``; ``sce_factory`` now takes ``sim_params`` (4 call sites). The MLP operates in volume-**local** frame (one MLP serves both volumes; nominal ``Ex_local=+E0``).

Optimizable Efield in ``run_optimization.py``: pass ``Efield`` in ``--params`` (combinable with scalars, e.g. ``--params velocity_cm_us,Efield``) plus ``--electric-dist-path <npz>`` (the GT distortion the MLP learns to match). GT sim runs WITH the static ``.npz`` map (``load_sce_per_volume`` now also reads ``.npz``, not just ``.h5``); the diff sim learns the MLP. Options: ``--efield-mode {potential,efield,correction}``, ``--efield-hidden``, ``--efield-lr-mult``. The optimizer vector is ``[scalar block (log-normalized) | flat MLP weights (raw)]``; only the scalar coords are recorded in the param/grad trajectories (full ``final_p`` saved + ``result['efield']`` metadata). Not compatible with ``--optimizer newton`` or ``--init-from-wandb-run``. Per-step cost is dominated by the deposit-level forward+backward, NOT the MLP (8.6k weights ≈ free).

``run_optimization.py`` was refactored: helper groups now live in the top-level package ``src/optlib/`` (``constants.py``, ``parsing.py`` [incl. ``parse_args(doc)``], ``params.py``, ``paths.py``, ``optim.py``, ``wandb_utils.py``). The driver adds ``src/`` to ``sys.path`` and imports them; it still owns ``main``, ``run_trial``, ``build_loss_fn``/``build_phase_fns``, ``apply_noise_to_gt``. ``losses.py``/``trial.py`` extraction is a deferred follow-up.


## Syncing results to S3
Run "bash sync_results_to_remote.sh" outside the sandbox.
To invalidate the CloudFront cache, run 'aws cloudfront create-invalidation --distribution-id E3SPQV5ITLTD1U --paths "/*" '

## Syncing code to S3DF
Run "bash slac_sync_code.sh" outside the sandbox

