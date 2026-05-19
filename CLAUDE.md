## High-level repo description
This is a differentiable LArTPC simulation repo written in JAX. We experiment with simulation parameter calibration using src/scripts/run_optimization.py. The loss landscape is generated using scripts/jobs_Landscape.sh.
Simulation calibration jobs are submitted to slurm using src/jobs/submit_jobs.py.


## Running code
Locally, use the Python interpreter at .venv/bin/python. If the command should run on S3DF, just print it using the "/sdf/home/g/gregork/envs/base_env/bin/python" interpreter. Add important things you learned about the project to this file, so that you can use this context later.
If you run the sync_results_to_remote.sh script, the plots/<...> folders will be available for access by external collaborators on URL https://d3jk0djzcq11zh.cloudfront.net/<...>.

## Important insights

Add important insights about the repo that you learn here.

Track ensemble: ``tools/random_boundary_tracks.py`` — ``generate_random_boundary_tracks`` builds N random x-face muons plus **three** fixed 1000 MeV chords that span East+West: ``Muon_diagCross_1000MeV``, ``Muon_throughEw_skew02_1000MeV``, ``Muon_throughWe_skew03_1000MeV`` (see that file for start/toward mm). Random-track energies are now balanced as evenly as possible across ``(1000, 500, 100)``; default ``N=12`` gives exactly ``4`` tracks at each energy, so total default tracks remain ``15`` with the 3 fixed chords. ``launch_2d_landscape_pairs.py`` uses the same list and passes ``--start-position-mm`` into ``2d_loss_landscape.py``. In ``plot_mixed_tracks_edep_wireplanes.py``, out-of-volume steps are dropped before ``build_deposit_data``; Plotly ``Scatter3d`` uses ``marker.line.width=0`` to reduce white halos.
`tests/test_optimization_run.py` uses stubbed `jax/optax/tools.*` modules to import `src/opt/run_optimization.py`; when optimizer/run-trial signatures evolve, keep the stubs and compatibility branches in sync (notably `optax.chain`, `jax.tree_util.tree_map`, and the `phase_schedule`-based `run_trial` path).


## Syncing results to S3
Run "bash sync_results_to_remote.sh" outside the sandbox.
To invalidate the CloudFront cache, run 'aws cloudfront create-invalidation --distribution-id E3SPQV5ITLTD1U --paths "/*" '

## Syncing code to S3DF
Run "bash slac_sync_code.sh" outside the sandbox


