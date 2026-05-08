Use the Python interpreter at .venv/bin/python. If the command should run on S3DF, just print it using the "python" interpregter. Add important things you learned about the project to this file, so that you can use this context later.
If you run the sync_results_to_remote.sh script, the results/ and plots/ folders will be available for access by external collaborators on URL https://static.gregor.science/55ZKUmPYcUSwDfW5st/JAXTPC/<results/plots>/...

Track ensemble: ``tools/random_boundary_tracks.py`` — ``generate_random_boundary_tracks`` builds N random x-face muons plus one fixed ``Muon_diagCross_1000MeV`` (start (2000,2000,2000) mm toward (−2000,−2000,−2000) mm). ``launch_2d_landscape_pairs.py`` uses the same list and passes ``--start-position-mm`` into ``2d_loss_landscape.py``. In ``plot_mixed_tracks_edep_wireplanes.py``, out-of-volume steps are dropped before ``build_deposit_data``; Plotly ``Scatter3d`` uses ``marker.line.width=0`` to reduce white halos.

