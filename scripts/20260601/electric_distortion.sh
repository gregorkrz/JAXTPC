python -m tools.efield_distortions --detector jaxtpc --Nxo 41 --Nyo 41 --Nzo 41 --output results/efield_distortions/sce_maps_jaxtpc_41.npz
python scripts/20260601/plot_efield_distortions.py results/efield_distortions/sce_maps_jaxtpc_41.npz

