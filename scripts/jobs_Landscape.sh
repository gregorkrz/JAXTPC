
# Run on the cluster to launch Landscape jobs
 /sdf/home/g/gregork/envs/base_env/bin/python src/analysis/launch_2d_landscape_pairs.py --gradients --run-date 20260508
 /sdf/home/g/gregork/envs/base_env/bin/python src/analysis/launch_2d_landscape_pairs.py --gradients --noise-scale 1.0 --run-date 20260508_noise

# diffusion_trans vs diffusion_long: 100×100 grid ±50%, three losses, all 15 tracks, no gradients
# ~10k forward passes per track; 2h wall time
 /sdf/home/g/gregork/envs/base_env/bin/python src/analysis/launch_2d_landscape_pairs.py \
     --params diffusion_trans_cm2_us diffusion_long_cm2_us \
     --grid 100 \
     --range-frac 0.5 \
     --loss sobolev_loss_geomean_log1p,mse_loss,l1_loss \
     --time 02:00:00 \
     --run-date 20260511_diffusion_100x100

