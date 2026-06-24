PLOTS_ONLY=0
for arg in "$@"; do
  [[ "$arg" == "--plots-only" ]] && PLOTS_ONLY=1
done

RECOMPUTE=""
[[ "$PLOTS_ONLY" -eq 0 ]] && RECOMPUTE="--recompute-bias"

# 1. Full mode (wire response)
/sdf/home/g/gregork/envs/base_env/bin/python tools/plot_diffusion_loss_study.py --mode full $RECOMPUTE
# 2. No-wire-response mode
/sdf/home/g/gregork/envs/base_env/bin/python tools/plot_diffusion_loss_study.py --mode no_wire_response $RECOMPUTE
# 3. Combined page — reads from the caches produced above, no recompute needed
/sdf/home/g/gregork/envs/base_env/bin/python tools/plot_diffusion_loss_study.py --mode combined




######## Run locally   #########
#python tools/plot_diffusion_study_tracks.py


