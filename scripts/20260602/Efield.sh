
python src/analysis/sim_param_sweeps/generate_cutoff_sweep_viewer.py --results-dir $RESULTS_DIR --output $PLOTS_DIR/cutoff_sweep/cutoff_sweep_viewer.html

python tools/eval_efield_mlp.py --results-dir $RESULTS_DIR/opt/E_debug
python tools/plot_efield_eval.py --results-dir $RESULTS_DIR/opt/E_debug --output  $PLOTS_DIR/20260602/efield_eval.html

# Efield correction seeds 0-4 (runs pj4p1rhm/laizzjbs/x583jryx/oby6s79u/90y5ksc2), 13 tracks, fixbug2
E_CORRECTION_SEEDS_DIR=/fs/ddn/sdf/group/atlas/d/gregork/jaxtpc/results/opt/correction_seeds
python tools/eval_efield_mlp.py --results-dir $E_CORRECTION_SEEDS_DIR
python tools/plot_efield_eval.py --results-dir $E_CORRECTION_SEEDS_DIR --output $PLOTS_DIR/20260602/efield_eval_correction_seeds.html

