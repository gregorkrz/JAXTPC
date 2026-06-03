
/sdf/home/g/gregork/envs/base_env/bin/python src/analysis/sim_param_sweeps/generate_cutoff_sweep_viewer.py --results-dir $RESULTS_DIR --output $PLOTS_DIR/cutoff_sweep/cutoff_sweep_viewer.html

python tools/eval_efield_mlp.py --results-dir $RESULTS_DIR/opt/E_debug
python tools/plot_efield_eval.py --results-dir $RESULTS_DIR/opt/E_debug --output  $PLOTS_DIR/20260602/efield_eval.html

