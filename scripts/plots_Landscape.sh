

# Build interactive HTML (covers both 20260508 and 20260508_noise — noise toggled via UI checkbox)
python src/plots/plot_landscape_interactive.py \
    --landscape-dir results/landscape \
    --run-dates 20260508 20260508_noise \
    --output plots/landscape_interactive_20260508.html

# Link: https://d3jk0djzcq11zh.cloudfront.net/landscape_interactive_20260508.html
