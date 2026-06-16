#!/usr/bin/env python3
"""
Write an index.html linking to all trajectories_*.html files in a directory
(as produced by tools/plot_opt_trajectories.py).

Usage:
  python tools/generate_trajectories_index.py --output-dir $PLOTS_DIR/trajectories_Run_Opt_20260609
"""

import argparse
import glob
import os

_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body>
<h2>{title}</h2>
<ul>
{items}
</ul>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--title', default=None, help='Defaults to the output dir name')
    args = p.parse_args()

    htmls = sorted(glob.glob(os.path.join(args.output_dir, 'trajectories_*.html')))
    if not htmls:
        print(f'No trajectories_*.html files found in {args.output_dir}')
        return

    title = args.title or os.path.basename(os.path.normpath(args.output_dir))
    items = []
    for path in htmls:
        fname = os.path.basename(path)
        tag = fname[len('trajectories_'):-len('.html')]
        items.append(f'  <li><a href="{fname}">{tag}</a></li>')

    html = _TEMPLATE.format(title=title, items='\n'.join(items))
    out_path = os.path.join(args.output_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
