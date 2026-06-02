"""
Plot SCE maps produced by tools/efield_distortions.py.

Usage
-----
    python3 scripts/20260601/plot_efield_distortions.py <sce_maps.npz>

Output
------
    <sce_maps>_plots.pdf  — two pages saved next to the input file.
    Page 1: E-field components (Ex, Ey, Ez) for east and west TPC sides.
    Page 2: Drift corrections (Δx, Δy, Δz) for east and west TPC sides.

Each 2D map is a (x, y) slice at the z midpoint; Ez and Δz use an (x, z)
slice at the y midpoint to show the orthogonal transverse component.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _grid_axes(origin, spacing, shape):
    """Return (x_arr, y_arr, z_arr) coordinate arrays for a volume grid."""
    nx, ny, nz = shape
    x = origin[0] + np.arange(nx) * spacing[0]
    y = origin[1] + np.arange(ny) * spacing[1]
    z = origin[2] + np.arange(nz) * spacing[2]
    return x, y, z


def _imshow(ax, data2d, x_arr, y_arr, xlabel, ylabel, title, cmap, vmin=None, vmax=None, cbar_label=""):
    """Thin wrapper around imshow with physical axis labels."""
    im = ax.imshow(
        data2d.T,
        origin="lower",
        aspect="auto",
        extent=[x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=7)


def _make_efield_page(fig, east_ef, east_orig, east_sp, west_ef, west_orig, west_sp):
    """Fill fig with E-field component maps (2 rows × 3 cols)."""
    axes = fig.subplots(2, 3)
    fig.suptitle("E-field maps", fontsize=12, y=1.01)

    for row, (ef, orig, sp, label) in enumerate([
        (east_ef, east_orig, east_sp, "East TPC"),
        (west_ef, west_orig, west_sp, "West TPC"),
    ]):
        nx, ny, nz, _ = ef.shape
        x, y, z = _grid_axes(orig, sp, (nx, ny, nz))
        iz_mid = nz // 2
        iy_mid = ny // 2

        # Ex(x, y) at z mid — longitudinal component
        vabs = np.max(np.abs(ef[:, :, iz_mid, 0]))
        _imshow(axes[row, 0], ef[:, :, iz_mid, 0], x, y,
                "x (cm)", "y (cm)", f"{label} — Ex(x,y) z=0",
                "RdBu_r", vmin=-vabs, vmax=vabs, cbar_label="V/cm")

        # Ey(x, y) at z mid — transverse
        vabs = np.max(np.abs(ef[:, :, iz_mid, 1]))
        _imshow(axes[row, 1], ef[:, :, iz_mid, 1], x, y,
                "x (cm)", "y (cm)", f"{label} — Ey(x,y) z=0",
                "RdBu_r", vmin=-vabs, vmax=vabs, cbar_label="V/cm")

        # Ez(x, z) at y mid — other transverse
        vabs = np.max(np.abs(ef[:, iy_mid, :, 2]))
        _imshow(axes[row, 2], ef[:, iy_mid, :, 2], x, z,
                "x (cm)", "z (cm)", f"{label} — Ez(x,z) y=0",
                "RdBu_r", vmin=-vabs, vmax=vabs, cbar_label="V/cm")

    fig.tight_layout()


def _make_corrections_page(fig, east_corr, east_orig, east_sp, west_corr, west_orig, west_sp):
    """Fill fig with drift correction maps (2 rows × 3 cols)."""
    axes = fig.subplots(2, 3)
    fig.suptitle("Drift corrections", fontsize=12, y=1.01)

    for row, (corr, orig, sp, label) in enumerate([
        (east_corr, east_orig, east_sp, "East TPC"),
        (west_corr, west_orig, west_sp, "West TPC"),
    ]):
        nx, ny, nz, _ = corr.shape
        x, y, z = _grid_axes(orig, sp, (nx, ny, nz))
        iz_mid = nz // 2
        iy_mid = ny // 2

        for col, (comp, slice2d, xa, ya, title_xy) in enumerate([
            (0, corr[:, :, iz_mid, 0], x, y, f"{label} — Δx(x,y) z=0"),
            (1, corr[:, :, iz_mid, 1], x, y, f"{label} — Δy(x,y) z=0"),
            (2, corr[:, iy_mid, :, 2], x, z, f"{label} — Δz(x,z) y=0"),
        ]):
            ylabel = "y (cm)" if col < 2 else "z (cm)"
            vabs = max(np.max(np.abs(slice2d)), 1e-6)
            _imshow(axes[row, col], slice2d, xa, ya,
                    "x (cm)", ylabel, title_xy,
                    "RdBu_r", vmin=-vabs, vmax=vabs, cbar_label="cm")

    fig.tight_layout()


def main(npz_path):
    data = np.load(npz_path)

    east_ef   = data["east_efield"]
    east_orig = data["east_origin"]
    east_sp   = data["east_spacing"]
    west_ef   = data["west_efield"]
    west_orig = data["west_origin"]
    west_sp   = data["west_spacing"]
    east_corr = data["east_corrections"]
    west_corr = data["west_corrections"]

    out_path = os.path.splitext(npz_path)[0] + "_plots.pdf"
    with PdfPages(out_path) as pdf:
        fig1 = plt.figure(figsize=(14, 8))
        _make_efield_page(fig1, east_ef, east_orig, east_sp, west_ef, west_orig, west_sp)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = plt.figure(figsize=(14, 8))
        _make_corrections_page(fig2, east_corr, east_orig, east_sp, west_corr, west_orig, west_sp)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(f"[plot_efield_distortions] saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <sce_maps.npz>")
        sys.exit(1)
    main(sys.argv[1])
