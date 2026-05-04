"""
JAXTPC Tools Package

Simulation tools for LArTPC detector simulation.

Main entry points:
- generate_detector: Parse YAML detector configuration
- DetectorSimulator: Main simulation class
  - process_event(): Production path (fori_loop batching, post-processing)
  - forward(): Differentiable path (gradients through SimParams)
- load_particle_step_data: Load HDF5 particle step data
- generate_noise / add_noise: Standalone noise generation (accepts SimConfig)
"""
from importlib import import_module

__all__ = []


def _optional_export(module_path, names):
    """Import names when available, skipping optional/heavy dependencies."""
    try:
        module = import_module(module_path)
    except Exception:
        return

    for name in names:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)
            __all__.append(name)


_optional_export(
    "tools.config",
    [
        # Core types
        "DepositData",
        "VolumeDeposits",
        "SimParams",
        "SimConfig",
        "ModifiedBoxParams",
        "EMBParams",
        "SCEOutputs",
        "VolumeGeometry",
        "VolumeIntermediates",
        "PlaneIntermediates",
        "DiffusionConfig",
        "TrackHitsConfig",
        "DigitizationConfig",
        "ResponseKernel",
        # Helpers
        "get_volume_deposits",
        # Factories
        "create_sim_params",
        "create_sim_config",
        "create_deposit_data",
        "pad_deposit_data",
        "create_track_hits_config",
        "create_digitization_config",
    ],
)

_optional_export("tools.geometry", ["generate_detector"])
_optional_export("tools.loader", ["load_particle_step_data", "build_deposit_data", "load_event"])
_optional_export("tools.simulation", ["DetectorSimulator"])
_optional_export("tools.recombination", ["RECOMB_MODELS", "compute_quanta", "XI_FN"])
_optional_export(
    "tools.particle_generator",
    [
        "generate_muon_segments",
        "generate_muon_segments_trig",
        "load_dedx_table_jax",
        "mask_outside_volume",
    ],
)
_optional_export("tools.wires", ["sparse_buckets_to_dense"])
_optional_export("tools.noise", ["add_noise", "generate_noise", "generate_noise_bucketed"])
