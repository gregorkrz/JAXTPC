import importlib.util
import inspect
import pathlib
import sys
import types
import unittest

import numpy as np


def _install_import_stubs() -> None:
    """Stub heavy simulation modules; tests only need optimization helpers."""
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None

    jax = types.ModuleType("jax")
    jax.config = types.SimpleNamespace(update=lambda *args, **kwargs: None)
    def _tree_map(fn, tree):
        if isinstance(tree, dict):
            return {k: _tree_map(fn, v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [_tree_map(fn, v) for v in tree]
        if isinstance(tree, tuple):
            return tuple(_tree_map(fn, v) for v in tree)
        return fn(tree)
    jax.tree_util = types.SimpleNamespace(tree_map=_tree_map)
    def _jit(fn=None, *args, **kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    jax.jit = _jit
    jax.block_until_ready = lambda value: value
    jax.devices = lambda: ["cpu-stub"]

    jax_numpy = types.ModuleType("jax.numpy")
    jax_numpy.array = lambda value, dtype=None: np.array(value, dtype=dtype)
    jax_numpy.asarray = lambda value, dtype=None: np.asarray(value, dtype=dtype)
    jax_numpy.sum = lambda value: np.sum(value)
    jax_numpy.mean = lambda value: np.mean(value)
    jax_numpy.zeros = lambda shape=(), dtype=float: np.zeros(shape, dtype=dtype)
    jax_numpy.zeros_like = lambda value: np.zeros_like(value)
    jax_numpy.float32 = np.float32
    jax_numpy.ndarray = np.ndarray

    class _SimpleOptState(dict):
        pass

    class _SimpleOptimizer:
        def __init__(self, lr, momentum=0.0):
            self.lr = lr
            self.momentum = momentum

        def _resolve_lr(self, step):
            return self.lr(step) if callable(self.lr) else self.lr

        def init(self, params):
            return _SimpleOptState(step=0, velocity=np.zeros_like(params, dtype=float))

        def update(self, grads, state):
            step = state["step"]
            lr_now = self._resolve_lr(step)
            if self.momentum:
                velocity = self.momentum * state["velocity"] - lr_now * np.array(grads)
                updates = velocity
            else:
                velocity = state["velocity"]
                updates = -lr_now * np.array(grads)
            state["step"] = step + 1
            state["velocity"] = velocity
            return updates, state

    def _cosine_decay_schedule(init_value, decay_steps):
        def _schedule(step):
            ratio = min(max(step / max(decay_steps, 1), 0.0), 1.0)
            return init_value * 0.5 * (1.0 + np.cos(np.pi * ratio))
        return _schedule

    optax = types.ModuleType("optax")
    optax.sgd = lambda lr, momentum=0.0: _SimpleOptimizer(lr, momentum=momentum)
    optax.adam = lambda lr: _SimpleOptimizer(lr, momentum=0.0)
    optax.cosine_decay_schedule = _cosine_decay_schedule
    optax.constant_schedule = lambda value: (lambda _step: value)
    optax.chain = lambda *transforms: transforms[-1]
    optax.clip_by_global_norm = lambda _norm: _SimpleOptimizer(0.0)
    optax.apply_updates = lambda params, updates: np.array(params) + np.array(updates)

    geometry = types.ModuleType("tools.geometry")
    geometry.generate_detector = lambda *args, **kwargs: None

    loader = types.ModuleType("tools.loader")
    loader.build_deposit_data = lambda *args, **kwargs: None
    loader.load_particle_step_data = lambda *args, **kwargs: None
    loader.load_event = lambda *args, **kwargs: None

    losses = types.ModuleType("tools.losses")
    losses.make_sobolev_weight = lambda *args, **kwargs: None
    losses.sobolev_loss = lambda *args, **kwargs: None
    losses.sobolev_loss_geomean_log1p = lambda *args, **kwargs: None
    losses.mse_loss = lambda *args, **kwargs: None
    losses.l1_loss = lambda *args, **kwargs: None

    particle_generator = types.ModuleType("tools.particle_generator")
    particle_generator.generate_muon_track = lambda *args, **kwargs: None
    particle_generator.generate_muon_segments = lambda *args, **kwargs: None
    particle_generator.generate_muon_segments_trig = lambda *args, **kwargs: None
    particle_generator.load_dedx_table_jax = lambda *args, **kwargs: None
    particle_generator.mask_outside_volume = lambda *args, **kwargs: None

    noise = types.ModuleType("tools.noise")
    noise.generate_noise = lambda *args, **kwargs: None
    noise.add_noise = lambda *args, **kwargs: None
    noise.generate_noise_bucketed = lambda *args, **kwargs: None

    simulation = types.ModuleType("tools.simulation")
    simulation.DetectorSimulator = object

    recombination = types.ModuleType("tools.recombination")
    recombination.RECOMB_MODELS = {}
    recombination.compute_quanta = lambda *args, **kwargs: None
    recombination.XI_FN = {}

    wires = types.ModuleType("tools.wires")
    wires.sparse_buckets_to_dense = lambda *args, **kwargs: None

    sys.modules["dotenv"] = dotenv
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jax_numpy
    sys.modules["optax"] = optax
    sys.modules["tools.geometry"] = geometry
    sys.modules["tools.loader"] = loader
    sys.modules["tools.losses"] = losses
    sys.modules["tools.noise"] = noise
    sys.modules["tools.particle_generator"] = particle_generator
    sys.modules["tools.simulation"] = simulation
    sys.modules["tools.recombination"] = recombination
    sys.modules["tools.wires"] = wires


def _resolve_opt_module_path() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "src" / "opt" / "run_optimization.py",
        repo_root / "run_optimization.py",
        repo_root / "src" / "opt" / "2d_opt.py",
        repo_root / "2d_opt.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate optimization module. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_opt_module():
    _install_import_stubs()
    module_path = _resolve_opt_module_path()
    spec = importlib.util.spec_from_file_location("opt_testable", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


OPT_MODULE = _load_opt_module()


def _parse_params_for_test(valid_input: str):
    if hasattr(OPT_MODULE, "parse_pairs"):
        return OPT_MODULE.parse_pairs(valid_input)
    if hasattr(OPT_MODULE, "parse_params"):
        return OPT_MODULE.parse_params(valid_input)
    raise AttributeError("No parse function found in optimization module")


class TestOptimizationHelpers(unittest.TestCase):
    @staticmethod
    def _optimizer_and_schedule(lr, max_steps):
        built = OPT_MODULE.make_optax_optimizer("sgd", lr, "constant", max_steps)
        if isinstance(built, tuple) and len(built) == 2:
            return built
        return built, None

    @staticmethod
    def _single_phase_schedule(max_steps, val_and_grad_fn):
        return [(max_steps, lambda _p: [lambda p: val_and_grad_fn(p)])]

    def test_parse_params_valid(self):
        parsed = _parse_params_for_test(
            "velocity_cm_us+lifetime_us,recomb_alpha+recomb_beta_90"
            if hasattr(OPT_MODULE, "parse_pairs")
            else "velocity_cm_us,recomb_alpha,recomb_beta_90"
        )
        if hasattr(OPT_MODULE, "parse_pairs"):
            self.assertEqual(
                parsed,
                [
                    ("velocity_cm_us", "lifetime_us"),
                    ("recomb_alpha", "recomb_beta_90"),
                ],
            )
        else:
            self.assertEqual(
                parsed,
                ["velocity_cm_us", "recomb_alpha", "recomb_beta_90"],
            )

    def test_parse_params_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            _parse_params_for_test(
                "velocity_cm_us+velocity_cm_us"
                if hasattr(OPT_MODULE, "parse_pairs")
                else "velocity_cm_us,velocity_cm_us"
            )
        with self.assertRaises(ValueError):
            _parse_params_for_test(
                "unknown+lifetime_us"
                if hasattr(OPT_MODULE, "parse_pairs")
                else "unknown,lifetime_us"
            )
        with self.assertRaises(ValueError):
            _parse_params_for_test(
                "velocity_cm_us,lifetime_us"
                if hasattr(OPT_MODULE, "parse_pairs")
                else ""
            )

    def test_make_optax_optimizer_rejects_unknown_optimizer(self):
        with self.assertRaises(ValueError):
            OPT_MODULE.make_optax_optimizer("bad_optimizer", 0.01, "constant", 10)

    def test_run_trial_reduces_loss(self):
        def val_and_grad_fn(p):
            loss = np.sum((p - 1.0) ** 2)
            grad = 2.0 * (p - 1.0)
            return loss, grad

        optimizer, schedule_fn = self._optimizer_and_schedule(0.2, 30)
        run_trial_params = inspect.signature(OPT_MODULE.run_trial).parameters
        if "phase_schedule" in run_trial_params:
            trial = OPT_MODULE.run_trial(
                p0=[2.0, -1.0],
                phase_schedule=self._single_phase_schedule(10, val_and_grad_fn),
                optimizer=optimizer,
                max_steps=10,
                tol=1e-12,
                patience=20,
                schedule_fn=schedule_fn,
            )
        elif "p0_pn_vec" in run_trial_params:
            trial = OPT_MODULE.run_trial(
                p0_pn_vec=[2.0, -1.0],
                val_and_grad_fn=val_and_grad_fn,
                optimizer=optimizer,
                max_steps=10,
                tol=1e-12,
                patience=20,
            )
        else:
            trial = OPT_MODULE.run_trial(
                p0=[2.0, -1.0],
                val_and_grad_fn=val_and_grad_fn,
                optimizer=optimizer,
                max_steps=10,
                tol=1e-12,
                patience=20,
            )

        self.assertEqual(len(trial["param_trajectory"]), 11)
        self.assertEqual(len(trial["loss_trajectory"]), 11)
        self.assertLess(trial["loss_trajectory"][-1], trial["loss_trajectory"][0])
        self.assertFalse(trial["stopped_early"])

    def test_run_trial_early_stops_on_flat_loss(self):
        def val_and_grad_fn(p):
            return 0.0, np.zeros_like(p)

        optimizer, schedule_fn = self._optimizer_and_schedule(0.1, 30)
        run_trial_params = inspect.signature(OPT_MODULE.run_trial).parameters
        if "phase_schedule" in run_trial_params:
            trial = OPT_MODULE.run_trial(
                p0=[1.0, 1.0],
                phase_schedule=self._single_phase_schedule(20, val_and_grad_fn),
                optimizer=optimizer,
                max_steps=20,
                tol=1e-8,
                patience=1,
                schedule_fn=schedule_fn,
            )
        elif "p0_pn_vec" in run_trial_params:
            trial = OPT_MODULE.run_trial(
                p0_pn_vec=[1.0, 1.0],
                val_and_grad_fn=val_and_grad_fn,
                optimizer=optimizer,
                max_steps=20,
                tol=1e-8,
                patience=1,
            )
        else:
            trial = OPT_MODULE.run_trial(
                p0=[1.0, 1.0],
                val_and_grad_fn=val_and_grad_fn,
                optimizer=optimizer,
                max_steps=20,
                tol=1e-8,
                patience=1,
            )

        self.assertTrue(trial["stopped_early"])
        self.assertLess(trial["steps_run"], 20)


if __name__ == "__main__":
    unittest.main()
