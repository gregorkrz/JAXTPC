import importlib.util
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
    jax.jit = lambda fn: fn
    jax.block_until_ready = lambda value: value
    jax.devices = lambda: ["cpu-stub"]

    jax_numpy = types.ModuleType("jax.numpy")
    jax_numpy.array = lambda value, dtype=None: np.array(value, dtype=dtype)
    jax_numpy.sum = lambda value: np.sum(value)
    jax_numpy.mean = lambda value: np.mean(value)
    jax_numpy.zeros = lambda shape=(), dtype=float: np.zeros(shape, dtype=dtype)
    jax_numpy.zeros_like = lambda value: np.zeros_like(value)
    jax_numpy.float32 = np.float32

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
    optax.apply_updates = lambda params, updates: np.array(params) + np.array(updates)

    geometry = types.ModuleType("tools.geometry")
    geometry.generate_detector = lambda *args, **kwargs: None

    loader = types.ModuleType("tools.loader")
    loader.build_deposit_data = lambda *args, **kwargs: None

    losses = types.ModuleType("tools.losses")
    losses.make_sobolev_weight = lambda *args, **kwargs: None
    losses.sobolev_loss = lambda *args, **kwargs: None
    losses.sobolev_loss_geomean_log1p = lambda *args, **kwargs: None

    particle_generator = types.ModuleType("tools.particle_generator")
    particle_generator.generate_muon_track = lambda *args, **kwargs: None

    simulation = types.ModuleType("tools.simulation")
    simulation.DetectorSimulator = object

    sys.modules["dotenv"] = dotenv
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jax_numpy
    sys.modules["optax"] = optax
    sys.modules["tools.geometry"] = geometry
    sys.modules["tools.loader"] = loader
    sys.modules["tools.losses"] = losses
    sys.modules["tools.particle_generator"] = particle_generator
    sys.modules["tools.simulation"] = simulation


def _load_2d_opt_module():
    _install_import_stubs()
    module_path = pathlib.Path(__file__).resolve().parents[1] / "2d_opt.py"
    spec = importlib.util.spec_from_file_location("two_d_opt_testable", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


TWO_D_OPT = _load_2d_opt_module()


class TestOptimizationHelpers(unittest.TestCase):
    def test_parse_pairs_valid(self):
        pairs = TWO_D_OPT.parse_pairs(
            "velocity_cm_us+lifetime_us,recomb_alpha+recomb_beta_90"
        )
        self.assertEqual(
            pairs,
            [
                ("velocity_cm_us", "lifetime_us"),
                ("recomb_alpha", "recomb_beta_90"),
            ],
        )

    def test_parse_pairs_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            TWO_D_OPT.parse_pairs("velocity_cm_us+velocity_cm_us")
        with self.assertRaises(ValueError):
            TWO_D_OPT.parse_pairs("unknown+lifetime_us")
        with self.assertRaises(ValueError):
            TWO_D_OPT.parse_pairs("velocity_cm_us,lifetime_us")

    def test_make_optax_optimizer_rejects_unknown_optimizer(self):
        with self.assertRaises(ValueError):
            TWO_D_OPT.make_optax_optimizer("bad_optimizer", 0.01, "constant", 10)

    def test_run_trial_reduces_loss(self):
        def val_and_grad_fn(p):
            loss = np.sum((p - 1.0) ** 2)
            grad = 2.0 * (p - 1.0)
            return loss, grad

        optimizer = TWO_D_OPT.make_optax_optimizer("sgd", 0.2, "constant", 30)
        trial = TWO_D_OPT.run_trial(
            p0_pn_vec=[2.0, -1.0],
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

        optimizer = TWO_D_OPT.make_optax_optimizer("sgd", 0.1, "constant", 30)
        trial = TWO_D_OPT.run_trial(
            p0_pn_vec=[1.0, 1.0],
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
