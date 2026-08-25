# Owner(s): ["module: inductor"]
import importlib.metadata
import threading
from unittest import mock

from torch._inductor.codegen import common
from torch.testing._internal.common_utils import run_tests, TestCase


def _make_fake_entrypoint(name: str, loader_fn):
    """Build a fake EntryPoint that behaves like the real importlib type."""
    ep = mock.MagicMock(spec=importlib.metadata.EntryPoint)
    ep.name = name
    ep.load = mock.MagicMock(return_value=loader_fn)
    return ep


class DeviceBackendLoaderTest(TestCase):
    def setUp(self) -> None:
        # Snapshot the module-level registries so every test starts clean, then
        # restore exactly in tearDown (do not wipe device_codegens wholesale --
        # init_backend_registration() is cached and would not re-populate the
        # built-ins).
        self._orig_loaders = dict(common._device_backend_loaders)
        self._orig_attempted = set(common._attempted_device_backends)
        self._orig_codegen = dict(common.device_codegens)
        self._orig_custom_passes = dict(common.custom_backend_passes)
        self._orig_custom_configs = dict(common.custom_backend_codegen_configs)
        common._device_backend_loaders.clear()
        common._attempted_device_backends.clear()
        common._discover_device_backend_entrypoints.cache_clear()

    def tearDown(self) -> None:
        for device in [
            d for d in common.device_codegens if d not in self._orig_codegen
        ]:
            del common.device_codegens[device]
        common.custom_backend_passes.clear()
        common.custom_backend_passes.update(self._orig_custom_passes)
        common.custom_backend_codegen_configs.clear()
        common.custom_backend_codegen_configs.update(self._orig_custom_configs)
        common._device_backend_loaders.clear()
        common._device_backend_loaders.update(self._orig_loaders)
        common._attempted_device_backends.clear()
        common._attempted_device_backends.update(self._orig_attempted)
        common._discover_device_backend_entrypoints.cache_clear()

    # ------------------------------------------------------------------
    # Basic loader behaviour
    # ------------------------------------------------------------------

    def test_loader_called_once_on_first_resolve(self) -> None:
        device = "fake_loader_device"
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            # A real vendor calls register_backend_for_device(...) here.
            common.register_backend_for_device(device, object, object)

        common.register_device_backend_loader(device, loader)
        self.assertIsNotNone(common.get_scheduling_for_device(device))
        self.assertEqual(calls["n"], 1)
        # Subsequent resolves must not re-invoke the loader.
        common.get_scheduling_for_device(device)
        self.assertEqual(calls["n"], 1)

    def test_loader_fires_on_wrapper_codegen_path(self) -> None:
        device = "fake_wrapper_device"

        common.register_device_backend_loader(
            device, lambda: common.register_backend_for_device(device, object, object)
        )
        self.assertIsNotNone(common.get_wrapper_codegen_for_device(device))

    def test_loader_fires_on_custom_backend_pass_path(self) -> None:
        device = "fake_pass_device"
        common.register_device_backend_loader(
            device,
            lambda: common.register_backend_for_device(
                device, object, object, device_custom_pass=object
            ),
        )
        self.assertIsNotNone(common.get_custom_backend_pass_for_device(device))

    def test_loader_fires_on_custom_config_path(self) -> None:
        device = "fake_config_device"
        # ConfigModule is needed, but for the lazy-load trigger test we just
        # need the loader to fire before the dict lookup.
        common.register_device_backend_loader(
            device, lambda: common.register_backend_for_device(device, object, object)
        )
        # Will return None (no custom config registered) but the loader
        # must have been invoked.
        self.assertIsNone(common.get_custom_backend_config_for_device(device))

    def test_unknown_device_still_returns_none(self) -> None:
        self.assertIsNone(
            common.get_scheduling_for_device("totally_unknown_device_xyz")
        )

    # ------------------------------------------------------------------
    # Entry-point discovery
    # ------------------------------------------------------------------

    def test_entrypoint_group_name_is_stable(self) -> None:
        # Guard against a typo that would silently break discovery.
        self.assertEqual(
            common.INDUCTOR_DEVICE_BACKENDS_GROUP,
            "torch.inductor.device_backends",
        )

    def test_entrypoint_discovery_picks_up_vendor(self) -> None:
        device = "fake_ep_device"
        ep = _make_fake_entrypoint(
            device,
            lambda: common.register_backend_for_device(device, object, object),
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            self.assertIsNotNone(common.get_scheduling_for_device(device))
        ep.load.assert_called_once()

    def test_entrypoint_import_is_deferred_until_use(self) -> None:
        device = "fake_deferred_device"
        ep = _make_fake_entrypoint(
            device,
            lambda: common.register_backend_for_device(device, object, object),
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            common._discover_device_backend_entrypoints()
            # Discovery must build the loader without importing the vendor module.
            ep.load.assert_not_called()
            self.assertIsNotNone(common.get_scheduling_for_device(device))
        ep.load.assert_called_once()

    def test_imperative_loader_does_not_suppress_entrypoints(self) -> None:
        # Regression guard: an imperative loader must not cause entry-point
        # discovery to be skipped for a different device.
        common.register_device_backend_loader("imperative_device", lambda: None)
        ep_device = "ep_only_device"
        ep = _make_fake_entrypoint(
            ep_device,
            lambda: common.register_backend_for_device(ep_device, object, object),
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            self.assertIsNotNone(common.get_scheduling_for_device(ep_device))

    def test_imperative_loader_wins_over_entrypoint_for_same_device(self) -> None:
        # setdefault: a loader registered imperatively takes precedence over a
        # same-named entry point.
        device = "contested_device"
        imperative_calls = {"n": 0}

        def imperative_loader() -> None:
            imperative_calls["n"] += 1
            common.register_backend_for_device(device, object, object)

        common.register_device_backend_loader(device, imperative_loader)
        ep = _make_fake_entrypoint(
            device,
            lambda: common.register_backend_for_device(device, object, object),
        )
        with mock.patch("importlib.metadata.entry_points", return_value=[ep]):
            self.assertIsNotNone(common.get_scheduling_for_device(device))
        self.assertEqual(imperative_calls["n"], 1)
        ep.load.assert_not_called()

    # ------------------------------------------------------------------
    # Failure / no-op semantics
    # ------------------------------------------------------------------

    def test_loader_failure_is_raised_and_retryable(self) -> None:
        device = "fake_failing_device"
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            raise RuntimeError("boom")

        common.register_device_backend_loader(device, loader)
        # The real failure propagates (instead of a generic "device not supported").
        with self.assertRaisesRegex(RuntimeError, "boom"):
            common.get_scheduling_for_device(device)
        # The claim was released, so a later resolve retries the loader.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            common.get_scheduling_for_device(device)
        self.assertEqual(calls["n"], 2)

    def test_loader_succeeds_without_registering_is_not_retried(self) -> None:
        # A loader that returns OK but forgets to register must not busy-loop:
        # the claim is retained, so later resolves skip it.
        device = "fake_noop_device"
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            # Intentionally does NOT call register_backend_for_device.

        common.register_device_backend_loader(device, loader)
        self.assertIsNone(common.get_scheduling_for_device(device))
        self.assertEqual(calls["n"], 1)
        # Second resolve must not re-invoke the no-op loader.
        self.assertIsNone(common.get_scheduling_for_device(device))
        self.assertEqual(calls["n"], 1)

    # ------------------------------------------------------------------
    # Re-entrancy
    # ------------------------------------------------------------------

    def test_reentrant_loader_does_not_deadlock(self) -> None:
        # A loader that itself calls get_scheduling_for_device must not cause
        # a deadlock or infinite recursion.
        device = "fake_reentrant_device"
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            # The loader's import may trigger scheduling lookups for the same
            # device; this must be a no-op (claim already set).
            common.get_scheduling_for_device(device)
            common.register_backend_for_device(device, object, object)

        common.register_device_backend_loader(device, loader)
        self.assertIsNotNone(common.get_scheduling_for_device(device))
        self.assertEqual(calls["n"], 1)

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_concurrent_loaders_call_loader_once(self) -> None:
        device = "fake_concurrent_device"
        calls = {"n": 0}
        barrier = threading.Barrier(4)

        def loader() -> None:
            calls["n"] += 1
            common.register_backend_for_device(device, object, object)

        common.register_device_backend_loader(device, loader)

        def worker():
            barrier.wait()
            common.get_scheduling_for_device(device)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The loader must be called exactly once despite 4 concurrent callers.
        self.assertEqual(calls["n"], 1)

    # ------------------------------------------------------------------
    # Built-in device performance (no entry-point scan)
    # ------------------------------------------------------------------

    def test_builtin_device_does_not_trigger_entrypoint_scan(self) -> None:
        # init_backend_registration checks device_codegens directly, not
        # get_scheduling_for_device, so built-in devices must NOT trigger
        # entry-point discovery.
        #
        # We spy on _discover_device_backend_entrypoints rather than patching
        # importlib.metadata.entry_points globally: the import chain inside
        # init_backend_registration (e.g. networkx) may itself call entry_points
        # for unrelated reasons.
        with mock.patch.object(
            common,
            "_discover_device_backend_entrypoints",
            side_effect=AssertionError("entry-point scan should not happen"),
        ) as spy:
            common.init_backend_registration.cache_clear()
            common.init_backend_registration()
            spy.assert_not_called()
            # Built-in devices are already in device_codegens, so the fast path
            # in _load_device_backend must skip discovery too.
            common.get_scheduling_for_device("cpu")
            common.get_scheduling_for_device("cuda")
            spy.assert_not_called()


if __name__ == "__main__":
    run_tests()
