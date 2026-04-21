import os
import tempfile
import unittest

from viewer import serve_viewer


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass


class TestServeViewerDiscovery(unittest.TestCase):
    def test_extract_dataset_name(self):
        self.assertEqual(
            serve_viewer.extract_dataset("myset_seg_0001.h5"),
            "myset",
        )
        self.assertEqual(
            serve_viewer.extract_dataset("myset_corr_0000.h5"),
            "myset",
        )
        self.assertEqual(
            serve_viewer.extract_dataset("myset_resp_0042.h5"),
            "myset",
        )
        self.assertEqual(
            serve_viewer.extract_dataset("myset_optical_0002.h5"),
            "myset",
        )
        self.assertIsNone(serve_viewer.extract_dataset("invalid_name.h5"))

    def test_find_h5_files_discovers_layouts_and_ignores_lzf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _touch(os.path.join(tmpdir, "seg", "sample_seg_0000.h5"))
            _touch(os.path.join(tmpdir, "sample_corr_0000.h5"))
            _touch(os.path.join(tmpdir, "resp", "sample_resp_0000.h5"))
            _touch(os.path.join(tmpdir, "resp", "sample_resp_lzf_0000.h5"))

            found = serve_viewer.find_h5_files(tmpdir)

            self.assertIn("seg", found)
            self.assertIn("corr", found)
            self.assertIn("resp", found)
            self.assertEqual(len(found["seg"]), 1)
            self.assertEqual(len(found["corr"]), 1)
            self.assertEqual(len(found["resp"]), 1)
            self.assertNotIn("sample_resp_lzf_0000.h5", found["resp"][0])

    def test_select_dataset_chooses_only_complete_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _touch(os.path.join(tmpdir, "complete_seg_0000.h5"))
            _touch(os.path.join(tmpdir, "complete_corr_0000.h5"))
            _touch(os.path.join(tmpdir, "complete_resp_0000.h5"))
            _touch(os.path.join(tmpdir, "incomplete_seg_0000.h5"))

            dataset, file_map = serve_viewer.select_dataset(tmpdir)

            self.assertEqual(dataset, "complete")
            self.assertTrue(all(kind in file_map for kind in ("seg", "corr", "resp")))

    def test_select_dataset_missing_required_kind_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _touch(os.path.join(tmpdir, "broken_seg_0000.h5"))
            _touch(os.path.join(tmpdir, "broken_resp_0000.h5"))

            with self.assertRaises(SystemExit) as ctx:
                serve_viewer.select_dataset(tmpdir, requested="broken")
            self.assertIn("missing corr files", str(ctx.exception))

    def test_build_manifest_returns_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = os.path.join(tmpdir, "seg", "sample_seg_0000.h5")
            corr_path = os.path.join(tmpdir, "corr", "sample_corr_0000.h5")
            resp_path = os.path.join(tmpdir, "resp", "sample_resp_0000.h5")
            for path in (seg_path, corr_path, resp_path):
                _touch(path)

            manifest = serve_viewer.build_manifest(
                tmpdir,
                {"seg": seg_path, "corr": corr_path, "resp": resp_path},
            )

            self.assertEqual(manifest["seg"], "seg/sample_seg_0000.h5")
            self.assertEqual(manifest["corr"], "corr/sample_corr_0000.h5")
            self.assertEqual(manifest["resp"], "resp/sample_resp_0000.h5")


if __name__ == "__main__":
    unittest.main()
