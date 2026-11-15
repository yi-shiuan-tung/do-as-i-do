"""Offline tests: no installs, downloads, GPU allocations, or changes to real repos."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SETUP = Path(__file__).resolve().parents[1]
ROOT = SETUP.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, SETUP / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SetupTests(unittest.TestCase):
    def test_patch_fresh_checkout_then_rerun(self):
        helper = load('apply_b200_patches')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'setup').mkdir()
            (root / 'setup/patches').symlink_to(SETUP / 'patches', target_is_directory=True)
            (root / 'env').symlink_to(ROOT / 'env', target_is_directory=True)
            for module, name in helper.PATCHES:
                repo = root / 'modules' / module
                repo.mkdir(parents=True, exist_ok=True)
                subprocess.run(['git', 'init', '-q', str(repo)], check=True)
                source = ROOT / 'modules' / module
                contents = (SETUP / 'patches' / f'{name}.patch').read_text()
                paths = [line[len('--- a/'):] for line in contents.splitlines() if line.startswith('--- a/')]
                for path in paths:
                    target = repo / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(subprocess.check_output(['git', '-C', str(source), 'show', f'HEAD:{path}']))
            with patch.object(helper, 'ROOT', root):
                helper.main()
                helper.main()
            hawor = root / 'modules/HaWoR/scripts/scripts_test_video/hawor_video.py'
            self.assertIn('torch.serialization.safe_globals', hawor.read_text())
            self.assertNotIn('weights_only=False', hawor.read_text())
            droid = root / 'modules/HaWoR/thirdparty/DROID-SLAM/setup.py'
            self.assertIn('compute_100', droid.read_text())
            self.assertNotIn('compute_120', droid.read_text())

    def test_download_failure_leaves_final_untouched(self):
        helper = load('fetch_b200_weights')
        with tempfile.TemporaryDirectory() as folder:
            final = Path(folder) / 'model.pt'
            with patch.object(helper.urllib.request, 'urlopen', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    helper.fetch('https://example.invalid/model', final)
            self.assertFalse(final.exists())

    def test_valid_weights_are_reused(self):
        helper = load('fetch_b200_weights')
        with tempfile.TemporaryDirectory() as folder:
            final = Path(folder) / 'model.pt'
            final.write_bytes(b'x' * 2048)
            with patch.object(helper.urllib.request, 'urlopen') as download:
                helper.fetch('https://example.invalid/model', final)
                download.assert_not_called()

    def test_html_and_lfs_are_rejected(self):
        helper = load('fetch_b200_weights')
        with tempfile.TemporaryDirectory() as folder:
            final = Path(folder) / 'model.pt'
            for prefix in [b'<html>', b'<!DOCTYPE html>', b'version https://git-lfs.github.com/spec/v1']:
                final.write_bytes(prefix + b'x' * 2048)
                self.assertFalse(helper.usable(final))

    def test_missing_mano_is_actionable(self):
        helper = load('fetch_b200_weights')
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(helper, 'HAWOR', Path(folder)), patch.dict(helper.os.environ, {'MANO_DIR': folder}):
                with self.assertRaisesRegex(RuntimeError, 'MANO_DIR'):
                    helper.install_mano()

    def test_real_dependency_conflict_is_not_suppressed(self):
        helper = load('check_packages')
        result = subprocess.CompletedProcess([], 1, 'jupyter 1.1.1 requires notebook, which is not installed.\n', '')
        with patch.object(helper.subprocess, 'run', return_value=result):
            with self.assertRaises(SystemExit) as error:
                helper.main()
        self.assertEqual(error.exception.code, 1)

    def test_cli_plan_and_invalid_stage(self):
        plan = subprocess.run(['bash', str(SETUP / 'install_b200.sh'), '--plan'], capture_output=True, text=True)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn('Torch 2.9 cu128', plan.stdout)
        invalid = subprocess.run(['bash', str(SETUP / 'install_b200.sh'), '--only', 'bad-stage'], capture_output=True)
        self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
