"""JWT secret resolution: the committed placeholder must never sign a token."""

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[1] / "root" / "voice_app")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP_DIR = tempfile.mkdtemp(prefix="voice-config-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "config.db")

import app as appmod  # noqa: E402
import database  # noqa: E402


class WebRtcBuildTests(unittest.TestCase):
    def test_actual_docker_patch_preserves_the_module(self):
        dockerfile=(ROOT/'Dockerfile').read_text()
        step=dockerfile.split('# Patch webrtcvad.py:',1)[1].split('RUN ',1)[1].split('\n#',1)[0].strip()
        command=shlex.split(step.replace('\\\n',''))
        self.assertEqual(command[:2],['python','-c'])
        with tempfile.TemporaryDirectory(prefix='vad-build-patch-') as directory:
            path=Path(directory)/'webrtcvad.py'
            path.write_text("import pkg_resources\n__version__ = pkg_resources.get_distribution('webrtcvad').version\nclass Vad:\n    def __init__(self, mode): self.mode = mode\n    def is_speech(self, frame, rate): return False\n")
            subprocess.run([sys.executable,'-c',command[2]],env=dict(os.environ,PYTHONPATH=directory),check=True,capture_output=True)
            patched=path.read_text()
            self.assertIn('class Vad:',patched)
            self.assertNotIn('pkg_resources',patched)
            subprocess.run([sys.executable,'-c',"import webrtcvad; assert webrtcvad.Vad(2).is_speech(bytes(960),16000) is False"],
                env=dict(os.environ,PYTHONPATH=directory),check=True,capture_output=True)


class JwtSecretTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("JWT_SECRET_KEY")
        database.init_db()

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("JWT_SECRET_KEY", None)
        else:
            os.environ["JWT_SECRET_KEY"] = self._saved

    def test_env_secret_is_used(self):
        os.environ["JWT_SECRET_KEY"] = "a-real-deployment-secret"
        self.assertEqual(appmod.resolve_jwt_secret(), ("a-real-deployment-secret", "env"))

    def test_committed_placeholder_is_refused(self):
        os.environ["JWT_SECRET_KEY"] = appmod._LEGACY_PLACEHOLDER_SECRET
        secret, source = appmod.resolve_jwt_secret()
        self.assertNotEqual(secret, appmod._LEGACY_PLACEHOLDER_SECRET)
        self.assertEqual(source, "database")

    def test_generated_secret_is_stable_across_calls(self):
        os.environ.pop("JWT_SECRET_KEY", None)
        first, source = appmod.resolve_jwt_secret()
        second, _ = appmod.resolve_jwt_secret()
        self.assertEqual(source, "database")
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 32)

    def test_blank_env_secret_falls_back(self):
        os.environ["JWT_SECRET_KEY"] = "   "
        secret, source = appmod.resolve_jwt_secret()
        self.assertEqual(source, "database")
        self.assertTrue(secret.strip())


if __name__ == "__main__":
    unittest.main()
