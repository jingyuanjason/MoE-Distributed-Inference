"""Configuration and launch tests that require neither vLLM nor a GPU."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from vllm_server import VLLMServer


class VLLMServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / "server config.yaml"
        self.config.write_text(
            "model: test/model\nport: 8000\nenable-expert-parallel: true\n",
            encoding="utf-8",
        )

    def test_reads_parameters(self):
        server = VLLMServer(self.config)
        self.assertEqual(server.config_path, self.config.resolve())
        self.assertEqual(server.parameters["model"], "test/model")
        self.assertEqual(server.parameters["port"], 8000)
        self.assertIs(server.parameters["enable-expert-parallel"], True)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            VLLMServer(self.config.parent / "missing.yaml")

    def test_invalid_configuration(self):
        for content in ("", "[]", "model: null", "model: ' '", "model: 42", "1: x"):
            with self.subTest(content=content):
                self.config.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    VLLMServer(self.config)

    def test_unsafe_yaml_is_rejected(self):
        self.config.write_text("!!python/object:builtins.object {}", encoding="utf-8")
        with self.assertRaises(yaml.YAMLError):
            VLLMServer(self.config)

    @patch("vllm_server._find_bundled_cuda_home", return_value=None)
    @patch("vllm_server.subprocess.run")
    @patch("vllm_server.shutil.which", return_value="/test/bin/vllm")
    def test_launches_with_original_config(self, which, run, cuda_home):
        VLLMServer(self.config).serve()
        which.assert_called_once_with("vllm")
        run.assert_called_once_with(
            ["/test/bin/vllm", "serve", "--config", str(self.config.resolve())],
            check=True,
            env=None,
        )

    @patch("vllm_server._find_bundled_cuda_home", return_value=Path("/bundled/cu13"))
    @patch("vllm_server.subprocess.run")
    @patch("vllm_server.shutil.which", return_value="/test/bin/vllm")
    def test_bundled_cuda_home_is_used(self, which, run, cuda_home):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_HOME", None)
            os.environ.pop("CUDA_PATH", None)
            VLLMServer(self.config).serve()
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_HOME"], "/bundled/cu13")
        self.assertEqual(env["PATH"], os.environ["PATH"])

    @patch("vllm_server._find_bundled_cuda_home", return_value=Path("/bundled/cu13"))
    @patch("vllm_server.subprocess.run")
    @patch("vllm_server.shutil.which", return_value="/test/bin/vllm")
    def test_user_cuda_home_is_respected(self, which, run, cuda_home):
        with patch.dict(os.environ, {"CUDA_HOME": "/custom/cuda"}):
            VLLMServer(self.config).serve()
        self.assertIsNone(run.call_args.kwargs["env"])

    @patch("vllm_server.subprocess.run")
    @patch("vllm_server.shutil.which", return_value=None)
    def test_missing_vllm(self, which, run):
        with self.assertRaisesRegex(RuntimeError, "not available on PATH"):
            VLLMServer(self.config).serve()
        run.assert_not_called()

    @patch("vllm_server.shutil.which", return_value="/test/bin/vllm")
    @patch("vllm_server.subprocess.run", side_effect=subprocess.CalledProcessError(1, "vllm"))
    def test_server_failure_is_propagated(self, run, which):
        with self.assertRaises(subprocess.CalledProcessError):
            VLLMServer(self.config).serve()

    def test_example_configuration(self):
        server = VLLMServer(Path(__file__).with_name("config.example.yaml"))
        self.assertEqual(server.parameters["port"], 8000)


if __name__ == "__main__":
    unittest.main()