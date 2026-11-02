"""Offline regression tests for shell and URL trust boundaries."""

import ast
import contextlib
import shlex
import sys
import time
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib import parse

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args + function.args.kwonlyargs:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class SecurityBoundariesTests(unittest.TestCase):
    def test_git_directory_is_one_argument_even_with_shell_metacharacters(self):
        runner = Mock(return_value=b"v1-0-gabc\n")
        fn = load_function("ultralytics/utils/checks.py", "git_describe", {
            "ROOT": Path("."), "contextlib": contextlib,
            "subprocess": SimpleNamespace(check_output=runner),
        })
        self.assertEqual(fn("repo;touch /tmp/should-not-exist"), "v1-0-gabc")
        runner.assert_called_once_with([
            "git", "-C", "repo;touch /tmp/should-not-exist", "describe", "--tags", "--long", "--always"
        ])

    def test_google_drive_detection_checks_actual_host(self):
        fake_drive = Mock(side_effect=AssertionError("must not treat attacker host as Google Drive"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.write_text("cached")
            fn = load_function("ultralytics/utils/downloads.py", "safe_download", {
                "parse": parse, "Path": Path, "get_google_drive_file_info": fake_drive,
            })
            fn("https://drive.google.com.attacker.example/file", file=output)
            fake_drive.assert_not_called()

    def test_requirement_name_is_not_interpreted_by_shell(self):
        runner = Mock(return_value=b"ok")
        pkg = SimpleNamespace(
            DistributionNotFound=type("MissingPackage", (Exception,), {}),
            VersionConflict=type("VersionConflict", (Exception,), {}),
        )
        pkg.require = Mock(side_effect=pkg.DistributionNotFound())
        pkg.parse_requirements = Mock(side_effect=ImportError())
        fn = load_function("ultralytics/utils/checks.py", "check_requirements", {
            "ROOT": Path("."), "Path": Path, "pkg": pkg, "shlex": shlex, "sys": sys,
            "AUTOINSTALL": True, "LOGGER": Mock(), "subprocess": SimpleNamespace(check_output=runner),
            "is_online": lambda: True, "check_python": lambda: None, "check_torchvision": lambda: None,
            "colorstr": lambda *args: "", "emojis": lambda s: s,
            "time": time,
        })
        import importlib
        with patch.object(importlib, "import_module", side_effect=ImportError()):
            self.assertTrue(fn(["missing; touch /tmp/not-run"], cmds="--extra-index-url https://example.com"))
        args = runner.call_args.args[0]
        self.assertEqual(args[5], "missing; touch /tmp/not-run")
        self.assertEqual(runner.call_args.kwargs, {})

    def test_edgetpu_paths_are_passed_as_arguments(self):
        run = Mock(return_value=SimpleNamespace(returncode=0, stdout=b"compiler 1.0"))
        fn = load_function("ultralytics/engine/exporter.py", "export_edgetpu", {
            "colorstr": lambda x: x, "LOGGER": Mock(), "LINUX": True, "Path": Path,
            "subprocess": SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL),
        })
        model = '/tmp/model";touch /tmp/not-run.tflite'
        obj = SimpleNamespace(_add_tflite_metadata=Mock())
        fn(obj, model)
        self.assertEqual(run.call_args.args[0][-1], model)
        self.assertEqual(run.call_args.kwargs, {"check": True})


if __name__ == "__main__":
    unittest.main()
