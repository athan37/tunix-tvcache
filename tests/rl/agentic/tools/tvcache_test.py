# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Tunix TVCache adapter (no live server required)."""

from __future__ import annotations

import abc
import asyncio
import json
import os
import platform
import sys
import tempfile
import threading
import time
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from tunix.rl.agentic.tools import base_tool
from tunix.rl.agentic.tools import calculator_tool
from tunix.rl.agentic.tools import tvcache as tvcache_lib


class _MutatingTool(base_tool.BaseTool):
  """Uses the default will_mutate_state (True)."""

  def get_json_schema(self) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

  def apply(self, **kwargs):
    del kwargs
    return base_tool.ToolOutput(name=self.name, output="unused")


class _FailingTool(base_tool.BaseTool):

  def get_json_schema(self) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

  def apply(self, **kwargs):
    return base_tool.ToolOutput(name=self.name, error="boom")


class _SlowTool(base_tool.BaseTool):
  """Blocks for ``delay`` seconds; used to prove thread-pool offload."""

  def __init__(self, name, description, delay=0.2):
    super().__init__(name=name, description=description)
    self.delay = delay
    self.started = threading.Event()
    self.entered = 0
    self._lock = threading.Lock()

  def get_json_schema(self) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

  def will_mutate_state(self, **kwargs) -> bool:
    del kwargs
    return False

  def apply(self, **kwargs):
    del kwargs
    with self._lock:
      self.entered += 1
      if self.entered >= 2:
        self.started.set()
    # Wait briefly so both concurrent calls can enter before finishing.
    time.sleep(self.delay)
    return base_tool.ToolOutput(name=self.name, output="done")


class FakeToolCall(abc.ABC):
  """Minimal stand-in for tvclient.ToolCall used in unit tests."""

  @abc.abstractmethod
  def to_dict(self) -> dict:
    raise NotImplementedError

  def will_mutate_state(self) -> bool:
    return True

  def serialize(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True)


class TvcacheAdapterTest(absltest.TestCase):

  def tearDown(self):
    tvcache_lib.clear_in_memory_snapshots()
    tvcache_lib._CachedToolCallCls = None  # pylint: disable=protected-access
    super().tearDown()

  def test_task_id_from_key(self):
    task_id = tvcache_lib.task_id_for(
        {"instance_id": "task-7", "prompt": "x"}, "instance_id"
    )
    self.assertEqual(task_id, "task-7")

  def test_task_id_hash_is_stable(self):
    task = {"b": 2, "a": 1}
    first = tvcache_lib.task_id_for(task, None)
    second = tvcache_lib.task_id_for({"a": 1, "b": 2}, None)
    self.assertEqual(first, second)
    self.assertEqual(len(first), 64)

  def test_task_id_missing_key_raises(self):
    with self.assertRaisesRegex(KeyError, "instance_id"):
      tvcache_lib.task_id_for({"prompt": "x"}, "instance_id")

  def test_in_memory_fork_isolation(self):
    async def _run():
      live = tvcache_lib.InMemoryToolSandbox(task_name="t1")
      await live.start()
      live.state["x"] = 1
      snap = await live.fork()
      live.state["x"] = 2
      restored = tvcache_lib.InMemoryToolSandbox(
          task_name="t1", env_id=snap.get_id()
      )
      await restored.start()
      self.assertEqual(restored.state["x"], 1)
      self.assertEqual(live.state["x"], 2)
      handle_id = snap.get_id()
      await snap.stop()
      with self.assertRaises(KeyError):
        dead = tvcache_lib.InMemoryToolSandbox(task_name="t1", env_id=handle_id)
        await dead.start()

    asyncio.run(_run())

  def test_in_memory_execute_failed_prefix(self):
    async def _run():
      sandbox = tvcache_lib.InMemoryToolSandbox(task_name="t1")
      await sandbox.start()
      tool = _FailingTool(name="fail", description="fails")
      call = types.SimpleNamespace(payload=tool, arguments={})
      result = await sandbox.execute(call)
      self.assertTrue(result.startswith("Failed to execute:"))
      self.assertIn("boom", result)

    asyncio.run(_run())

  def test_cached_tool_call_serialization_and_mutate(self):
    fake_tvclient = types.SimpleNamespace(ToolCall=FakeToolCall)
    with mock.patch.object(
        tvcache_lib, "_require_tvclient", return_value=fake_tvclient
    ):
      calc = calculator_tool.CalculatorTool(
          name="calculator", description="calc"
      )
      call_a = tvcache_lib.make_tool_call(
          calc,
          base_tool.ToolCall(
              name="calculator", arguments={"b": 2, "a": 1, "op": "+"}
          ),
      )
      call_b = tvcache_lib.make_tool_call(
          calc,
          base_tool.ToolCall(
              name="calculator", arguments={"a": 1, "op": "+", "b": 2}
          ),
      )
      self.assertEqual(call_a.serialize(), call_b.serialize())
      self.assertFalse(call_a.will_mutate_state())
      self.assertIs(call_a.payload, calc)

      mutating = _MutatingTool(name="set", description="set")
      call_m = tvcache_lib.make_tool_call(
          mutating,
          base_tool.ToolCall(name="set", arguments={"key": "k", "value": 3}),
      )
      self.assertTrue(call_m.will_mutate_state())


def _tvclient_available() -> bool:
  try:
    import tvclient  # pylint: disable=g-import-not-at-top,unused-import

    return True
  except ImportError:
    return False


@absltest.skipUnless(
    os.environ.get("TVCACHE_TEST_ADDR") and _tvclient_available(),
    "Requires TVCACHE_TEST_ADDR and google-tunix[tvcache]",
)
class TvcacheIntegrationTest(absltest.TestCase):
  """Live-server integration covering hit, fast-forward, test, and concurrency."""

  def tearDown(self):
    tvcache_lib.clear_in_memory_snapshots()
    super().tearDown()

  def _config(self) -> tvcache_lib.TvcacheConfig:
    addr = os.environ["TVCACHE_TEST_ADDR"]
    host, port_s = addr.rsplit(":", 1)
    return tvcache_lib.TvcacheConfig(
        host=host,
        port=int(port_s),
        sandbox_class="tunix.rl.agentic.tools.tvcache.InMemoryToolSandbox",
        snapshot=False,
    )

  def test_second_rollout_hits_cache(self):
    tool = calculator_tool.CalculatorTool(name="calculator", description="c")
    task_id = f"integration-hit-{os.getpid()}"
    config = self._config()

    def _one_rollout() -> str:
      bridge = tvcache_lib.TvcacheBridge(config, task_id)
      try:
        return bridge.call(
            tvcache_lib.make_tool_call(
                tool,
                base_tool.ToolCall(
                    name="calculator", arguments={"a": 2, "b": 3, "op": "+"}
                ),
            )
        )
      finally:
        bridge.close()

    first = _one_rollout()
    second = _one_rollout()
    self.assertEqual(first, second)
    self.assertIn("5", first)

  def test_fast_forward_after_hit_and_test_grades_state(self):
    """After a hit leaves the live sandbox behind, test() still grades right."""

    class _CounterTool(base_tool.BaseTool):

      def get_json_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

      def will_mutate_state(self, **kwargs) -> bool:
        del kwargs
        return True

      def apply_in_sandbox(self, sandbox, **kwargs):
        del kwargs
        sandbox.state["count"] = int(sandbox.state.get("count", 0)) + 1
        sandbox.state["_reward"] = float(sandbox.state["count"])
        return base_tool.ToolOutput(
            name=self.name, output=str(sandbox.state["count"])
        )

      def apply(self, **kwargs):
        return base_tool.ToolOutput(name=self.name, output="unused")

    tool = _CounterTool(name="inc", description="increment")
    task_id = f"integration-ff-{os.getpid()}"
    config = self._config()

    # First rollout populates the cache with one mutating call.
    bridge1 = tvcache_lib.TvcacheBridge(config, task_id)
    try:
      out1 = bridge1.call(
          tvcache_lib.make_tool_call(tool, base_tool.ToolCall(name="inc", arguments={}))
      )
      grade1 = bridge1.test()
    finally:
      bridge1.close()
    self.assertIn("1", out1)
    self.assertEqual(grade1, "1.0")

    # Second rollout hits the cache for the same call (live sandbox falls
    # behind), then grades via test() which fast-forwards before scoring.
    bridge2 = tvcache_lib.TvcacheBridge(config, task_id)
    try:
      out2 = bridge2.call(
          tvcache_lib.make_tool_call(tool, base_tool.ToolCall(name="inc", arguments={}))
      )
      grade2 = bridge2.test()
    finally:
      bridge2.close()
    self.assertEqual(out1, out2)
    self.assertEqual(grade1, grade2)

  def test_concurrent_slow_calls_run_in_parallel(self):
    """Two slow tool calls on the shared loop finish near max(delay), not 2x."""
    tvcache_lib.configure_thread_pool(8)
    tool = _SlowTool(name="slow", description="slow", delay=0.25)
    task_id_a = f"integration-slow-a-{os.getpid()}"
    task_id_b = f"integration-slow-b-{os.getpid()}"
    config = self._config()

    results: list[str] = []
    errors: list[BaseException] = []

    def _rollout(task_id: str) -> None:
      bridge = tvcache_lib.TvcacheBridge(config, task_id)
      try:
        results.append(
            bridge.call(
                tvcache_lib.make_tool_call(
                    tool, base_tool.ToolCall(name="slow", arguments={})
                )
            )
        )
      except BaseException as e:  # pylint: disable=broad-exception-caught
        errors.append(e)
      finally:
        bridge.close()

    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=_rollout, args=(task_id_a,)),
        threading.Thread(target=_rollout, args=(task_id_b,)),
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    elapsed = time.perf_counter() - t0

    self.assertEqual(errors, [])
    self.assertEqual(len(results), 2)
    # If both ran on the asyncio loop without offload, elapsed ~= 0.5s.
    # With the thread pool, elapsed should be close to a single delay.
    self.assertLess(elapsed, 0.45)
    self.assertGreaterEqual(tool.entered, 2)


class ServerLookupTest(absltest.TestCase):
  """Binary lookup and local server lifecycle. No real server is started."""

  def setUp(self):
    super().setUp()
    tvcache_lib._server_ready = False  # pylint: disable=protected-access
    tvcache_lib._server_process = None  # pylint: disable=protected-access
    tvcache_lib._server_log = None  # pylint: disable=protected-access

  def tearDown(self):
    tvcache_lib._server_ready = False  # pylint: disable=protected-access
    tvcache_lib._server_process = None  # pylint: disable=protected-access
    tvcache_lib._server_log = None  # pylint: disable=protected-access
    super().tearDown()

  def _executable(self, directory: str, name: str = "tvcache") -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
      handle.write("#!/bin/sh\n")
    os.chmod(path, 0o755)
    return path

  def test_explicit_path(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._executable(tmp)
      self.assertEqual(tvcache_lib.find_server_binary(path), path)

  def test_env_before_path(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._executable(tmp, "from-env")
      with mock.patch.dict(os.environ, {"TVCACHE_BINARY": path}):
        with mock.patch.object(
            tvcache_lib.shutil, "which", return_value="/other/tvcache"
        ):
          self.assertEqual(tvcache_lib.find_server_binary(), path)

  def test_prefix_before_path(self):
    with tempfile.TemporaryDirectory() as tmp:
      os.makedirs(os.path.join(tmp, "bin"))
      path = self._executable(os.path.join(tmp, "bin"))
      with mock.patch.dict(os.environ, {"TVCACHE_BINARY": ""}):
        with mock.patch.object(sys, "prefix", tmp):
          with mock.patch.object(
              tvcache_lib.shutil, "which", return_value="/other/tvcache"
          ):
            self.assertEqual(tvcache_lib.find_server_binary(), path)

  def test_path_lookup(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._executable(tmp)
      with mock.patch.dict(os.environ, {"TVCACHE_BINARY": ""}):
        with mock.patch.object(sys, "prefix", tmp):
          with mock.patch.object(tvcache_lib.shutil, "which", return_value=path):
            self.assertEqual(tvcache_lib.find_server_binary(), path)

  def test_missing_names_install_step(self):
    with mock.patch.dict(os.environ, {"TVCACHE_BINARY": ""}):
      with mock.patch.object(sys, "prefix", "/no/such/prefix"):
        with mock.patch.object(tvcache_lib.shutil, "which", return_value=None):
          with mock.patch.object(
              tvcache_lib, "bundled_server_binary", return_value=None
          ):
            with self.assertRaisesRegex(FileNotFoundError, "was not found"):
              tvcache_lib.find_server_binary()

  def test_bundled_binary_matches_this_platform_when_present(self):
    bundled = tvcache_lib.bundled_server_binary()
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
      suffix = "tvcache-darwin-arm64"
    elif sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
      suffix = "tvcache-linux-x86_64"
    else:
      self.assertIsNone(bundled)
      return
    if bundled is None:
      return
    self.assertTrue(str(bundled).endswith(suffix))

  def test_bundled_binary_when_nothing_else_is_installed(self):
    with mock.patch.dict(os.environ, {"TVCACHE_BINARY": ""}):
      with mock.patch.object(sys, "prefix", "/no/such/prefix"):
        with mock.patch.object(tvcache_lib.shutil, "which", return_value=None):
          with mock.patch.object(
              tvcache_lib, "bundled_server_binary", return_value="/shipped/tvcache"
          ):
            self.assertEqual(
                tvcache_lib.find_server_binary(), "/shipped/tvcache"
            )

  def test_existing_binary_is_used_before_the_bundled_one(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._executable(tmp, "from-env")
      with mock.patch.dict(os.environ, {"TVCACHE_BINARY": path}):
        with mock.patch.object(
            tvcache_lib, "bundled_server_binary", return_value="/shipped/tvcache"
        ) as bundled:
          self.assertEqual(tvcache_lib.find_server_binary(), path)
    bundled.assert_not_called()

  def test_configured_path_must_be_executable(self):
    with self.assertRaisesRegex(FileNotFoundError, "not executable"):
      tvcache_lib.find_server_binary("/no/such/tvcache")

  def test_resolve_builds_when_the_server_is_missing(self):
    with mock.patch.dict(os.environ, {"TVCACHE_BINARY": ""}):
      with mock.patch.object(
          tvcache_lib,
          "find_server_binary",
          side_effect=FileNotFoundError("tvcache server was not found"),
      ):
        with mock.patch.object(
            tvcache_lib, "install_server_binary", return_value="/cache/tvcache"
        ) as install:
          found = tvcache_lib.resolve_server_binary()
    self.assertEqual(found, "/cache/tvcache")
    install.assert_called_once()

  def test_resolve_does_not_replace_a_configured_path(self):
    with mock.patch.dict(os.environ, {"TVCACHE_BINARY": "/no/such/tvcache"}):
      with mock.patch.object(tvcache_lib, "install_server_binary") as install:
        with self.assertRaisesRegex(FileNotFoundError, "not executable"):
          tvcache_lib.resolve_server_binary()
    install.assert_not_called()

  def test_install_requires_cargo(self):
    with mock.patch.object(tvcache_lib, "_is_executable", return_value=False):
      with mock.patch.object(tvcache_lib.shutil, "which", return_value=None):
        with self.assertRaisesRegex(FileNotFoundError, "rustup"):
          tvcache_lib.install_server_binary()

  def test_accepts_tool_cache_from_a_named_argument(self):
    class _ToolEnv:
      def __init__(self, tvcache=None):
        del tvcache

    class _ForwardingEnv(_ToolEnv):
      def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class _PlainEnv:
      def __init__(self, prompt="", **kwargs):
        del prompt, kwargs

    self.assertTrue(tvcache_lib.accepts_tool_cache(_ToolEnv))
    self.assertTrue(tvcache_lib.accepts_tool_cache(_ForwardingEnv))
    self.assertFalse(tvcache_lib.accepts_tool_cache(_PlainEnv))

  def test_gsm8k_does_not_accept_tool_cache(self):
    from tunix.dist_rl.examples.math_gsm8k_dist import gsm8k  # pylint: disable=g-import-not-at-top

    self.assertFalse(tvcache_lib.accepts_tool_cache(gsm8k.GSM8KEnv))

  def test_default_config(self):
    config = tvcache_lib.default_config()
    self.assertEqual(config.host, "127.0.0.1")
    self.assertEqual(config.port, 7878)
    self.assertFalse(config.snapshot)

  def test_rejects_port_other_than_7878(self):
    with self.assertRaisesRegex(ValueError, "7878"):
      tvcache_lib.ensure_local_server(port=7879)

  def test_reuses_listening_port(self):
    with mock.patch.object(tvcache_lib, "_port_open", return_value=True):
      with mock.patch.object(tvcache_lib.subprocess, "Popen") as popen:
        tvcache_lib.ensure_local_server()
        tvcache_lib.ensure_local_server()
    popen.assert_not_called()

  def test_starts_once_and_stops_at_cleanup(self):
    proc = mock.Mock()
    proc.poll.return_value = None
    proc.pid = 99
    opened = {"n": 0}

    def _port_open(port, host="127.0.0.1"):
      del port, host
      opened["n"] += 1
      return opened["n"] > 1

    with tempfile.TemporaryDirectory() as tmp:
      path = self._executable(tmp)
      log_path = os.path.join(tmp, "tvcache.log")
      with mock.patch.object(tvcache_lib, "_port_open", side_effect=_port_open):
        with mock.patch.object(
            tvcache_lib, "find_server_binary", return_value=path
        ):
          with mock.patch.object(
              tvcache_lib.subprocess, "Popen", return_value=proc
          ) as popen:
            tvcache_lib.ensure_local_server(log_path=log_path)
            tvcache_lib.ensure_local_server(log_path=log_path)
      popen.assert_called_once()
      argv = popen.call_args.args[0]
      self.assertEqual(argv[0], path)
      self.assertIn("--eviction-policy=ttl_and_watermark", argv)
      self.assertIn("--high-watermark-mb=1024", argv)
      self.assertIn("--low-watermark-mb=768", argv)
    tvcache_lib._stop_local_server()  # pylint: disable=protected-access
    proc.terminate.assert_called_once()


if __name__ == "__main__":
  absltest.main()
