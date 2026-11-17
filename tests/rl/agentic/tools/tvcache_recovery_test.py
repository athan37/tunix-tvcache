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

"""Recovery and pinned-executor tests for TvcacheBridge.

The executor tests import tvclient and are skipped when it is not installed.
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Any

from absl.testing import absltest
from tunix.rl.agentic.tools import tvcache as tvcache_lib


def _tvclient():
  try:
    import tvclient  # pylint: disable=g-import-not-at-top
  except ImportError:
    raise absltest.SkipTest("tvclient is not installed") from None
  return tvclient


class _Call:
  """Stand-in tool call for catch-up tests that do not import tvclient."""

  def __init__(self, name: str, *, mutates: bool):
    self.name = name
    self._mutates = mutates

  def will_mutate_state(self) -> bool:
    return self._mutates

  def serialize(self) -> str:
    return json.dumps({"name": self.name}, sort_keys=True)

  def to_dict(self) -> dict[str, Any]:
    return {"name": self.name}


class CountingSandbox:
  """Records start, execute, and stop. One instance, one owner."""

  fail_replay = False

  def __init__(self, task_name: str, env_id: str | None = None):
    self.task_name = task_name
    self.env_id = env_id or f"live-{id(self)}"
    self.executed: list[Any] = []
    self.starts = 0
    self.stops = 0
    self.index = CountingSandbox._next
    CountingSandbox._next += 1
    CountingSandbox.instances.append(self)

  _next = 0
  instances: list[CountingSandbox] = []

  @classmethod
  def reset(cls) -> None:
    cls._next = 0
    cls.instances = []
    cls.fail_replay = False

  def get_id(self) -> str:
    return self.env_id

  async def start(self) -> None:
    self.starts += 1

  async def stop(self) -> None:
    self.stops += 1

  async def fork(self) -> CountingSandbox:
    raise AssertionError("snapshot=false must not fork")

  async def execute(self, tool_call: Any) -> str:
    if CountingSandbox.fail_replay:
      raise RuntimeError("replay failed")
    self.executed.append(tool_call)
    if tvcache_lib.is_test_tool_call(tool_call):
      return "1.0"
    name = getattr(tool_call, "name", "call")
    return json.dumps({"obs": name, "reward": 0, "done": False, "info": {}})


class ScriptedClient:
  """TVCache client double. ``find_results`` are values or exceptions."""

  def __init__(self) -> None:
    self.timeout = 5
    self.retries = 0
    self.find_results: list[Any] = []
    self.fail_insert_paths: set[tuple[str, ...]] = set()
    self.finds: list[list[str]] = []
    self.inserts: list[list[str]] = []
    self.connects = 0
    self.corrupt_history: Any = None
    self.fail_fresh_replay = False

  async def connect(self) -> None:
    self.connects += 1

  async def close(self) -> None:
    return None

  async def find(self, task_id: str, path: list[str]) -> Any:
    tvclient = _tvclient()
    self.finds.append(list(path))
    item = None if not self.find_results else self.find_results.pop(0)
    if isinstance(item, Exception):
      raise item
    return tvclient.FindResult(
        self,
        task_id,
        tvclient.FindResponse(value=item, sandbox_path=None, sandbox_id=None),
    )

  async def insert(
      self,
      task_id: str,
      path: list[str],
      value: str | None,
      sandbox_id: str | None,
      runtime: int | None = None,
      ttl: int | None = None,
  ) -> Any:
    del task_id, value, sandbox_id, runtime, ttl
    tvclient = _tvclient()
    self.inserts.append(list(path))
    if self.corrupt_history is not None:
      self.corrupt_history.sandbox_history[:] = ["not-a-prefix"]
      if self.fail_fresh_replay:
        CountingSandbox.fail_replay = True
    if tuple(path) in self.fail_insert_paths:
      raise OSError("insert failed")
    return tvclient.InsertResult(value_inserted=True, sandbox_inserted=False)

  async def get_fork_cost(self, task_id: str) -> int:
    del task_id
    return 0

  async def report_fork_cost(self, task_id: str, sample_ms: int) -> int:
    del task_id
    return sample_ms

  async def unref_sandbox(self, task_id: str, sandbox_id: str) -> None:
    del task_id, sandbox_id


def _names(sandbox: CountingSandbox) -> list[str]:
  names = []
  for call in sandbox.executed:
    if tvcache_lib.is_test_tool_call(call):
      names.append("test")
    else:
      names.append(getattr(call, "name", "call"))
  return names


class CatchUpTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    CountingSandbox.reset()

  def test_reuses_a_prefix_and_replays_the_gap(self):
    async def _run():
      first = _Call("a", mutates=True)
      second = _Call("b", mutates=True)
      sandbox = CountingSandbox("task")
      await sandbox.start()
      await sandbox.execute(first)
      return await tvcache_lib._WorkspaceRecovery.catch_up(  # pylint: disable=protected-access
          sandbox,
          [first.serialize()],
          [first, second],
          CountingSandbox,
          "task",
      )

    sandbox = asyncio.run(_run())
    self.assertEqual(_names(sandbox), ["a", "b"])
    self.assertEqual(sandbox.starts, 1)

  def test_invalid_prefix_stops_the_old_sandbox(self):
    async def _run():
      call = _Call("a", mutates=True)
      old = CountingSandbox("task")
      await old.start()
      fresh = await tvcache_lib._WorkspaceRecovery.catch_up(  # pylint: disable=protected-access
          old, ["nope"], [call], CountingSandbox, "task"
      )
      return old, fresh

    old, fresh = asyncio.run(_run())
    self.assertEqual(old.stops, 1)
    self.assertEqual(_names(fresh), ["a"])
    self.assertEqual(fresh.starts, 1)

  def test_replay_failure_stops_the_sandbox_it_started(self):
    CountingSandbox.fail_replay = True

    async def _run():
      call = _Call("a", mutates=True)
      with self.assertRaises(RuntimeError):
        await tvcache_lib._WorkspaceRecovery.catch_up(  # pylint: disable=protected-access
            None, [], [call], CountingSandbox, "task"
        )

    asyncio.run(_run())
    started = [sb for sb in CountingSandbox.instances if sb.starts]
    self.assertTrue(started)
    for sandbox in started:
      self.assertEqual(sandbox.stops, 1)


@absltest.skipUnless(
    __import__("importlib").util.find_spec("tvclient") is not None,
    "tvclient is not installed",
)
class ExecutorContractTest(absltest.TestCase):
  """Locks the pinned tvclient facts the adapter relies on."""

  def setUp(self):
    super().setUp()
    CountingSandbox.reset()
    tvcache_lib.reset_connect_backoff_for_test()

  def tearDown(self):
    tvcache_lib.reset_connect_backoff_for_test()
    super().tearDown()

  def test_failed_insert_keeps_history_and_the_same_call(self):
    tvclient = _tvclient()

    async def _run():
      executor = tvclient.TvcacheToolExecutor(
          CountingSandbox,
          tvcache_lib._get_cached_tool_call_cls(),  # pylint: disable=protected-access
          "task",
          host="127.0.0.1",
          port=9,
          adaptive=False,
          fixed_threshold_seconds=1e18,
      )
      client = ScriptedClient()
      executor.client = client
      executor.snapshot_policy.client = client
      edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
      client.find_results = [None]
      client.fail_insert_paths.add((edit.serialize(),))
      with self.assertRaises(OSError):
        await executor.lookup_or_execute([edit])
      return executor, edit

    executor, edit = asyncio.run(_run())
    self.assertIs(executor.rollout_sandbox.executed[0], edit)
    self.assertEqual(executor.sandbox_history, [edit.serialize()])
    self.assertIsNotNone(executor.rollout_sandbox)

  def test_adapter_translates_rpc_errors_but_not_sandbox_errors(self):
    _tvclient()
    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1", port=9, snapshot=False, connect_timeout_s=0.2
    )
    bridge = tvcache_lib.TvcacheBridge(
        config, "task", sandbox_cls=CountingSandbox
    )
    adapter = bridge._adapter  # pylint: disable=protected-access
    tracker = bridge._tracker  # pylint: disable=protected-access
    tracker.state = tvcache_lib._CallState.RAISED  # pylint: disable=protected-access
    with self.assertRaises(OSError):
      adapter._translate(OSError("sandbox"))  # pylint: disable=protected-access
    tracker.state = tvcache_lib._CallState.NOT_STARTED  # pylint: disable=protected-access
    with self.assertRaises(tvcache_lib.CacheUnavailable):
      adapter._translate(OSError("rpc"))  # pylint: disable=protected-access

  def _bridge(self, client: ScriptedClient, *, snapshot: bool = False):
    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1",
        port=9,
        snapshot=snapshot,
        connect_timeout_s=0.2,
        connect_retry_after_s=60,
    )
    bridge = tvcache_lib.TvcacheBridge(
        config, "task", sandbox_cls=CountingSandbox
    )
    bridge._adapter.bind_client(client)  # pylint: disable=protected-access
    return bridge

  def test_insert_failure_during_replay_runs_each_call_once(self):
    _tvclient()
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    read = tvcache_lib.make_cached_tool_call("read", {}, mutates=False)
    later = tvcache_lib.make_cached_tool_call("later", {}, mutates=False)
    client = ScriptedClient()
    client.find_results = [
        json.dumps({"obs": "edited", "reward": 0, "done": False, "info": {}}),
        None,
    ]
    client.fail_insert_paths.add((edit.serialize(),))
    bridge = self._bridge(client)
    first = json.loads(bridge.call(edit))
    second = json.loads(bridge.call(read))
    third = json.loads(bridge.call(later))
    grade = bridge.test()
    self.assertEqual(first["obs"], "edited")
    self.assertEqual(second["obs"], "read")
    self.assertEqual(third["obs"], "later")
    self.assertEqual(grade, "1.0")
    self.assertEqual(len(client.finds), 2)
    self.assertEqual(client.inserts, [[edit.serialize()]])
    sandbox = bridge._bypass_sandbox  # pylint: disable=protected-access
    self.assertEqual(_names(sandbox), ["edit", "read", "later", "test"])
    bridge.close()
    self.assertEqual(sandbox.stops, 1)
    self.assertEqual(sandbox.starts, 1)

  def test_insert_failure_for_the_current_call_returns_its_output(self):
    _tvclient()
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    client = ScriptedClient()
    client.find_results = [None]
    client.fail_insert_paths.add((edit.serialize(),))
    bridge = self._bridge(client)
    output = json.loads(bridge.call(edit))
    self.assertEqual(output["obs"], "edit")
    self.assertEqual(_names(bridge._bypass_sandbox), ["edit"])  # pylint: disable=protected-access
    finds = len(client.finds)
    bridge.call(tvcache_lib.make_cached_tool_call("next", {}, mutates=False))
    self.assertEqual(len(client.finds), finds)
    bridge.close()

  def test_lookup_failure_replays_then_runs_once(self):
    _tvclient()
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    read = tvcache_lib.make_cached_tool_call("read", {}, mutates=False)
    client = ScriptedClient()
    client.find_results = [
        json.dumps({"obs": "edited", "reward": 0, "done": False, "info": {}}),
        OSError("find failed"),
    ]
    bridge = self._bridge(client)
    bridge.call(edit)
    output = json.loads(bridge.call(read))
    self.assertEqual(output["obs"], "read")
    sandbox = bridge._bypass_sandbox  # pylint: disable=protected-access
    self.assertEqual(_names(sandbox), ["edit", "read"])
    bridge.close()
    self.assertEqual(sandbox.stops, sandbox.starts)

  def test_uncertain_completion_ends_the_call(self):
    _tvclient()

    class _Dying(CountingSandbox):

      async def execute(self, tool_call: Any) -> str:
        raise tvcache_lib.CacheUnavailable("timed out mid-command")

    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1", port=9, snapshot=False, connect_timeout_s=0.2
    )
    bridge = tvcache_lib.TvcacheBridge(config, "task", sandbox_cls=_Dying)
    client = ScriptedClient()
    client.find_results = [None]
    bridge._adapter.bind_client(client)  # pylint: disable=protected-access
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    with self.assertRaises(tvcache_lib.CacheUnavailable):
      bridge.call(edit)
    bridge.close()

  def test_recovery_replace_and_halfway_failure_stop_once(self):
    _tvclient()
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    read = tvcache_lib.make_cached_tool_call("read", {}, mutates=False)
    client = ScriptedClient()
    client.find_results = [
        json.dumps({"obs": "edited", "reward": 0, "done": False, "info": {}}),
        None,
    ]
    client.fail_insert_paths.add((edit.serialize(),))
    bridge = self._bridge(client)
    bridge.call(edit)
    client.corrupt_history = bridge._adapter.executor  # pylint: disable=protected-access
    bridge.call(read)
    bridge.close()
    for sandbox in CountingSandbox.instances:
      if sandbox.starts:
        self.assertEqual(sandbox.stops, 1)

    CountingSandbox.reset()
    client = ScriptedClient()
    client.find_results = [
        json.dumps({"obs": "edited", "reward": 0, "done": False, "info": {}}),
        None,
    ]
    client.fail_insert_paths.add((edit.serialize(),))
    client.fail_fresh_replay = True
    bridge = self._bridge(client)
    bridge.call(edit)
    client.corrupt_history = bridge._adapter.executor  # pylint: disable=protected-access
    with self.assertRaises(RuntimeError):
      bridge.call(read)
    bridge.close()
    for sandbox in CountingSandbox.instances:
      if sandbox.starts:
        self.assertEqual(sandbox.stops, 1)

  def test_materialize_then_close_stops_once(self):
    _tvclient()
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    client = ScriptedClient()
    client.find_results = [
        json.dumps({"obs": "edited", "reward": 0, "done": False, "info": {}}),
    ]
    bridge = self._bridge(client)
    bridge.call(edit)
    sandbox = bridge.materialize()
    self.assertEqual(_names(sandbox), ["edit"])
    bridge.close()
    self.assertEqual(sandbox.stops, 1)

  def test_snapshot_true_does_not_recover(self):
    _tvclient()
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    client = ScriptedClient()
    client.find_results = [OSError("down")]
    bridge = self._bridge(client, snapshot=True)
    with self.assertRaises(OSError):
      bridge.call(edit)
    self.assertFalse(bridge._bypass)  # pylint: disable=protected-access
    bridge.close()

  def test_connect_failure_is_fail_open_and_backs_off(self):
    _tvclient()
    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1",
        port=9,
        snapshot=False,
        connect_timeout_s=0.2,
        connect_retry_after_s=60,
    )

    async def _refuse() -> None:
      raise OSError("refused")

    connects: list[str] = []

    async def _count_refuse() -> None:
      connects.append("called")
      raise OSError("refused")

    with self.assertLogs(tvcache_lib.logger, level="WARNING") as logs:
      bridge = tvcache_lib.TvcacheBridge(
          config, "task", sandbox_cls=CountingSandbox
      )
      bridge._adapter.executor.client.connect = _refuse  # pylint: disable=protected-access
      output = json.loads(
          bridge.call(tvcache_lib.make_cached_tool_call("edit", {}, mutates=True))
      )
      bridge.close()
      self.assertEqual(output["obs"], "edit")
      second = tvcache_lib.TvcacheBridge(
          config, "task", sandbox_cls=CountingSandbox
      )
      second._adapter.executor.client.connect = _count_refuse  # pylint: disable=protected-access
      second.call(tvcache_lib.make_cached_tool_call("edit", {}, mutates=True))
      self.assertEqual(second.stats()["connect_skipped"], 1)
      second.close()
    self.assertEqual(
        sum("running uncached" in line for line in logs.output), 1
    )
    self.assertEqual(connects, [])

    tvcache_lib.reset_connect_backoff_for_test()
    # A previous outage is what arms the "reachable again" log. Reproduce it,
    # then let the next episode's connect succeed.
    with self.assertLogs(tvcache_lib.logger, level="WARNING") as logs:
      down = tvcache_lib.TvcacheBridge(
          config, "task", sandbox_cls=CountingSandbox
      )
      down._adapter.executor.client.connect = _refuse  # pylint: disable=protected-access
      down.call(tvcache_lib.make_cached_tool_call("edit", {}, mutates=True))
      down.close()
      # The backoff would skip the next connect. Drop only the timer.
      tvcache_lib._connect_unavailable_until = 0.0  # pylint: disable=protected-access

      async def _ok() -> None:
        return None

      up = tvcache_lib.TvcacheBridge(
          config, "task", sandbox_cls=CountingSandbox
      )
      up._adapter.executor.client.connect = _ok  # pylint: disable=protected-access
      client = ScriptedClient()
      client.find_results = [OSError("still down")]
      # connect is patched; bind_client would replace it. Install find only.
      up._adapter.executor.client.find = client.find  # pylint: disable=protected-access
      up._adapter.executor.snapshot_policy.client = client  # pylint: disable=protected-access
      up.call(tvcache_lib.make_cached_tool_call("edit", {}, mutates=True))
      up.close()
    self.assertTrue(any("reachable again" in line for line in logs.output))

  def test_recovery_failure_becomes_a_trajectory_error(self):
    """A raised recovery failure is the episode ERROR the rollout manager records."""
    _tvclient()

    class _Dying(CountingSandbox):

      async def execute(self, tool_call: Any) -> str:
        raise tvcache_lib.CacheUnavailable("timed out mid-command")

    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1", port=9, snapshot=False, connect_timeout_s=0.2
    )
    bridge = tvcache_lib.TvcacheBridge(config, "task", sandbox_cls=_Dying)
    client = ScriptedClient()
    client.find_results = [None]
    bridge._adapter.bind_client(client)  # pylint: disable=protected-access
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)

    class _Collector:
      traj_id = "traj-recovery"

      async def run_episode(self) -> None:
        bridge.call(edit)

    # Importing the manager pulls the rollout stack. The check is the existing
    # `_run_and_enqueue` path: any exception becomes a TrajectoryError.
    from tunix.dist_rl.common import datatypes
    from tunix.dist_rl.rollout import manager as manager_lib

    host = types.SimpleNamespace(
        _staging_buffer=None,
        _active_collectors={},
        _active_tasks={},
    )
    resolved: list[Any] = []
    asyncio.run(
        manager_lib.RolloutManager._run_and_enqueue(  # pylint: disable=protected-access
            host,
            _Collector(),
            datatypes.RolloutRequest(
                prompt="p",
                prompt_id="prompt-1",
                group_index=2,
                target_policy_version=7,
            ),
            None,
            resolved.append,
            slot_key="slot",
        )
    )
    bridge.close()
    self.assertEqual(len(resolved), 1)
    error = resolved[0]
    self.assertIsInstance(error, manager_lib.trajectory_lib.TrajectoryError)
    self.assertEqual(error.trajectory_id, "traj-recovery")
    self.assertEqual(error.prompt_id, "prompt-1")
    self.assertEqual(error.error_type, "CacheUnavailable")
    self.assertIn("current call was started", error.error_message)
    self.assertEqual(error.metadata["group_index"], 2)
    self.assertEqual(error.metadata["policy_version"], 7)
