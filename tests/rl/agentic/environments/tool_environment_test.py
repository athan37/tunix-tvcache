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

"""Unit tests for ToolEnvironment, including the optional TVCache path."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

from absl.testing import absltest
from tunix.rl.agentic.environments import tool_environment
from tunix.rl.agentic.tools import calculator_tool
from tunix.rl.agentic.tools import tvcache as tvcache_lib


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict:
  return {
      "id": call_id,
      "function": {
          "name": name,
          "arguments": json.dumps(arguments),
      },
  }


class ToolEnvironmentTest(absltest.TestCase):

  def test_disabled_path_unchanged(self):
    env = tool_environment.ToolEnvironment(
        task={"prompt": "add"},
        tool_map={"calculator": calculator_tool.CalculatorTool},
        reward_fn=lambda task, action: 1.0,
        max_steps=5,
    )
    obs = env.reset()
    self.assertEqual(obs[0]["prompt"], "add")
    result = env.step([
        _tool_call("calculator", {"a": 1, "b": 2, "op": "+"}, "c1"),
    ])
    observation, reward, done, _ = result
    self.assertFalse(done)
    self.assertEqual(reward, 0.0)
    self.assertEqual(observation["tool_outputs"]["c1"], "3")
    env.close()

  def test_from_dict_pops_tvcache(self):
    env = tool_environment.ToolEnvironment.from_dict({
        "tool_map": {"calculator": calculator_tool.CalculatorTool},
        "max_steps": 3,
        "tvcache": {"host": "10.0.0.1", "port": 9000},
        "prompt": "q",
    })
    config = env._tvcache_config  # pylint: disable=protected-access
    self.assertEqual(config.host, "10.0.0.1")
    self.assertEqual(config.port, 9000)
    self.assertEqual(env.task, {"prompt": "q"})
    env.close()

  def test_cached_path_sends_cumulative_history(self):
    histories: list[list[Any]] = []

    def _call(tool_call):
      # Mimic bridge.call: append then return based on length.
      if not histories:
        histories.append([])
      histories[0].append(tool_call)
      return f"out-{len(histories[0])}"

    fake_bridge = mock.Mock()
    fake_bridge.call.side_effect = _call
    fake_bridge.close = mock.Mock()

    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1", port=7878, task_id_key="instance_id"
    )
    with mock.patch.object(
        tvcache_lib, "TvcacheBridge", return_value=fake_bridge
    ) as bridge_cls:
      with mock.patch.object(
          tvcache_lib,
          "make_tool_call",
          side_effect=_fake_tool_call,
      ):
        env = tool_environment.ToolEnvironment(
            task={"prompt": "p", "instance_id": "same"},
            tool_map={"calculator": calculator_tool.CalculatorTool},
            reward_fn=lambda task, action: 0.0,
            max_steps=5,
            tvcache=config,
        )
        env.reset()
        bridge_cls.assert_called_once()
        self.assertEqual(
            bridge_cls.call_args.args[1],
            tvcache_lib.cache_task_id(
                "",
                {"prompt": "p", "instance_id": "same"},
                task_id_key="instance_id",
            ),
        )
        # No rollout_id kwarg.
        self.assertEqual(bridge_cls.call_args.kwargs, {})

        env.step([
            _tool_call("calculator", {"a": 1, "b": 1, "op": "+"}, "c1"),
            _tool_call("calculator", {"a": 2, "b": 2, "op": "+"}, "c2"),
        ])
        self.assertEqual(fake_bridge.call.call_count, 2)
        first_call = fake_bridge.call.call_args_list[0].args[0]
        second_call = fake_bridge.call.call_args_list[1].args[0]
        self.assertIsNot(first_call, second_call)

        env.close()
        fake_bridge.close.assert_called()

  def test_same_task_same_task_id(self):
    fake_bridge = mock.Mock()
    fake_bridge.call.return_value = "x"
    fake_bridge.close = mock.Mock()
    config = tvcache_lib.TvcacheConfig(host="127.0.0.1", port=7878)
    task_ids = []

    def _capture(config, task_id):
      del config
      task_ids.append(task_id)
      return fake_bridge

    with mock.patch.object(tvcache_lib, "TvcacheBridge", side_effect=_capture):
      for _ in range(2):
        env = tool_environment.ToolEnvironment(
            task={"b": 2, "a": 1},
            tool_map={"calculator": calculator_tool.CalculatorTool},
            tvcache=config,
        )
        env.reset()
        env.close()
    self.assertEqual(task_ids[0], task_ids[1])


  def test_insert_failure_uses_the_shared_bridge_fallback(self):
    try:
      import tvclient  # pylint: disable=g-import-not-at-top,unused-import
    except ImportError:
      self.skipTest("tvclient is not installed")
    from tests.rl.agentic.tools import tvcache_recovery_test as recovery
    from tunix.rl.agentic.tools import base_tool

    class _Edit(base_tool.BaseTool):

      def get_json_schema(self):
        return {"type": "function", "function": {"name": self.name}}

      def apply(self, **kwargs):
        del kwargs
        return base_tool.ToolOutput(name=self.name, output="unused")

    recovery.CountingSandbox.reset()
    tvcache_lib.reset_connect_backoff_for_test()
    edit = tvcache_lib.make_cached_tool_call("edit", {}, mutates=True)
    client = recovery.ScriptedClient()
    client.find_results = ["edited", None]
    client.fail_insert_paths.add((edit.serialize(),))
    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1", port=9, snapshot=False, connect_timeout_s=0.2
    )
    env = tool_environment.ToolEnvironment(
        task={"prompt": "p"},
        tool_map={
            "edit": _Edit,
            "calculator": calculator_tool.CalculatorTool,
        },
        tvcache=config,
    )
    env.reset()
    bridge = tvcache_lib.TvcacheBridge(
        config, "task", sandbox_cls=recovery.CountingSandbox
    )
    bridge._adapter.bind_client(client)  # pylint: disable=protected-access
    env._tvcache_bridge.close()  # pylint: disable=protected-access
    env._tvcache_bridge = bridge  # pylint: disable=protected-access
    calls = [edit]
    tools = {tool.name: tool for tool in env.tool_manager.get_tools()}
    real_make = tvcache_lib.make_tool_call

    def _make(tool, call):
      del tool
      if calls and call.name == "edit":
        return calls.pop(0)
      return real_make(tools[call.name], call)

    with mock.patch.object(tvcache_lib, "make_tool_call", side_effect=_make):
      obs, _reward, _done, _info = env.step([
          _tool_call("edit", {}, "e1"),
          _tool_call("calculator", {"a": 1, "b": 1, "op": "+"}, "c1"),
      ])
    outputs = obs["tool_outputs"]
    self.assertEqual(outputs["e1"], "edited")
    self.assertIn("c1", outputs)
    sandbox = bridge._bypass_sandbox  # pylint: disable=protected-access
    executed = [getattr(call, "name", "") for call in sandbox.executed]
    self.assertEqual(executed.count("edit"), 1)
    self.assertEqual(executed.count("calculator"), 1)
    env.close()
    self.assertEqual(sandbox.stops, 1)


def _fake_tool_call(tool, call):
  """Builds a lightweight stand-in for CachedToolCall used by the mock."""
  return mock.Mock(
      name=call.name,
      arguments=dict(call.arguments),
      payload=tool,
      will_mutate_state=mock.Mock(return_value=False),
  )


if __name__ == "__main__":
  absltest.main()
