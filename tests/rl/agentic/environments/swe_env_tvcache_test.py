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

"""Unit tests for SWEEnv TVCache wiring (no Docker / tvclient required)."""

from __future__ import annotations

import json
from unittest import mock

from absl.testing import absltest
from examples.deepswe import swe_env
from tunix.rl.agentic.tools import tvcache as tvcache_lib


class SWEEnvTvcacheTest(absltest.TestCase):

  def test_accepts_tvcache_mapping_without_typeerror(self):
    env = swe_env.SWEEnv(
        entry={
            "instance_id": "i1",
            "problem_statement": "fix it",
            "docker_image": "img",
        },
        backend="docker",
        max_steps=3,
        tvcache={"host": "127.0.0.1", "port": 7878, "snapshot": False},
    )
    config = env._tvcache_config  # pylint: disable=protected-access
    self.assertIsInstance(config, tvcache_lib.TvcacheConfig)
    self.assertEqual(config.host, "127.0.0.1")
    self.assertFalse(config.snapshot)
    env.close()

  def test_initial_observation_skips_container_when_tvcache_on(self):
    fake_bridge = mock.Mock()
    fake_bridge.close = mock.Mock()
    fake_bridge.test.return_value = "1.0"
    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1",
        port=7878,
        sandbox_class="examples.deepswe.r2egym_tvcache.R2EGymSandbox",
        task_id_key="instance_id",
        snapshot=False,
    )
    with mock.patch.object(
        tvcache_lib, "TvcacheBridge", return_value=fake_bridge
    ) as bridge_cls:
      env = swe_env.SWEEnv(
          entry={
              "instance_id": "same",
              "problem_statement": "fix the bug",
              "docker_image": "img",
          },
          backend="docker",
          max_steps=2,
          tvcache=config,
      )
      obs = env._initial_observation()  # pylint: disable=protected-access
      self.assertEqual(obs, "fix the bug")
      self.assertIsNone(env.env)
      bridge_cls.assert_called_once()
      self.assertEqual(bridge_cls.call_args.args[1], "same")
      self.assertIn("sandbox_cls", bridge_cls.call_args.kwargs)
      # final_reward_fn routes through bridge.test
      self.assertEqual(env.final_reward_fn(), 1.0)
      fake_bridge.test.assert_called_once()
      env.close()
      fake_bridge.close.assert_called()

  def test_step_routes_through_bridge(self):
    fake_bridge = mock.Mock()
    fake_bridge.call.return_value = json.dumps(
        {"obs": "ok", "reward": 0, "done": False, "info": {}}
    )
    fake_bridge.close = mock.Mock()
    config = tvcache_lib.TvcacheConfig(
        host="127.0.0.1",
        port=7878,
        sandbox_class="examples.deepswe.r2egym_tvcache.R2EGymSandbox",
        task_id_key="instance_id",
    )
    action = mock.Mock()
    action.function_name = "search"
    action.parameters = {"search_term": "foo"}

    # Avoid importing r2egym / tvclient in this unit test.
    fake_action_cls = mock.Mock()
    fake_tool_call = mock.Mock()
    with mock.patch.object(tvcache_lib, "TvcacheBridge", return_value=fake_bridge):
      with mock.patch.object(swe_env, "Action", fake_action_cls):
        with mock.patch(
            "examples.deepswe.r2egym_tvcache.make_r2egym_tool_call",
            return_value=fake_tool_call,
        ) as make_call:
          env = swe_env.SWEEnv(
              entry={
                  "instance_id": "t",
                  "problem_statement": "p",
                  "docker_image": "img",
              },
              backend="docker",
              max_steps=2,
              tvcache=config,
          )
          env._initial_observation()  # pylint: disable=protected-access
          result = env._step_impl(action)  # pylint: disable=protected-access
          self.assertEqual(result.observation, "ok")
          self.assertFalse(result.done)
          make_call.assert_called_once_with(action)
          fake_bridge.call.assert_called_once_with(fake_tool_call)
          env.close()


if __name__ == "__main__":
  absltest.main()
