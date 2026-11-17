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

"""Unit tests for the DeepSWE / R2E-Gym TVCache adapter."""

from __future__ import annotations

import types
from unittest import mock

from absl.testing import absltest
from examples.deepswe import r2egym_tvcache


class R2EGymTvcacheTest(absltest.TestCase):

  def tearDown(self):
    r2egym_tvcache.clear_snapshot_images_for_tests()
    super().tearDown()

  def test_action_mutates_classification(self):
    self.assertFalse(r2egym_tvcache.r2egym_action_mutates("search", {"query": "x"}))
    self.assertFalse(
        r2egym_tvcache.r2egym_action_mutates(
            "file_editor", {"command": "view", "path": "/x"}
        )
    )
    self.assertTrue(
        r2egym_tvcache.r2egym_action_mutates(
            "file_editor", {"command": "str_replace", "path": "/x"}
        )
    )
    self.assertTrue(
        r2egym_tvcache.r2egym_action_mutates(
            "execute_bash", {"command": "ls"}
        )
    )
    self.assertTrue(r2egym_tvcache.r2egym_action_mutates("finish", {}))
    self.assertTrue(r2egym_tvcache.r2egym_action_mutates("submit", {}))

  def test_bind_sandbox_cls_closes_over_owner(self):
    owner_a = types.SimpleNamespace(use_agent_sandbox=False, entry={}, name="a")
    owner_b = types.SimpleNamespace(use_agent_sandbox=False, entry={}, name="b")
    cls_a = r2egym_tvcache.bind_sandbox_cls(owner_a)
    cls_b = r2egym_tvcache.bind_sandbox_cls(owner_b)
    sand_a = cls_a(task_name="same-instance")
    sand_b = cls_b(task_name="same-instance")
    self.assertIs(sand_a._owner, owner_a)  # pylint: disable=protected-access
    self.assertIs(sand_b._owner, owner_b)  # pylint: disable=protected-access
    self.assertIsNot(type(sand_a), type(sand_b))

  def test_docker_fork_commits_container(self):
    owner = types.SimpleNamespace(
        use_agent_sandbox=False,
        entry={"instance_id": "t1", "docker_image": "img"},
        env=None,
        handle=None,
        fleet=None,
    )
    sandbox_cls = r2egym_tvcache.bind_sandbox_cls(owner)
    sandbox = sandbox_cls(task_name="t1")
    sandbox.env = types.SimpleNamespace(
        runtime=types.SimpleNamespace(container_name="ctr-abc")
    )
    sandbox.env_id = "live-1"

    with mock.patch.object(
        r2egym_tvcache, "_docker", return_value=mock.Mock()
    ) as docker_mock:
      forked = sandbox._fork_sync()  # pylint: disable=protected-access

    docker_mock.assert_called_once()
    args = docker_mock.call_args.args[0]
    self.assertEqual(args[0], "commit")
    self.assertEqual(args[1], "ctr-abc")
    self.assertTrue(forked.env_id.startswith("snap-"))
    self.assertTrue(forked._is_snapshot_handle)  # pylint: disable=protected-access
    self.assertIs(forked._owner, owner)  # pylint: disable=protected-access

  def test_fork_raises_on_agent_sandbox(self):
    owner = types.SimpleNamespace(use_agent_sandbox=True, entry={}, env=None)
    sandbox = r2egym_tvcache.bind_sandbox_cls(owner)(task_name="t2")
    sandbox.env = object()
    with self.assertRaisesRegex(NotImplementedError, "agent-sandbox"):
      sandbox._fork_sync()  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
