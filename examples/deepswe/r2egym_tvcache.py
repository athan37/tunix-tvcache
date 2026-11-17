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

"""TVCache sandbox adapter for R2E-Gym / DeepSWE.

``tvclient`` constructs sandboxes as ``(task_name, env_id)``. Because many
GRPO episodes share the same cache ``task_id`` (``instance_id``), each episode
binds a sandbox class that closes over its ``SWEEnv`` owner instead of using a
process-global registry keyed by task id.

Blocking Docker / RepoEnv work is offloaded through ``run_in_tvcache_executor``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import uuid
from collections.abc import Mapping
from typing import Any

from tunix.rl.agentic.tools import tvcache as tvcache_lib

logger = logging.getLogger(__name__)

# Snapshot image handles created by docker commit (env_id -> image tag).
_SNAPSHOT_IMAGES: dict[str, str] = {}
_SNAPSHOT_LOCK = threading.Lock()

_READ_ONLY_FUNCTIONS = frozenset({"search"})
_EDITOR_FUNCTIONS = frozenset({"file_editor", "str_replace_editor"})
_READ_ONLY_EDITOR_COMMANDS = frozenset({"view"})


def r2egym_action_mutates(function_name: str, parameters: Mapping[str, Any]) -> bool:
  """Returns whether an R2E-Gym / SWE-agent action mutates sandbox state."""
  name = (function_name or "").strip()
  if name in _READ_ONLY_FUNCTIONS:
    return False
  if name in _EDITOR_FUNCTIONS:
    command = str(parameters.get("command", "")).strip()
    return command not in _READ_ONLY_EDITOR_COMMANDS
  # execute_bash, finish, submit, and unknown tools are treated as mutating.
  return True


def make_r2egym_tool_call(action: Any) -> Any:
  """Wraps an R2E-Gym Action as a CachedToolCall for the bridge."""
  function_name = getattr(action, "function_name", "") or ""
  parameters = dict(getattr(action, "parameters", {}) or {})
  return tvcache_lib.make_cached_tool_call(
      name=function_name,
      arguments=parameters,
      mutates=r2egym_action_mutates(function_name, parameters),
      payload=action,
  )


def _docker(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      ["docker", *args],
      check=check,
      capture_output=True,
      text=True,
  )


def bind_sandbox_cls(owner: Any) -> type:
  """Returns an R2EGymSandbox subclass closed over one SWEEnv episode."""

  class BoundR2EGymSandbox(R2EGymSandbox):
    def __init__(self, task_name: str, env_id: str | None = None):
      super().__init__(task_name, env_id=env_id, owner=owner)

  BoundR2EGymSandbox.__name__ = "BoundR2EGymSandbox"
  BoundR2EGymSandbox.__qualname__ = "BoundR2EGymSandbox"
  return BoundR2EGymSandbox


class R2EGymSandbox:
  """TVCache sandbox that owns an R2E-Gym RepoEnv (or agent-sandbox handle).

  Value-only mode (``snapshot=False``) never calls ``fork``. Snapshot mode
  uses ``docker commit`` for local Docker backends and raises on Kubernetes
  agent-sandbox.

  Prefer ``bind_sandbox_cls(owner)`` so concurrent episodes that share an
  ``instance_id`` do not race on a global context map.
  """

  def __init__(
      self,
      task_name: str,
      env_id: str | None = None,
      *,
      owner: Any | None = None,
  ):
    self.task_name = task_name
    self.env_id = env_id
    self.env: Any = None
    self.handle: Any = None
    self._owner = owner
    self._is_snapshot_handle = False
    if env_id is not None:
      with _SNAPSHOT_LOCK:
        self._is_snapshot_handle = env_id in _SNAPSHOT_IMAGES

  def get_id(self) -> str | None:
    return self.env_id

  def _require_owner(self) -> Any:
    if self._owner is None:
      raise RuntimeError(
          "R2EGymSandbox has no owner; construct via "
          "examples.deepswe.r2egym_tvcache.bind_sandbox_cls(env)"
      )
    return self._owner

  async def start(self) -> None:
    await tvcache_lib.run_in_tvcache_executor(self._start_sync)

  def _start_sync(self) -> None:
    owner = self._require_owner()
    if self.env_id is not None and self._is_snapshot_handle:
      self._start_from_snapshot(owner)
      return
    if self.env_id is not None:
      raise RuntimeError(
          f"R2EGymSandbox.start called with unknown env_id={self.env_id!r}"
      )
    if owner.use_agent_sandbox:
      owner._init_agent_sandbox_env()  # pylint: disable=protected-access
    else:
      owner._init_local_repo_env()  # pylint: disable=protected-access
    self.env = owner.env
    self.handle = getattr(owner, "handle", None)
    self.env_id = f"live-{uuid.uuid4().hex}"

  def _start_from_snapshot(self, owner: Any) -> None:
    with _SNAPSHOT_LOCK:
      image = _SNAPSHOT_IMAGES.get(self.env_id)
    if not image:
      raise KeyError(f"Unknown R2EGym snapshot handle {self.env_id!r}")
    if owner.use_agent_sandbox:
      raise NotImplementedError(
          "R2EGymSandbox snapshots are not supported with agent-sandbox / K8s"
      )
    # Build a RepoEnv from the committed image by rewriting docker_image.
    entry = dict(owner.entry)
    entry["docker_image"] = image
    saved_entry = owner.entry
    owner.entry = entry
    try:
      owner._init_local_repo_env()  # pylint: disable=protected-access
    finally:
      owner.entry = saved_entry
    self.env = owner.env
    self.handle = None
    # Restore produces a new live id; the snapshot handle remains reusable.
    self.env_id = f"live-{uuid.uuid4().hex}"
    self._is_snapshot_handle = False

  async def fork(self) -> R2EGymSandbox:
    return await tvcache_lib.run_in_tvcache_executor(self._fork_sync)

  def _fork_sync(self) -> R2EGymSandbox:
    owner = self._require_owner()
    if owner.use_agent_sandbox:
      raise NotImplementedError(
          "R2EGymSandbox.fork requires local Docker; agent-sandbox has no "
          "snapshot support. Set tvcache_config.snapshot=false (value-only)."
      )
    if self.env is None:
      raise RuntimeError("Cannot fork an unstarted R2EGymSandbox")
    container = _container_name(self.env)
    if not container:
      raise RuntimeError(
          "Cannot resolve Docker container name for R2EGymSandbox.fork"
      )
    handle_id = f"snap-{uuid.uuid4().hex}"
    image_tag = f"tvcache-r2e-{handle_id}"
    _docker(["commit", container, image_tag])
    with _SNAPSHOT_LOCK:
      _SNAPSHOT_IMAGES[handle_id] = image_tag
    return type(self)(task_name=self.task_name, env_id=handle_id)

  async def execute(self, tool_call: Any) -> str:
    return await tvcache_lib.run_in_tvcache_executor(self._execute_sync, tool_call)

  def _execute_sync(self, tool_call: Any) -> str:
    if self.env is None:
      raise RuntimeError("R2EGymSandbox.execute called before start")
    if tvcache_lib.is_test_tool_call(tool_call):
      try:
        reward = self.env.compute_reward()
      except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Failed to execute: {e}"
      return str(float(reward))

    action = getattr(tool_call, "payload", None)
    if action is None:
      raise TypeError(
          "R2EGymSandbox.execute expects a CachedToolCall with an Action "
          f"payload; got {type(tool_call)!r}"
      )
    try:
      obs, reward, done, info = self.env.step(action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"Failed to execute: {e}"
    return json.dumps(
        {
            "obs": str(obs),
            "reward": reward,
            "done": bool(done),
            "info": _json_safe(info),
        },
        sort_keys=True,
        default=str,
    )

  async def stop(self) -> None:
    await tvcache_lib.run_in_tvcache_executor(self._stop_sync)

  def _stop_sync(self) -> None:
    if self._is_snapshot_handle and self.env_id is not None:
      with _SNAPSHOT_LOCK:
        image = _SNAPSHOT_IMAGES.pop(self.env_id, None)
      if image:
        _docker(["rmi", "-f", image], check=False)
      self.env_id = None
      return

    owner = self._owner
    env = self.env
    handle = self.handle
    self.env = None
    self.handle = None

    if env is not None:
      try:
        env.close()
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("R2EGymSandbox failed to close RepoEnv")
      if owner is not None and getattr(owner, "env", None) is env:
        owner.env = None

    fleet = None
    if owner is not None:
      from examples.deepswe import sandbox_utils  # pylint: disable=g-import-not-at-top

      fleet = owner.fleet or getattr(sandbox_utils, "_GLOBAL_FLEET", None)
    if handle is not None and fleet is not None:
      try:
        fleet.release(handle)
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("R2EGymSandbox failed to release sandbox handle")
      if owner is not None and getattr(owner, "handle", None) is handle:
        owner.handle = None

    self.env_id = None


def _container_name(env: Any) -> str | None:
  runtime = getattr(env, "runtime", None)
  if runtime is None:
    return None
  for attr in ("container_name", "cntr_name", "pod_name"):
    value = getattr(runtime, attr, None)
    if value:
      return str(value)
  container = getattr(runtime, "container", None)
  if container is not None:
    name = getattr(container, "name", None) or getattr(container, "id", None)
    if name:
      return str(name)
  return None


def _json_safe(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return str(value)


def clear_snapshot_images_for_tests() -> None:
  """Removes tracked snapshot images (tests only)."""
  with _SNAPSHOT_LOCK:
    images = list(_SNAPSHOT_IMAGES.items())
    _SNAPSHOT_IMAGES.clear()
  for _, image in images:
    _docker(["rmi", "-f", image], check=False)
