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

"""TVCache adapter for Tunix agentic environments.

Provides an opt-in bridge from Tunix tool / sandbox calls to the ``tvclient``
``TvcacheToolExecutor``. ``tvclient`` is imported lazily so Tunix still
imports without the optional ``tvcache`` extra installed.
"""

from __future__ import annotations

import atexit
import asyncio
import concurrent.futures
import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

from tunix.rl.agentic.tools import base_tool

logger = logging.getLogger(__name__)

_TVCACHE_INSTALL_HINT = (
    "TVCache support requires the optional tvclient package. "
    "Install it with: pip install 'google-tunix[tvcache]'"
)

# tvcache-rust listens on this port and does not accept --port.
_SERVER_PORT = 7878
_SERVER_REPO = "https://github.com/abhivijay96/tvcache-rust.git"
_SERVER_MISSING = "tvcache server was not found"
_SERVER_INSTALL_HINT = (
    "enable_tool_cache=true needs the tvcache server, and Rust is not "
    "installed, so Tunix cannot build it. Install Rust, then rerun: "
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
)
_install_lock = threading.Lock()
_SERVER_START_ARGS = (
    "--eviction-policy=ttl_and_watermark",
    "--high-watermark-mb=1024",
    "--low-watermark-mb=768",
)

_server_lock = threading.Lock()
_server_process: subprocess.Popen[bytes] | None = None
_server_log = None
_server_ready = False
_server_atexit_registered = False

# Module-level snapshot store for InMemoryToolSandbox (single-process only).
_IN_MEMORY_SNAPSHOTS: dict[str, dict[str, Any]] = {}
_IN_MEMORY_LOCK = threading.Lock()

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()

# Shared pool for blocking sandbox work (RepoEnv.step, Docker commit, tools).
# Sized to rollout max_concurrency so concurrent episodes do not serialize.
_DEFAULT_MAX_WORKERS = 256
_thread_pool: concurrent.futures.ThreadPoolExecutor | None = None
_thread_pool_lock = threading.Lock()
_thread_pool_max_workers = _DEFAULT_MAX_WORKERS

_CachedToolCallCls: type | None = None

# Threshold large enough that SnapshotPolicy.should_snapshot() never fires.
_VALUE_ONLY_THRESHOLD_SECONDS = 1e18


def _require_tvclient():
  """Imports tvclient or raises a clear InstallError."""
  try:
    import tvclient  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(_TVCACHE_INSTALL_HINT) from e
  return tvclient


def configure_thread_pool(max_workers: int) -> None:
  """Sizes the shared sandbox thread pool (call once from the rollout node)."""
  global _thread_pool, _thread_pool_max_workers
  if max_workers <= 0:
    raise ValueError(f"max_workers must be positive, got {max_workers}")
  with _thread_pool_lock:
    _thread_pool_max_workers = max_workers
    if _thread_pool is not None:
      _thread_pool.shutdown(wait=False)
      _thread_pool = None


def _get_thread_pool() -> concurrent.futures.ThreadPoolExecutor:
  """Returns the process-wide pool used to offload blocking sandbox work."""
  global _thread_pool
  with _thread_pool_lock:
    if _thread_pool is None:
      _thread_pool = concurrent.futures.ThreadPoolExecutor(
          max_workers=_thread_pool_max_workers,
          thread_name_prefix="tunix-tvcache",
      )
    return _thread_pool


def _run(coro: Any, timeout: float | None = None) -> Any:
  """Runs a coroutine on the process-wide TVCache loop and waits.

  ``TrajectoryCollectEngine`` calls the synchronous ``env.step`` from a worker
  thread, so the async tvclient executor needs a loop of its own.
  """
  global _loop
  with _loop_lock:
    if _loop is None:
      loop = asyncio.new_event_loop()

      def _thread() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

      threading.Thread(
          target=_thread, name="tunix-tvcache-loop", daemon=True
      ).start()
      _loop = loop
    loop = _loop
  return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)


async def run_in_tvcache_executor(fn, /, *args, **kwargs) -> Any:
  """Runs a blocking callable on the shared TVCache thread pool."""
  loop = asyncio.get_running_loop()
  return await loop.run_in_executor(
      _get_thread_pool(), lambda: fn(*args, **kwargs)
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TvcacheConfig:
  """Configuration for enabling TVCache on a tool-using environment.

  Attributes:
    host: TVCache server hostname.
    port: TVCache server port.
    sandbox_class: Dotted path to a sandbox class constructed as
      ``(task_name, env_id)`` with ``start``, ``fork``, ``execute``, and
      ``stop``.
    task_id_key: Optional key in the task dict whose value is the cache
      ``task_id``. When None, the task id is a sha256 of the task JSON.
    snapshot: When False (default), value-only mode: hits skip execution and
      misses fast-forward the live sandbox, but tvclient never forks. When
      True, adaptive snapshotting is enabled (requires a fork-capable sandbox).
  """

  host: str = "127.0.0.1"
  port: int = 7878
  sandbox_class: str = "tunix.rl.agentic.tools.tvcache.InMemoryToolSandbox"
  task_id_key: str | None = None
  snapshot: bool = False


def default_config() -> TvcacheConfig:
  """Returns cache settings a user does not have to fill in."""
  return TvcacheConfig()


def _is_executable(path: str) -> bool:
  return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def accepts_tool_cache(env_cls: type) -> bool:
  """Returns whether `env_cls` can use a TVCache config.

  True when `env_cls` or one of its bases names a ``tvcache`` argument.
  A subclass that only has ``**kwargs`` still counts when it forwards them
  to such a base, as ``DeepSWEEnv`` does for ``SWEEnv``. An environment that
  never names ``tvcache``, such as ``GSM8KEnv``, returns False.

  Args:
    env_cls: Environment class the rollout worker will construct.
  """
  for klass in getattr(env_cls, "__mro__", ()):
    init = klass.__dict__.get("__init__")
    if init is None:
      continue
    try:
      parameters = inspect.signature(init).parameters
    except (TypeError, ValueError):
      continue
    if "tvcache" in parameters:
      return True
  return False


def bundled_server_binary() -> str | None:
  """Returns a server binary placed beside this package, for this OS and CPU.

  Looks for a macOS arm64 or Linux x86-64 file under ``_tvcache_bin``.
  Returns None when that file is not part of this install.
  """
  machine = platform.machine().lower()
  if sys.platform == "darwin" and machine in ("arm64", "aarch64"):
    name = "tvcache-darwin-arm64"
  elif sys.platform.startswith("linux") and machine in ("x86_64", "amd64"):
    name = "tvcache-linux-x86_64"
  else:
    return None
  path = os.path.join(os.path.dirname(__file__), "_tvcache_bin", name)
  if not os.path.isfile(path):
    return None
  if not os.access(path, os.X_OK):
    try:
      mode = os.stat(path).st_mode | 0o755
      os.chmod(path, mode)
    except OSError:
      return None
  return path if _is_executable(path) else None


def find_server_binary(binary: str | None = None) -> str:
  """Locates the tvcache server without building or downloading it.

  Search order: ``binary``, ``TVCACHE_BINARY``, ``<sys.prefix>/bin/tvcache``,
  ``tvcache`` on ``PATH``, then a binary placed beside this package.

  Args:
    binary: Explicit path. When set but not executable, the search stops.

  Returns:
    An executable path.

  Raises:
    FileNotFoundError: If no executable server is found.
  """
  explicit: list[str] = []
  if binary and binary.strip():
    explicit.append(binary.strip())
  env_binary = os.environ.get("TVCACHE_BINARY", "").strip()
  if env_binary and env_binary not in explicit:
    explicit.append(env_binary)
  for path in explicit:
    if _is_executable(path):
      return path
    raise FileNotFoundError(
        f"Configured tvcache path is not executable: {path}"
    )
  prefix_binary = os.path.join(sys.prefix, "bin", "tvcache")
  if _is_executable(prefix_binary):
    return prefix_binary
  found = shutil.which("tvcache")
  if found:
    return found
  bundled = bundled_server_binary()
  if bundled:
    return bundled
  raise FileNotFoundError(_SERVER_MISSING)


def server_cache_dir() -> str:
  """Directory where a first-run build of the tvcache server is stored."""
  base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
      os.path.expanduser("~"), ".cache"
  )
  return os.path.join(base, "tunix", "tvcache-rust")


def install_server_binary() -> str:
  """Builds tvcache-rust into the user cache and returns the binary.

  A previous build is reused. This clones upstream source on the user's
  machine; Tunix does not ship the binary.

  Raises:
    FileNotFoundError: If ``cargo`` is not installed.
    RuntimeError: If the clone or build fails.
  """
  with _install_lock:
    root = server_cache_dir()
    binary = os.path.join(root, "target", "release", "tvcache")
    if _is_executable(binary):
      return binary
    if shutil.which("cargo") is None:
      raise FileNotFoundError(_SERVER_INSTALL_HINT)
    if shutil.which("git") is None:
      raise FileNotFoundError(
          "enable_tool_cache=true needs git to fetch the tvcache server."
      )
    os.makedirs(os.path.dirname(root), exist_ok=True)
    if not os.path.isdir(os.path.join(root, ".git")):
      logger.info("Fetching tvcache server into %s", root)
      _run_install(
          ["git", "clone", "--depth", "1", _SERVER_REPO, root], cwd=None
      )
    logger.info(
        "Building tvcache server in %s (first run, this can take a minute)",
        root,
    )
    _run_install(["cargo", "build", "--release"], cwd=root)
    if not _is_executable(binary):
      raise RuntimeError(f"cargo build finished but {binary} is missing")
    return binary


def _run_install(argv: list[str], cwd: str | None) -> None:
  try:
    subprocess.run(argv, cwd=cwd, check=True)
  except subprocess.CalledProcessError as exc:
    raise RuntimeError(
        "Building the tvcache server failed "
        f"({ ' '.join(argv) }, exit {exc.returncode}). "
        "Install Rust from https://rustup.rs and rerun, or set "
        "TVCACHE_BINARY to an existing tvcache binary."
    ) from exc


def resolve_server_binary(binary: str | None = None) -> str:
  """Finds the tvcache server, building it on first use when it is missing.

  An explicit path or ``TVCACHE_BINARY`` that is not executable is an error.
  Tunix does not replace a path the user set.
  """
  try:
    return find_server_binary(binary)
  except FileNotFoundError:
    configured = (binary or "").strip() or os.environ.get(
        "TVCACHE_BINARY", ""
    ).strip()
    if configured:
      raise
    logger.info("tvcache server is not installed; building it")
    return install_server_binary()


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
  try:
    with socket.create_connection((host, port), timeout=0.2):
      return True
  except OSError:
    return False


def _stop_local_server() -> None:
  """Stops the server this process started. A reused server is left running."""
  global _server_process, _server_log, _server_ready
  with _server_lock:
    proc = _server_process
    log = _server_log
    _server_process = None
    _server_log = None
    _server_ready = False
  if proc is not None and proc.poll() is None:
    proc.terminate()
    try:
      proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
      proc.kill()
      proc.wait(timeout=5)
  if log is not None:
    log.close()


def ensure_local_server(
    port: int = _SERVER_PORT,
    binary: str | None = None,
    log_path: str | None = None,
) -> None:
  """Starts the local tvcache server, or keeps one that is already listening.

  Idempotent for this process. A server that was already listening is reused
  and is not stopped at exit. A server this function starts is stopped at
  process exit.

  Args:
    port: Port to wait on. tvcache-rust always binds 7878.
    binary: Optional explicit server path. See ``find_server_binary``.
    log_path: Where to append the server's stdout and stderr.

  Raises:
    ValueError: If ``port`` is not 7878.
    FileNotFoundError: If the server cannot be found and Rust is not installed
      to build it.
    RuntimeError: If the server process exits before it listens.
    TimeoutError: If the port is still closed after about 30 seconds.
  """
  global _server_process, _server_log, _server_ready, _server_atexit_registered
  if port != _SERVER_PORT:
    raise ValueError(
        "tvcache-rust currently binds to port 7878; "
        f"got port={port}. Set the port to 7878 "
        "(or extend the binary to accept --port)."
    )
  with _server_lock:
    if _server_ready:
      return
    if _port_open(port):
      _server_ready = True
      logger.info("Reusing tvcache server already listening on port %s", port)
      return
    proc = _server_process
    if proc is not None and proc.poll() is None:
      owned_log = log_path
    else:
      server = resolve_server_binary(binary)
      if log_path is None:
        log_path = os.path.join(tempfile.gettempdir(), "tunix-tvcache.log")
      owned_log = log_path
      log = open(log_path, "ab")  # pylint: disable=consider-using-with
      proc = subprocess.Popen(  # pylint: disable=consider-using-with
          [server, *_SERVER_START_ARGS],
          stdout=log,
          stderr=subprocess.STDOUT,
          start_new_session=True,
      )
      _server_process = proc
      _server_log = log
      if not _server_atexit_registered:
        atexit.register(_stop_local_server)
        _server_atexit_registered = True
  deadline = time.monotonic() + 30
  while time.monotonic() < deadline:
    if proc.poll() is not None:
      _stop_local_server()
      raise RuntimeError(
          "tvcache server exited before listening on port "
          f"{port}. See {owned_log}"
      )
    if _port_open(port):
      with _server_lock:
        _server_ready = True
      logger.info("Started tvcache server pid=%s on port %s", proc.pid, port)
      return
    time.sleep(0.1)
  _stop_local_server()
  raise TimeoutError(
      f"tvcache server did not listen on port {port} within 30s. "
      f"See {owned_log}"
  )


def task_id_for(task: Mapping[str, Any], task_id_key: str | None) -> str:
  """Derives a stable TVCache task id from the episode task dict."""
  if task_id_key is not None:
    if task_id_key not in task:
      raise KeyError(
          f"tvcache task_id_key {task_id_key!r} not found in task keys "
          f"{sorted(task.keys())}"
      )
    return str(task[task_id_key])
  payload = json.dumps(dict(task), sort_keys=True, default=str)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_cached_tool_call_cls() -> type:
  """Lazily defines CachedToolCall once tvclient is available."""
  global _CachedToolCallCls
  if _CachedToolCallCls is not None:
    return _CachedToolCallCls

  tvclient = _require_tvclient()

  class CachedToolCall(tvclient.ToolCall):  # type: ignore[misc, valid-type]
    """Generic tvclient ToolCall with an opaque payload for the sandbox."""

    def __init__(
        self,
        name: str,
        arguments: Mapping[str, Any],
        mutates: bool,
        payload: Any = None,
    ):
      self.name = name
      self.arguments = dict(arguments)
      self._mutates = bool(mutates)
      self.payload = payload

    def to_dict(self) -> dict[str, Any]:
      return {"name": self.name, "arguments": self.arguments}

    def will_mutate_state(self) -> bool:
      return self._mutates

  _CachedToolCallCls = CachedToolCall
  return CachedToolCall


def make_cached_tool_call(
    name: str,
    arguments: Mapping[str, Any],
    *,
    mutates: bool,
    payload: Any = None,
) -> Any:
  """Constructs a CachedToolCall for one sandbox invocation."""
  return _get_cached_tool_call_cls()(
      name=name,
      arguments=arguments,
      mutates=mutates,
      payload=payload,
  )


def make_tool_call(tool: base_tool.BaseTool, call: base_tool.ToolCall) -> Any:
  """Constructs a CachedToolCall for one BaseTool invocation."""
  return make_cached_tool_call(
      name=call.name,
      arguments=call.arguments,
      mutates=bool(tool.will_mutate_state(**call.arguments)),
      payload=tool,
  )


def is_test_tool_call(tool_call: Any) -> bool:
  """True when ``tool_call`` is tvclient's terminal TestToolCall sentinel."""
  to_dict = getattr(tool_call, "to_dict", None)
  if to_dict is None:
    return False
  try:
    payload = to_dict()
  except Exception:  # pylint: disable=broad-exception-caught
    return False
  return (
      isinstance(payload, Mapping)
      and payload.get("TV_CACHE_TOOL_TYPE") == "TESTING TOOL"
  )


class InMemoryToolSandbox:
  """In-process sandbox with deepcopy snapshots for tests and examples.

  A user sandbox is any class constructed as ``(task_name, env_id)`` that
  implements ``start``, ``fork``, ``execute``, and ``stop``. This one keeps
  snapshots in a process-local dict. Blocking tool work is offloaded to the
  shared TVCache thread pool so concurrent episodes do not serialize on the
  asyncio loop.
  """

  def __init__(self, task_name: str, env_id: str | None = None):
    self.task_name = task_name
    self.env_id = env_id
    self.state: dict[str, Any] = {}

  def get_id(self) -> str | None:
    return self.env_id

  async def start(self) -> None:
    """Boots a fresh sandbox or restores a snapshot handle."""
    await run_in_tvcache_executor(self._start_sync)

  def _start_sync(self) -> None:
    if self.env_id is None:
      self.env_id = f"live-{uuid.uuid4().hex}"
      self.state = {}
      return
    with _IN_MEMORY_LOCK:
      if self.env_id not in _IN_MEMORY_SNAPSHOTS:
        raise KeyError(
            f"InMemoryToolSandbox snapshot {self.env_id!r} not found"
        )
      self.state = copy.deepcopy(_IN_MEMORY_SNAPSHOTS[self.env_id])
    # Restore produces a new live id; the snapshot handle remains reusable.
    self.env_id = f"live-{uuid.uuid4().hex}"

  async def fork(self) -> InMemoryToolSandbox:
    """Snapshots this live sandbox into a cold handle."""
    return await run_in_tvcache_executor(self._fork_sync)

  def _fork_sync(self) -> InMemoryToolSandbox:
    handle_id = f"snap-{uuid.uuid4().hex}"
    with _IN_MEMORY_LOCK:
      _IN_MEMORY_SNAPSHOTS[handle_id] = copy.deepcopy(self.state)
    return InMemoryToolSandbox(task_name=self.task_name, env_id=handle_id)

  async def execute(self, tool_call: Any) -> str:
    """Runs one Tunix tool call and returns a string for the cache."""
    return await run_in_tvcache_executor(self._execute_sync, tool_call)

  def _execute_sync(self, tool_call: Any) -> str:
    if is_test_tool_call(tool_call):
      return str(self.state.get("_reward", 0.0))
    tool = getattr(tool_call, "payload", None)
    if tool is None:
      raise TypeError(
          "InMemoryToolSandbox.execute expects a CachedToolCall with a "
          f"`.payload` tool; got {type(tool_call)!r}"
      )
    arguments = getattr(tool_call, "arguments", {})
    output = tool.apply_in_sandbox(self, **arguments)
    if not isinstance(output, base_tool.ToolOutput):
      raise TypeError(
          f"apply_in_sandbox must return ToolOutput; got {type(output)!r}"
      )
    if output.error:
      return f"Failed to execute: {output.error}"
    return repr(output)

  async def stop(self) -> None:
    """Deletes this sandbox or snapshot from the in-memory store."""
    await run_in_tvcache_executor(self._stop_sync)

  def _stop_sync(self) -> None:
    if self.env_id is None:
      return
    with _IN_MEMORY_LOCK:
      _IN_MEMORY_SNAPSHOTS.pop(self.env_id, None)
    self.env_id = None
    self.state = {}


class TvcacheBridge:
  """Owns one TvcacheToolExecutor and the call history for a single episode."""

  def __init__(
      self,
      config: TvcacheConfig,
      task_id: str,
      *,
      sandbox_cls: type | None = None,
  ):
    """Creates a bridge for one episode.

    Args:
      config: TVCache connection / policy settings.
      task_id: Cache namespace (shared across GRPO samples of the same task).
      sandbox_cls: Optional sandbox class object. When set, skips importing
        ``config.sandbox_class``. DeepSWE passes a per-episode bound class so
        concurrent samples that share ``task_id`` do not race.
    """
    tvclient = _require_tvclient()
    if sandbox_cls is None:
      module_path, class_name = config.sandbox_class.rsplit(".", 1)
      sandbox_cls = getattr(importlib.import_module(module_path), class_name)
    self._task_id = task_id
    self._history: list[Any] = []
    adaptive = bool(config.snapshot)
    fixed_threshold = 0.0 if adaptive else _VALUE_ONLY_THRESHOLD_SECONDS
    self._executor = tvclient.TvcacheToolExecutor(
        sandbox_cls,
        _get_cached_tool_call_cls(),
        task_id,
        host=config.host,
        port=config.port,
        adaptive=adaptive,
        fixed_threshold_seconds=fixed_threshold,
    )
    self._connected = False
    self._lock = threading.Lock()

  def _ensure_connected(self) -> None:
    if self._connected:
      return
    with self._lock:
      if self._connected:
        return

      async def _connect() -> None:
        await self._executor.__aenter__()

      _run(_connect())
      self._connected = True

  def call(self, tool_call: Any) -> str:
    """Appends ``tool_call`` to history and looks up or executes it."""
    self._history.append(tool_call)
    self._ensure_connected()
    history_list = list(self._history)

    async def _lookup() -> str:
      return await self._executor.lookup_or_execute(history_list)

    return _run(_lookup())

  def test(self) -> str:
    """Runs the terminal cached test call and stops the live sandbox."""
    self._ensure_connected()
    history_list = list(self._history)

    async def _test() -> str:
      return await self._executor.test(history_list)

    return _run(_test())

  def close(self) -> None:
    """Stops the live sandbox and closes the cache connection."""
    with self._lock:
      if not self._connected:
        self._history = []
        return

      async def _close() -> None:
        await self._executor.close()

      try:
        _run(_close())
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "TvcacheBridge.close failed for task %s", self._task_id
        )
      finally:
        self._connected = False
        self._history = []


def clear_in_memory_snapshots() -> None:
  """Clears the in-memory snapshot store (for tests)."""
  with _IN_MEMORY_LOCK:
    _IN_MEMORY_SNAPSHOTS.clear()
