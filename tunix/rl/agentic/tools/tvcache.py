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
import enum
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
      Recovery after a cache error runs only when this is False.
    namespace: Run id that prefixes every cache key. Empty uses a per-process
      id, so a notebook does not share keys with another process.
    rpc_timeout_s: Timeout for one cache RPC after the connection is up.
    connect_timeout_s: Timeout for the episode's connect attempt.
    connect_retry_after_s: After a failed connect, later episodes in this
      process skip the attempt until this many seconds have passed.
  """

  host: str = "127.0.0.1"
  port: int = 7878
  sandbox_class: str = "tunix.rl.agentic.tools.tvcache.InMemoryToolSandbox"
  task_id_key: str | None = None
  snapshot: bool = False
  namespace: str = ""
  rpc_timeout_s: float = 5.0
  connect_timeout_s: float = 2.0
  connect_retry_after_s: float = 60.0


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


# Notebooks have no launcher run id. One id per process keeps their keys
# off any other process's cache entries.
_PROCESS_NAMESPACE = f"local-{uuid.uuid4().hex}"

_LOCAL_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1", "0.0.0.0"})


def is_local_host(host: str | None) -> bool:
  """True when ``host`` is this machine, so the rollout may start the server.

  A name that does not resolve is remote. Kubernetes service names fail closed
  here, which is what a rollout Pod wants: it should not start a local server
  for a Service it cannot see yet.
  """
  if host is None:
    return True
  normalized = host.strip().lower()
  if normalized in _LOCAL_HOSTS:
    return True
  try:
    infos = socket.getaddrinfo(normalized, None)
  except socket.gaierror:
    return False
  local_ips = {"127.0.0.1", "::1"}
  try:
    for item in socket.getaddrinfo(socket.gethostname(), None):
      local_ips.add(item[4][0])
  except socket.gaierror:
    pass
  return any(item[4][0] in local_ips for item in infos)


def cache_task_id(
    namespace: str | None,
    task: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    *,
    task_id_key: str | None = None,
) -> str:
  """Returns ``<namespace>/<sha256>`` for one episode's cache identity.

  ``task_id_key`` still selects the task part, via :func:`task_id_for`.
  ``extra`` carries whatever else would make a cached value wrong to reuse,
  such as the sandbox image and the grading settings. An empty namespace
  uses a per-process id.
  """
  identity: dict[str, Any] = {"task": task_id_for(task, task_id_key)}
  if extra:
    identity["extra"] = json.loads(
        json.dumps(dict(extra), sort_keys=True, default=str)
    )
  digest = hashlib.sha256(
      json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
  ).hexdigest()
  scope = (namespace or "").strip() or _PROCESS_NAMESPACE
  return f"{scope}/{digest}"


class CacheUnavailable(Exception):
  """A TVCache RPC failed. The sandbox may still hold a usable workspace."""


class _CallState(enum.Enum):
  NOT_STARTED = "not_started"
  STARTED = "started"
  COMPLETED = "completed"
  RAISED = "raised"


@dataclasses.dataclass
class _CacheStats:
  """Counters one episode reports through ``tool_cache_stats``."""

  hits: int = 0
  misses: int = 0
  fallback_completed: int = 0
  fallback_not_started: int = 0
  fallback_error: int = 0
  connect_skipped: int = 0
  bypassed: int = 0
  grading_hit: int = 0

  def as_dict(self) -> dict[str, int]:
    return dataclasses.asdict(self)


class _CallTracker:
  """Records whether the current call has run inside ``sandbox.execute``."""

  def __init__(self) -> None:
    self.current: Any = None
    self.expect_test = False
    self.state = _CallState.NOT_STARTED
    self.output: str | None = None

  def begin(self, tool_call: Any, *, expect_test: bool = False) -> None:
    self.current = tool_call
    self.expect_test = expect_test
    self.state = _CallState.NOT_STARTED
    self.output = None

  def matches(self, tool_call: Any) -> bool:
    if self.current is not None and tool_call is self.current:
      return True
    return self.expect_test and is_test_tool_call(tool_call)

  def mark_started(self) -> None:
    self.state = _CallState.STARTED
    self.output = None

  def mark_completed(self, output: str) -> None:
    self.state = _CallState.COMPLETED
    self.output = output

  def mark_raised(self) -> None:
    self.state = _CallState.RAISED
    self.output = None


def _tracked(sandbox_cls: type, tracker: _CallTracker) -> type:
  """Returns a sandbox subclass that updates ``tracker`` for the current call.

  Replay executes the history objects, not the object the bridge passed in, so
  identity tells them apart. ``test()`` builds its own ``TestToolCall``, so
  that path matches with :func:`is_test_tool_call`. A ``CacheUnavailable``
  raised from ``execute`` leaves the state at ``STARTED``: the call's outcome
  is unknown.
  """

  class TrackedSandbox(sandbox_cls):
    """Sandbox whose ``execute`` reports the current call to ``tracker``."""

    async def execute(self, tool_call: Any) -> str:
      current = tracker.matches(tool_call)
      if current:
        tracker.mark_started()
      try:
        value = await super().execute(tool_call)
      except CacheUnavailable:
        raise
      except Exception:
        if current:
          tracker.mark_raised()
        raise
      if current:
        tracker.mark_completed(value)
      return value

  TrackedSandbox.__name__ = f"Tracked{sandbox_cls.__name__}"
  return TrackedSandbox


def _call_mutates(tool_call: Any) -> bool:
  will_mutate = getattr(tool_call, "will_mutate_state", None)
  if not callable(will_mutate):
    return True
  return bool(will_mutate())


def _call_serialize(tool_call: Any) -> str:
  return str(tool_call.serialize())


async def _stop_sandbox(sandbox: Any) -> None:
  if sandbox is None:
    return
  try:
    await sandbox.stop()
  except Exception:  # pylint: disable=broad-exception-caught
    logger.debug("Failed to stop TVCache sandbox", exc_info=True)


class _WorkspaceRecovery:
  """Rebuilds a sandbox from the episode's mutating calls, without the server.

  The caller owns the sandbox this returns. A sandbox this class stops is not
  returned.
  """

  @staticmethod
  async def catch_up(
      sandbox: Any,
      applied: list[str],
      history: list[Any],
      sandbox_cls: type,
      task_name: str,
  ) -> Any:
    """Replays mutating calls missing from ``applied`` and returns the sandbox.

    ``applied`` is the executor's ``sandbox_history``: serialized mutating
    calls already executed. When it is a prefix of the episode chain, that
    sandbox is kept. Otherwise it is stopped and a fresh one is started.
    """
    mutating = [call for call in history if _call_mutates(call)]
    serialized = [_call_serialize(call) for call in mutating]
    applied_list = list(applied)
    reusable = (
        sandbox is not None and serialized[: len(applied_list)] == applied_list
    )
    if not reusable:
      await _stop_sandbox(sandbox)
      sandbox = sandbox_cls(task_name=task_name)
      try:
        await sandbox.start()
      except Exception:
        await _stop_sandbox(sandbox)
        raise
      start_index = 0
    else:
      start_index = len(applied_list)
    try:
      for call in mutating[start_index:]:
        await sandbox.execute(call)
    except Exception:
      await _stop_sandbox(sandbox)
      raise
    return sandbox


def _cache_error_types(tvclient: Any) -> tuple[type[BaseException], ...]:
  types: list[type[BaseException]] = [
      OSError,
      EOFError,
      asyncio.IncompleteReadError,
      asyncio.TimeoutError,
      CacheUnavailable,
  ]
  server_error = getattr(tvclient, "TVCacheServerError", None)
  if isinstance(server_error, type):
    types.append(server_error)
  return tuple(types)


class _ExecutorAdapter:
  """The only code that reads tvclient executor internals.

  Pinned to tvclient ``45d8c7d``. Contract tests lock these facts in:

  * ``lookup_or_execute`` runs the same current-call object the bridge passed.
  * ``sandbox_history`` is updated before ``insert``.
  * A failed ``insert`` leaves ``rollout_sandbox`` in place. ``snapshot: true``
    forks before ``insert``, so recovery stays off on that path.
  * ``TVCacheClient`` is created in ``TvcacheToolExecutor.__init__``.
    ``SnapshotPolicy`` keeps its own reference, so this adapter does not
    replace the client object on the production path. It classifies errors at
    this boundary instead.
  """

  def __init__(
      self,
      config: TvcacheConfig,
      task_id: str,
      sandbox_cls: type,
      tracker: _CallTracker,
      stats: _CacheStats,
  ):
    tvclient = _require_tvclient()
    self._tvclient = tvclient
    self._config = config
    self._tracker = tracker
    self._stats = stats
    self._grading = False
    self._error_types = _cache_error_types(tvclient)
    adaptive = bool(config.snapshot)
    fixed_threshold = 0.0 if adaptive else _VALUE_ONLY_THRESHOLD_SECONDS
    self.executor = tvclient.TvcacheToolExecutor(
        sandbox_cls,
        _get_cached_tool_call_cls(),
        task_id,
        host=config.host,
        port=int(config.port),
        adaptive=adaptive,
        fixed_threshold_seconds=fixed_threshold,
    )
    self._install_client(self.executor.client)

  def _install_client(self, client: Any) -> None:
    client.retries = 0
    client.timeout = float(self._config.rpc_timeout_s)
    original_find = client.find
    stats = self._stats

    async def _find(*args: Any, **kwargs: Any) -> Any:
      result = await original_find(*args, **kwargs)
      if getattr(result, "value", None) is not None:
        stats.hits += 1
        if self._grading:
          stats.grading_hit += 1
      else:
        stats.misses += 1
      return result

    client.find = _find
    self.executor.client = client
    self.executor.snapshot_policy.client = client

  def bind_client(self, client: Any) -> None:
    """Points the executor at ``client`` and counts its ``find`` results.

    Tests use this for a scripted client. Production uses the client the
    executor constructed.
    """
    self._install_client(client)

  def _apply_connect_timeout(self) -> None:
    self.executor.client.timeout = float(self._config.connect_timeout_s)
    self.executor.client.retries = 0

  def _apply_rpc_timeout(self) -> None:
    self.executor.client.timeout = float(self._config.rpc_timeout_s)

  async def connect(self) -> None:
    self._apply_connect_timeout()
    await self.executor.__aenter__()
    self._apply_rpc_timeout()

  def _translate(self, exc: BaseException) -> None:
    if self._tracker.state is _CallState.RAISED:
      raise exc
    if isinstance(exc, self._error_types):
      if isinstance(exc, CacheUnavailable):
        raise exc
      raise CacheUnavailable(str(exc)) from exc
    raise exc

  async def lookup(self, history: list[Any]) -> str:
    try:
      return await self.executor.lookup_or_execute(list(history))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._translate(exc)
      raise

  async def test(self, history: list[Any]) -> str:
    self._grading = True
    try:
      return await self.executor.test(list(history))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._translate(exc)
      raise
    finally:
      self._grading = False

  def release_sandbox(self) -> tuple[Any, list[str]]:
    """Hands the live sandbox to the caller and forgets it on the executor."""
    executor = self.executor
    sandbox = executor.rollout_sandbox
    applied = list(executor.sandbox_history)
    executor.rollout_sandbox = None
    executor.sandbox_history = []
    return sandbox, applied

  async def close(self) -> None:
    await self.executor.close()


_connect_state_lock = threading.Lock()
_connect_unavailable_until = 0.0
_connect_down = False


def reset_connect_backoff_for_test() -> None:
  """Clears the process-wide connect backoff. Tests call this."""
  global _connect_unavailable_until, _connect_down
  with _connect_state_lock:
    _connect_unavailable_until = 0.0
    _connect_down = False


def _connect_suppressed() -> bool:
  with _connect_state_lock:
    return time.monotonic() < _connect_unavailable_until


def _note_connect_failure(config: TvcacheConfig) -> None:
  global _connect_unavailable_until, _connect_down
  with _connect_state_lock:
    _connect_unavailable_until = (
        time.monotonic() + float(config.connect_retry_after_s)
    )
    if _connect_down:
      return
    _connect_down = True
  logger.warning(
      "TVCache unavailable at %s:%s; running uncached",
      config.host,
      config.port,
  )


def _note_connect_success() -> None:
  global _connect_down
  with _connect_state_lock:
    if not _connect_down:
      return
    _connect_down = False
    _connect_unavailable_until = 0.0
  logger.warning("TVCache reachable again")


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
  """Owns one TvcacheToolExecutor and the call history for a single episode.

  With ``snapshot: false``, a cache error falls open: a finished command's
  output is returned, a command that has not started is run once after the
  workspace is caught up, and an uncertain or failed recovery ends the
  episode. The episode then stays uncached. ``snapshot: true`` keeps today's
  path, except a failed connect still runs the episode uncached.
  """

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
    if sandbox_cls is None:
      module_path, class_name = config.sandbox_class.rsplit(".", 1)
      sandbox_cls = getattr(importlib.import_module(module_path), class_name)
    self._config = config
    self._task_id = task_id
    self._sandbox_cls = sandbox_cls
    self._history: list[Any] = []
    self._stats = _CacheStats()
    self._tracker = _CallTracker()
    self._adapter = _ExecutorAdapter(
        config,
        task_id,
        _tracked(sandbox_cls, self._tracker),
        self._tracker,
        self._stats,
    )
    self._connected = False
    self._bypass = False
    self._bypass_sandbox: Any = None
    self._lock = threading.Lock()

  def stats(self) -> dict[str, int]:
    """Returns a copy of this episode's cache counters."""
    return self._stats.as_dict()

  def _test_call(self) -> Any:
    tvclient = _require_tvclient()
    return tvclient.TestToolCall("")

  async def _start_fresh(self) -> Any:
    sandbox = self._sandbox_cls(task_name=self._task_id)
    await sandbox.start()
    return sandbox

  def _enter_bypass(self, sandbox: Any) -> None:
    self._bypass = True
    self._bypass_sandbox = sandbox
    self._stats.bypassed = 1

  def _open_uncached(self) -> None:
    if self._bypass_sandbox is not None:
      self._bypass = True
      self._stats.bypassed = 1
      return
    self._enter_bypass(_run(self._start_fresh()))

  def _try_connect(self) -> bool:
    """Connects once, or starts an uncached sandbox when the cache is down."""
    if self._bypass:
      return False
    if self._connected:
      return True
    if _connect_suppressed():
      self._stats.connect_skipped += 1
      self._open_uncached()
      return False
    try:
      _run(
          self._adapter.connect(),
          timeout=float(self._config.connect_timeout_s) + 1.0,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      _note_connect_failure(self._config)
      self._open_uncached()
      return False
    _note_connect_success()
    self._connected = True
    return True

  def _recover(self, current: Any, *, grading: bool) -> str:
    state = self._tracker.state
    if state is _CallState.COMPLETED and self._tracker.output is not None:
      self._stats.fallback_completed += 1
      sandbox, _applied = self._adapter.release_sandbox()
      self._enter_bypass(sandbox)
      return self._tracker.output
    if state is _CallState.NOT_STARTED:
      self._stats.fallback_not_started += 1
      sandbox, applied = self._adapter.release_sandbox()
      # The current call is already on the history, and it has not run.
      # Replay only the edits before it, then run it once below.
      prior = list(self._history)
      if not grading and prior and prior[-1] is current:
        prior = prior[:-1]
      try:
        sandbox = _run(
            _WorkspaceRecovery.catch_up(
                sandbox,
                applied,
                prior,
                self._sandbox_cls,
                self._task_id,
            )
        )
      except Exception:
        self._stats.fallback_error += 1
        raise
      self._enter_bypass(sandbox)
      if grading:
        return _run(sandbox.execute(self._test_call()))
      return _run(sandbox.execute(current))
    self._stats.fallback_error += 1
    raise CacheUnavailable(
        f"TVCache failed while the current call was {state.value}"
    )

  def call(self, tool_call: Any) -> str:
    """Appends ``tool_call`` to history and looks up or executes it."""
    self._history.append(tool_call)
    if self._bypass:
      return _run(self._bypass_sandbox.execute(tool_call))
    if not self._try_connect():
      return _run(self._bypass_sandbox.execute(tool_call))
    history = list(self._history)
    if self._config.snapshot:
      return _run(self._adapter.executor.lookup_or_execute(history))
    self._tracker.begin(tool_call)
    try:
      return _run(self._adapter.lookup(history))
    except CacheUnavailable:
      return self._recover(tool_call, grading=False)

  def test(self) -> str:
    """Runs the terminal test call. A cache hit does not rebuild the sandbox."""
    if self._bypass:
      return _run(self._bypass_sandbox.execute(self._test_call()))
    if not self._try_connect():
      return _run(self._bypass_sandbox.execute(self._test_call()))
    history = list(self._history)
    if self._config.snapshot:
      return _run(self._adapter.executor.test(history))
    self._tracker.begin(None, expect_test=True)
    try:
      return _run(self._adapter.test(history))
    except CacheUnavailable:
      return self._recover(None, grading=True)

  def materialize(self) -> Any:
    """Returns a sandbox with every mutating call applied.

    Enters bypass. Requires ``snapshot: false``. The benchmark audit runs
    commands on this sandbox, outside the cache.
    """
    if self._config.snapshot:
      raise RuntimeError("materialize() requires snapshot=false")
    if self._bypass_sandbox is not None:
      return self._bypass_sandbox
    sandbox: Any = None
    applied: list[str] = []
    if self._connected:
      sandbox, applied = self._adapter.release_sandbox()
    sandbox = _run(
        _WorkspaceRecovery.catch_up(
            sandbox,
            applied,
            list(self._history),
            self._sandbox_cls,
            self._task_id,
        )
    )
    self._enter_bypass(sandbox)
    return sandbox

  def close(self) -> None:
    """Stops the sandbox this bridge owns, then closes the executor."""
    with self._lock:

      async def _close() -> None:
        sandbox = self._bypass_sandbox
        self._bypass_sandbox = None
        if sandbox is not None:
          await _stop_sandbox(sandbox)
        if self._connected:
          await self._adapter.close()

      try:
        if self._connected or self._bypass_sandbox is not None:
          _run(_close())
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "TvcacheBridge.close failed for task %s", self._task_id
        )
      finally:
        self._connected = False
        self._bypass = False
        self._bypass_sandbox = None
        self._history = []


def clear_in_memory_snapshots() -> None:
  """Clears the in-memory snapshot store (for tests)."""
  with _IN_MEMORY_LOCK:
    _IN_MEMORY_SNAPSHOTS.clear()
