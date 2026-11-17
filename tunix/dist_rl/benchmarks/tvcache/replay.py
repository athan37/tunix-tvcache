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

"""Compare cache-off and cache-on replays of the same DeepSWE episodes.

The Kubernetes job and the fault runs are described in ``README.md``. This
module is the part that decides whether a replay passed: normalized tool
output, reward, repo state, and grading hits. It does not start a cluster.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

# Durations, memory addresses, temp paths, and process ids. Anything else that
# differs between two cache-off passes is added at runtime.
_VOLATILE = (
    re.compile(r"\b\d+(?:\.\d+)?s\b"),
    re.compile(r"\bin \d+(?:\.\d+)? seconds\b"),
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"/tmp/[A-Za-z0-9._/-]+"),
    re.compile(r"\bpid[=: ]+\d+\b", re.IGNORECASE),
)

# Duplicate misses at or above this share of misses justify per-process
# single-flight. Cross-pod de-duplication still needs a server claim op.
SINGLEFLIGHT_SHARE = 0.5


def normalize_output(text: str, extra: tuple[re.Pattern[str], ...] = ()) -> str:
  """Blanks volatile fields so two real runs of one command can be compared."""
  normalized = text
  for pattern in (*_VOLATILE, *extra):
    normalized = pattern.sub("<volatile>", normalized)
  return normalized


def learned_volatile_patterns(
    first: Mapping[str, str], second: Mapping[str, str]
) -> tuple[re.Pattern[str], ...]:
  """Returns patterns for lines that already differ with the cache off.

  ``first`` and ``second`` map a step id to that step's raw output.
  """
  patterns: list[re.Pattern[str]] = []
  for key, left in first.items():
    right = second.get(key)
    if right is None or left == right:
      continue
    # Quote the whole left value. A later pass treats that exact string as
    # volatile, which is what a second cache-off run already disagreed on.
    patterns.append(re.compile(re.escape(left)))
  return tuple(patterns)


def outputs_equivalent(
    actual: str,
    expected: str,
    extra: tuple[re.Pattern[str], ...] = (),
) -> bool:
  return normalize_output(actual, extra) == normalize_output(expected, extra)


def cache_is_empty(dump: Mapping[str, Any], namespace: str) -> bool:
  """True when ``dump_all`` has no task under ``namespace``."""
  prefix = namespace.rstrip("/") + "/"
  tasks = dump.get("tasks", dump)
  if isinstance(tasks, Mapping):
    return not any(str(key).startswith(prefix) for key in tasks)
  if isinstance(tasks, list):
    return not any(str(item).startswith(prefix) for item in tasks)
  return False


def repo_states_equal(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
  """Compares status, diff, and file hashes from a materialized workspace."""
  keys = ("status", "diff", "files_sha256")
  return all(left.get(key) == right.get(key) for key in keys)


def grading_hit_rate(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
  """Reports grading hits. A miss is ``absent`` or ``evicted``.

  Eviction can legitimately miss, so the rate is reported rather than required
  for the 16-episode workload. The scripted warm case is the exception: it
  must hit.
  """
  hits = 0
  misses: list[dict[str, str]] = []
  for episode in episodes:
    if episode.get("grading_hit"):
      hits += 1
      continue
    misses.append({
        "episode": str(episode.get("episode_id", "")),
        "reason": str(episode.get("grading_miss_reason", "absent")),
    })
  total = len(episodes)
  return {
      "hits": hits,
      "total": total,
      "rate": (hits / total) if total else 0.0,
      "misses": misses,
  }


def scripted_replay_ok(episode: Mapping[str, Any]) -> list[str]:
  """Checks the edit-hit then read-miss case. Returns failure strings."""
  failures: list[str] = []
  if episode.get("edit_executed"):
    failures.append("cached edit was executed")
  if episode.get("replayed_edit") != 1:
    failures.append("edit was not replayed exactly once on the miss")
  if episode.get("read_executed") != 1:
    failures.append("the read did not run exactly once")
  if not episode.get("read_matches_cache_off"):
    failures.append("the read does not match the cache-off output")
  if not episode.get("grading_hit"):
    failures.append("the warm scripted grade was not a cache hit")
  if episode.get("reward") != episode.get("cache_off_reward"):
    failures.append("reward differs from cache-off")
  return failures


def singleflight_decision(
    duplicate_misses: int | None, misses: int | None
) -> str:
  """Decides whether to add single-flight from a finished benchmark.

  Returns ``not-yet`` when the four-replica run has not reported counts.
  ``per-process`` when duplicate misses are at least half of misses.
  ``leave`` otherwise. Cross-pod de-duplication is an upstream claim op
  either way; this function does not add it.
  """
  if duplicate_misses is None or misses is None or misses <= 0:
    return "not-yet"
  if duplicate_misses / misses >= SINGLEFLIGHT_SHARE:
    return "per-process"
  return "leave"


def main() -> int:
  """Prints how to run the comparison. The cluster passes are in the README."""
  print(
      "TVCache replay helpers. Run the passes in "
      "tunix/dist_rl/benchmarks/tvcache/README.md. "
      f"singleflight={singleflight_decision(None, None)}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
