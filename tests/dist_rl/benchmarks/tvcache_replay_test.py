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

"""Unit tests for the TVCache replay comparison helpers."""

from absl.testing import absltest
from tunix.dist_rl.benchmarks.tvcache import replay


class ReplayComparisonTest(absltest.TestCase):

  def test_normalize_blanks_durations_and_learned_lines(self):
    extra = replay.learned_volatile_patterns(
        {"s": "pid=4 took 1.2s"}, {"s": "pid=9 took 3s"}
    )
    self.assertTrue(
        replay.outputs_equivalent(
            "pid=4 took 1.2s", "pid=9 took 3s", extra
        )
    )
    self.assertTrue(
        replay.outputs_equivalent("done in 3.4 seconds", "done in 9 seconds")
    )
    self.assertFalse(replay.outputs_equivalent("alpha", "beta"))

  def test_empty_cache_and_repo_state(self):
    self.assertTrue(
        replay.cache_is_empty({"tasks": {"other/abc": {}}}, "run-1")
    )
    self.assertFalse(
        replay.cache_is_empty({"tasks": {"run-1/abc": {}}}, "run-1")
    )
    self.assertTrue(
        replay.repo_states_equal(
            {"status": "", "diff": "", "files_sha256": "aa"},
            {"status": "", "diff": "", "files_sha256": "aa"},
        )
    )

  def test_scripted_case_requires_a_grading_hit(self):
    good = {
        "edit_executed": False,
        "replayed_edit": 1,
        "read_executed": 1,
        "read_matches_cache_off": True,
        "grading_hit": True,
        "reward": 1.0,
        "cache_off_reward": 1.0,
    }
    self.assertEqual(replay.scripted_replay_ok(good), [])
    missed = dict(good)
    missed["grading_hit"] = False
    self.assertIn(
        "the warm scripted grade was not a cache hit",
        replay.scripted_replay_ok(missed),
    )

  def test_grading_hit_rate_lists_evictions(self):
    report = replay.grading_hit_rate([
        {"episode_id": "a", "grading_hit": True},
        {
            "episode_id": "b",
            "grading_hit": False,
            "grading_miss_reason": "evicted",
        },
    ])
    self.assertEqual(report["rate"], 0.5)
    self.assertEqual(report["misses"], [{"episode": "b", "reason": "evicted"}])

  def test_singleflight_waits_for_counts(self):
    self.assertEqual(replay.singleflight_decision(None, None), "not-yet")
    self.assertEqual(replay.singleflight_decision(1, 10), "leave")
    self.assertEqual(replay.singleflight_decision(6, 10), "per-process")


if __name__ == "__main__":
  absltest.main()
