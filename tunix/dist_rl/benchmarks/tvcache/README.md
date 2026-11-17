# TVCache benchmark

Same 16 DeepSWE episodes (2 R2E-Gym instances, 8 generations, `max_turns: 20`,
`snapshot: false`), cache off versus on. The comparison helpers live in
`replay.py`. This run is the production check, together with the pinned
executor tests in `tests/rl/agentic/tools/tvcache_recovery_test.py`.

## Network

`job.yaml` is an Indexed Job of 4 Pods plus a one-shot `dump_all` Pod. Both
carry `tunix.dev/tvcache-bench=<run_id>`. The TVCache NetworkPolicy admits
that label and this run's rollout JobSets, and nothing else. Before the
passes, start a Pod without the label and confirm it cannot open port 7878.

Replace `RUN_ID`, `TVCACHE_ID`, and `TUNIX_IMAGE` in `job.yaml`. `TVCACHE_ID`
is the Service name `<run_id>-tvcache`.

## Passes

1. **Record.** One live run with the cache off and
   `trajectory_store_config.enable: true`. Save each episode's action list.
2. **Cache off, twice.** Record per-step outputs, the reward, and the grade.
   Fields that differ between these two passes are volatile, on top of the
   fixed patterns in `replay.normalize_output` (durations, addresses, temp
   paths, pids).
3. **Cache on, cold.** Recreate the Deployment. `dump_all` must show no task
   under this run's namespace (`replay.cache_is_empty`). Abort otherwise.
4. **Cache on, warm.** Normal grading, no audit. Every step's normalized
   output matches cache-off. Every reward matches. Report the grading hit
   rate across the 16 episodes. List each miss as `absent` or `evicted`.
   Eviction may miss; the rate is reported, not required.
5. **Audit, separate.** After the warm pass, one cache-off pass and one warm
   pass. With the cache on, call `bridge.materialize()` and run
   `git status --porcelain`, `git diff`, and a sha256 of tracked and untracked
   files in `/testbed` on that sandbox. The two repo states must match
   (`replay.repo_states_equal`). This pass does not judge grading.

## Scripted case

This one is required.

1. Episode 1 runs edit `E`, then read `R1`, then grading.
2. Episode 2 runs `E` (a hit), then a new read `R2` (a miss), then grading.

Episode 2 must show: `E` returned with no sandbox execution; on `R2`, a
sandbox start, `E` replayed once, then `R2` once; `R2` matches cache-off; the
warm grade is a cache hit with the cache-off reward. `replay.scripted_replay_ok`
returns the failures.

## Metrics

Lookup p50/p95 split by hit and miss. Insert p50/p95. Sandbox startups.
Replayed commands and seconds. Newly executed commands and seconds. Net tool
time saved (cache-off time of the calls that hit, minus replay, extra
startups, and find/insert time), cross-checked against total tool wait.
Duplicate misses by Pod. Grading hit rate. Fallback counts by state. Wall
time from the first start to the 16th finish, and per-episode wall time.

The four-Pod job reports episodes per Pod and fails unless 4 Pods ran
episodes. A live GRPO step with `rollout_replicas: 4`, repeated 3 times per
condition, reads the worker id from `RolloutWorker._stamp_worker_lineage` and
does the same.

## Fault runs

1. Delete the cache Pod during a warm pass. In-flight episodes fall open and
   finish uncached with equivalent outputs. Only an uncertain completion or a
   failed recovery is an `ERROR`. Step telemetry shows the fallback counts.
   After the Deployment is back, episodes reconnect within
   `connect_retry_after_s`.
2. Leave the cache unready. `k8s.sh` times out the readiness wait, logs
   `TVCache not ready after Ns; training starts uncached, rollouts will
   connect when it is ready`, and training starts. Telemetry shows
   `connect_skipped`. Scaling the Deployment back up resumes caching without
   a process restart.

Write one JSON summary per run.

## Single-flight

No duplicate-miss counts exist until the four-replica pass above is run, so
per-process single-flight is not added. `replay.singleflight_decision` returns
`not-yet` for that case. After the run, add per-process single-flight in
`TvcacheBridge` only when duplicate misses are at least half of misses.
De-duplicating across Pods needs a claim operation in tvcache-rust, which
would be a separate proposal to the TVCache authors.
