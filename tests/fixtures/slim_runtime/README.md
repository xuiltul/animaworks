# Slim runtime comparison

`cases.json` contains 24 invented cases with no customer data. Twelve cases
exercise cron output handling; twelve are qualitative acceptance rubrics.
The deterministic sandbox does **not** score the qualitative cases or claim
real-model competence. Human acceptance, correction time and financial cost
remain unmeasured.

Read-only historical aggregation:

```sh
uv run python scripts/slim_runtime_report.py /path/to/runtime --start 2026-09-01 --end 2026-09-07 --output /tmp/baseline.json
```

Docker smoke (requires an existing compatible Linux image; no automatic pull):

```sh
uv run python scripts/slim_runtime_sandbox.py --image python:latest --output /tmp/slim-current
uv run python scripts/slim_runtime_sandbox.py --source /path/to/baseline-worktree --monitor-profile unfiltered --output /tmp/slim-baseline
```

The wrapper mounts source/Python read-only, a scratch output directory writable,
and disables external networking and Linux capabilities. Its local fake HTTP
model is reachable only inside the container. It runs the actual FastAPI
lifespan, worker process, a model-backed cron task, and worker restart. Native
model, avatar and model catalog warmups are replaced in the harness. Twelve
command probes use real subprocesses and the production cron execution
contract, but a direct model proxy; they exclude AgentCore recall/prompt costs.
The current-source smoke additionally publishes tasks through HTTP and the
actual worker watcher: an undeclared attempt ends pending without retry,
a durable wakeup is delivered, explicit resume creates one new attempt,
completion unlocks a dependency whose deterministic Mode A tool call declares
done, and cancellation/restarts do not replay work. A duplicate ready submission
is executed once, a stale attempt finish is rejected, and cancellation interrupts
a deliberately quiet model endpoint before it replies, with heartbeat disabled.
The predecessor completion
is an explicit operator HTTP update, not a claim of model quality. SQLite is
inspected for assertions; execution is not invoked directly by the test.
The worker task exercises AgentCore separately. Model token counts returned by
the fake endpoint are synthetic; reported prompt lengths are characters.

`--monitor-profile quiet` supplies an anchored `skip_pattern`; `unfiltered`
does not. This measures the configuration effect separately from preserving
failed-command follow-up. A reduction in total calls is not the acceptance
criterion: missing error reviews must increase calls. Output contains the
individual cases so the two effects cannot be confused. Runtime fixtures and
logs remain under the supplied output directory; no real runtime is mounted.

The same real-process regression can be run with pytest (skipped by default):

```sh
ANIMAWORKS_RUN_DOCKER_SMOKE=1 uv run pytest tests/e2e/test_slim_runtime_docker.py -q
```
