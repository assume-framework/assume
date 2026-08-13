# Archive — full-length working docs

**Do not read these files by default. They cost ~35 000 tokens together and
will eat a fresh session's context before any work starts.**

| file | lines | what it is |
|---|---:|---|
| `HANDOFF_full.md` | 764 | the long handoff as it stood after run 13 (2026-08-11) |
| `RUNS_full.md` | 1844 | the long run log, runs 01–13, every number and every correction |

The working docs are one directory up and are what you should read:

* [`../HANDOFF.md`](../HANDOFF.md) — the summary, ~2 pages
* [`../RUNS.md`](../RUNS.md) — runs 01–13 condensed, ~9 pages
* [`../RUNS_Continuation.md`](../RUNS_Continuation.md) — **new runs go here**, incl. the cluster

Nothing was deleted in the shortening: every number, table and caveat that the
short versions drop is still here verbatim. The short versions are lossy on
*prose*, not on results.

## When to open one of these

Only when the short version has failed you, and then **grep, don't read**:

```bash
cd examples/inputs/2_nodes_paper_small/rl_benchmark/archive
grep -n "act-x30"      RUNS_full.md       # a specific number
grep -n "^### 11"      RUNS_full.md       # a run's full section, then read that range
sed  -n '906,1125p'    RUNS_full.md       # run 12, in full
```

Concretely, the things that only exist here:

- the per-phase narrative of runs 06 and 08 (why the pooled path metric was wrong)
- run 01b's per-seed crossing table and the `policy_delay` derivation
- run 12's three offline `act_share` rounds, in full, with the `obs`-scaling ladder
- run 13's `act-all` vs `act-own` block-share control, in full
- the long form of corrections 5–7 and 11–17, each with the evidence that overturned it
- the recipe prose for recreating `single_10ep_standard.npz`

**Do not edit these files.** They are a snapshot. Corrections go in
`../RUNS.md`; new work goes in `../RUNS_Continuation.md`.
