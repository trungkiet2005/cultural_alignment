# Phase 6 — extension experiments (GPU, optional)

Phases 1–4 cover every experiment directly asked for by the reviewer list.
Phase 6 adds **one additional study** that the paper text promises but
didn't yet quantitatively back up: the per-persona floor safeguard
mentioned in §Broader impact.

| Script | What | ETA on H100 |
|---|---|---|
| `exp_r2_persona_floor.py` | Sweeps the minority-protection floor $f \in \{0, 0.5, 1.0, 2.0\}$ on Phi-4 × 3-country panel; tracks MIS, flip rate, and the worst-persona utility so we can show the floor actually protects minorities without wrecking alignment. | ≈2h |

## Controller hook (already committed)

`experiment_DM/exp24_dpbr_core.py` now respects `EXP24_PERSONA_FLOOR`:
when set > 0, each persona's post-correction utility $v(g_i/\sigma)$ is
clamped from below at $-\mathtt{PERSONA\_FLOOR}$ inside the IS pass
(see [experiment_DM/exp24_dpbr_core.py](../../../../experiment_DM/exp24_dpbr_core.py)).
The script mutates the module-level constant between sweep cells, so it
re-uses a single model load across all 4×3 = 12 cells.

## Run

```python
!python exp_paper/review/round2/phase6_extensions/exp_r2_persona_floor.py
```

Override grid:

```python
!R2_FLOOR_GRID=0,0.25,0.5,1.0,2.0 R2_COUNTRIES=USA,JPN,VNM,ETH \
  python exp_paper/review/round2/phase6_extensions/exp_r2_persona_floor.py
```

## Output

`results/exp24_round2/persona_floor/`
- `persona_floor_summary.csv` — one row per (floor, country) with MIS /
  flip rate / mean worst-persona reward / persona spread
- `persona_floor_partial.csv` — streaming checkpoint (safe across crashes)

## What the numbers mean

- `mean_worst_reward` at `floor=0` is the baseline — how deeply the
  worst-performing persona was pushed negative on average per scenario.
  Larger floors should raise this toward zero while MIS barely moves, which
  is the safeguard's claim.
- `mean_persona_spread` = max − min across personas, per scenario, macro
  averaged. A smaller spread at higher floors is the direct signature that
  minority personas are being protected.

Once the CSV lands, `phase5_analysis/aggregate_round2.py` can be extended
to produce a `persona_floor_safeguard.tex` table; the script currently
prints enough info to hand-build it if preferred.
