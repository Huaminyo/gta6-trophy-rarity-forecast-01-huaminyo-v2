[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Huaminyo/gta6-trophy-rarity-forecast-01-huaminyo-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Huaminyo/gta6-trophy-rarity-forecast-01-huaminyo-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast v2 — @huaminyo

Forecasts every Grand Theft Auto VI (PS5) trophy's Exophase earn rate at the
case target as five quantiles, for
[`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250, top 20). One file, Python 3.13 standard library only, deterministic.
v2 of the @huaminyo entry: a refresh-inversion hazard model, written fresh —
not a copy of v1's log-earners fit.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input dir, writes `<id>.out` per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Model

A rate is earners / players tracked, and Exophase only refreshes its counts
every few hours, so consecutive identical reads are ONE refresh state, not
many observations.

**Refresh inversion.** The history collapses into distinct refresh states,
each dated at the midpoint of the gap in which the refresh must have
happened. Between two states, each trophy's earners grow by an increment
over a known exposure (player-hours).

**Hazard model.** The per-player-hour hazard of earning trophy j is
log-linear in log(1 + t): early trophies have a decaying hazard (the pool
of players who have not got them yet shrinks), late trophies a rising one
(players reach them over time). Each hazard line is fit by weighted least
squares over the observed intervals and shrunk toward its tier's slope —
harder when the history is noisy (the shrinkage adapts to the median
leave-one-refresh-out error). The forecast compounds the hazard forward
against the fitted players curve (recency-weighted log-linear in log(1+t))
by numerical integration: E(T) = E_last + ∫ h(t)·P(t) dt. No floor keeps a
rate above its last read: when the player flood outruns the earners, a rate
falls — the live page shows platinum 0.63% → 0.00%.

**Count consistency.** The per-trophy earners are pulled toward a fitted
total trophies earned; the pull's strength adapts to the same noise meter
(weak on clean histories, full on noisy ones).

**Flat history (no refresh observed yet — the practice cases).** The next
event is a refresh, and the kit documents what the launch-night refresh
looked like (players 638 → 3,868, trophies earned 943 → 4,276): a fresh
flood in which every newcomer holds the one 100% trophy and almost nothing
else. The leftover trophy budget splits by chapter-weighted easiness.

**Spreads.** Per-tier trimmed RMS of the leave-one-refresh-out logit
errors (worst 10% dropped — a few near-zero trophies would otherwise set
every band), pooled toward the global RMS; plus each trophy's hazard-slope
standard error carried over the forecast span, plus the fit residual as a
floor. Per-tier multipliers (platinum/gold 1.0, silver 1.4, bronze 1.8)
calibrated on the backtest so the 90% band covers about 90%.

## Local validation (exact pinball scorer from the rules)

- Practice cases vs five reconstructions of the 04:30Z refreshed page:
  mean loss **0.100** (central 0.040, story-heavy 0.109, activity-heavy
  0.105, uniform 0.094, dilution 0.153), coverage 0.90 — vs 1.39–1.69 for
  the house persistence baseline. (v1's server dev score: 0.09043; the
  reconstruction mean is not the same metric.)
- Real-data check (forecast 04:30Z → 15:50Z, scored on the 55 REAL live
  Exophase rates): loss **0.507**, miss 0.12 pp, coverage 0.93 — vs 0.949
  for persistence.
- Synthetic final suite (6 f00-like cases, day-7 truth; dynamics
  calibrated on the live page): mean loss **0.195**, coverage 0.93 —
  clean cases 0.121 / 0.158 / 0.124, noisy-refresh cases 0.262 / 0.274 /
  0.229 — vs 0.34–0.67 for persistence.
- Deterministic (byte-identical across runs), 0.57 s on the final-size
  case (limit 8 s), output 2,789 bytes (limit 65,536).

## What v2 changes vs v1

- Interval hazards + refresh inversion replace the direct log-earners
  OLS (v1's Branch B); the dilution branch is now the hazard integral
  itself, and the "not below last rate" floor is gone.
- Per-tier sigma from trimmed leave-one-refresh-out errors replaces the
  horizon + residual + Poisson spread; multipliers calibrated to
  coverage ≈ 0.9.
- A level-based log-E blend and a concave-quadratic players fit were
  tried on the backtest and reverted (both hurt the synthetic finals).
