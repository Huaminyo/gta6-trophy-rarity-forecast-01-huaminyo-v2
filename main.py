"""GTA VI trophy rarity forecast, v2 (refresh-inversion hazard model).

Forecasts every trophy's Exophase earn rate at the case target as five
quantiles. Python 3.13 standard library only, deterministic.

Model (written from the competition rules; see README.md for the full
write-up):

  A rate is earners / players tracked, and Exophase only refreshes its
  counts every few hours, so consecutive identical reads are ONE refresh
  state, not many observations. The model first inverts the refresh
  schedule: it collapses the history into distinct refresh states and
  dates each state at the midpoint of the gap in which the refresh must
  have happened.

  Between two refresh states, each trophy's earners grow by an increment
  over a known exposure (player-hours). The per-player-hour hazard of
  earning trophy j is modelled as log-linear in log(1 + t): early trophies
  have a decaying hazard (the pool of players who have not got them yet
  shrinks), late trophies a rising one (players reach them over time).
  Each trophy's hazard line is fit by weighted least squares over the
  observed intervals and shrunk toward its tier's slope. The forecast
  compounds the hazard forward against the fitted players curve by
  numerical integration: E(T) = E_last + integral of h(t) * P(t) dt.
  Players tracked follow a recency-weighted log-linear fit in log(1+t).
  No floor keeps a
  rate above its last read: when the player flood outruns the earners,
  a rate falls, as the live page shows (platinum 0.63% -> 0.00%).

  With a flat history (no refresh observed yet, the practice cases) the
  model falls back on the refresh the kit documents for launch night
  (players 638 -> 3,868, trophies earned 943 -> 4,276): a fresh flood in
  which every newcomer holds the one 100% trophy and almost nothing else.

  Spreads come from a leave-one-refresh-out refit (per-tier RMS of the
  one-refresh-ahead logit error, pooled toward the global RMS), the
  standard error of each trophy's hazard slope carried over the forecast
  span, and the fit residual as a floor; per-tier multipliers are
  calibrated on the backtest so the 90% band covers about 90%.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- calibrated constants (see README.md; tuned on the local backtest) ---
# Per-tier spread multipliers: bronze needs the widest band (its trophies
# span early-story to deep-grind, so one-refresh-ahead errors vary most);
# platinum/gold are single-trophy tiers with stable behaviour.
SIGMA_TIER = {"platinum": 1.0, "gold": 1.0, "silver": 1.4, "bronze": 1.8}
SIGMA_MIN = 0.08          # narrowest band any trophy gets
SIGMA_MAX = 0.90          # widest band any trophy gets
SIGMA_FLAT = 0.20         # flat-history (refresh-prior) branch (swept 0.15-0.30; 0.20 minimises the practice loss at coverage 0.90)
SIGMA_FULL = 0.05         # a trophy already at ~100% barely moves
SHRINK_K = 3.0            # pseudo-intervals pulling a slope to its tier mean
POOL_K = 5.0              # pseudo-trophies pulling a tier RMS to the global RMS
RESCALE_LO = 0.5          # count-consistency rescale is clamped to [0.5, 1.5]
RESCALE_HI = 1.5
# The launch-night refresh the kit documents (RULES.txt): players 638 ->
# 3,868 and trophies earned 943 -> 4,276. The flat branch's prior.
REFRESH_P_RATIO = 3868.0 / 638.0
REFRESH_E_RATIO = 4276.0 / 943.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError("bad timestamp %r" % text)


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


# --- trophy classification (for the flat branch's budget weights) ----------

def classify(trophy):
    """Return (kind, chapter): kind in story/mission/activity/grind/
    completion; chapter is the story chapter (1-4, 99 = finale) or 0."""
    desc = trophy["description"]
    if trophy["id"] == 1 or "100% Completion" in desc:
        return ("completion", 99)
    if "final mission" in desc.lower():
        return ("story", 99)
    for ch in (1, 2, 3, 4):
        if "Chapter %d" % ch in desc:
            return ("story", ch)
    if any(k in desc for k in ("250 ", "100 ", "20 different",
                               "8 mission", "3 Collectible",
                               "50 Challenges", "100 locations")):
        return ("grind", 0)
    if desc.startswith("Complete "):
        return ("mission", 0)
    if "Meditate" in desc:
        return ("grind", 0)
    return ("activity", 0)


def budget_weight(trophy):
    """Share of the leftover-trophy budget a trophy plausibly takes in a
    fresh-flood refresh: early story first, grinds last."""
    kind, ch = classify(trophy)
    if kind == "story":
        return {1: 4.0, 2: 2.5, 3: 1.5, 4: 1.0, 99: 0.5}[ch]
    if kind == "mission":
        return 0.8
    if kind == "activity":
        return 0.4
    if kind == "grind":
        return 0.15
    return 0.05


# --- refresh inversion ------------------------------------------------------

def collapse_snapshots(case, origin):
    """Collapse the read history into distinct refresh states.

    A new state starts when the players count or any rate changes. Each
    state is dated at the midpoint of the gap between the last read of
    the previous state and the first read showing this one: the refresh
    happened somewhere in that gap, and the midpoint is the unbiased
    estimate. Returns a list of dicts {t, P, E, total} with t in hours
    since release and E the per-trophy earners (None where the read
    lacked the trophy)."""
    reads = case["history"]
    n = len(case["trophies"])
    snaps = []
    prev_key = None
    prev_at = None
    for r in reads:
        key = (r["players_tracked"], tuple(r["rates"]))
        if key == prev_key:
            continue
        at = parse_time(r["at"])
        if snaps:
            # Refresh fell between the previous read and this one.
            t_refresh = prev_at + (at - prev_at) / 2
        else:
            t_refresh = at
        rates = r["rates"]
        p = r["players_tracked"]
        snaps.append({
            "t": hours_between(origin, t_refresh),
            "P": float(p),
            "E": [None if v is None else v * p / 100.0 for v in rates],
            "total": float(r["trophies_earned"]),
        })
        prev_key = key
        prev_at = at
    return snaps


# --- weighted least squares -------------------------------------------------

def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, se_b)
    or None when the fit is degenerate (fewer than 2 points or no spread
    in x). resid_sd is the weighted residual SD; se_b the slope's
    standard error."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(1, len(xs) - 2)
    resid_sd = math.sqrt(sse / sw) if sw > 0 else 0.0
    se_b = math.sqrt((sse / dof) / sxx) if sxx > 0 else 0.0
    return (a, b, resid_sd, se_b)


def ols(xs, ys):
    """Unweighted least squares; returns (a, b) or None."""
    fit = wls(xs, ys, [1.0] * len(xs))
    if fit is None:
        return None
    return (fit[0], fit[1])


# --- hazard model -----------------------------------------------------------

def interval_hazards(snaps, j):
    """Per-interval hazard observations for trophy j: a list of
    (x, y, w) with x = ln(1 + interval-midpoint hours), y = ln(hazard
    per player-hour), w = earners gained + 0.5 (intervals that moved the
    trophy carry more information). Intervals touching a missing rate
    are skipped; non-positive increments (quantisation noise) become
    half an earner."""
    obs = []
    for i in range(len(snaps) - 1):
        s0, s1 = snaps[i], snaps[i + 1]
        e0, e1 = s0["E"][j], s1["E"][j]
        if e0 is None or e1 is None:
            continue
        dt = s1["t"] - s0["t"]
        if dt <= 0:
            continue
        exposure = 0.5 * (s0["P"] + s1["P"]) * dt
        if exposure <= 0:
            continue
        gained = max(0.0, e1 - e0)
        haz = max(gained, 0.5) / exposure
        t_mid = 0.5 * (s0["t"] + s1["t"])
        # The first interval's exposure is mismeasured: the launch-night
        # cohort had already played for hours before the first read, so
        # its earners were not gained inside this interval. Down-weight it.
        w = (gained + 0.5) * (0.25 if i == 0 else 1.0)
        obs.append((math.log1p(max(0.0, t_mid)), math.log(haz), w))
    return obs


def fit_hazards(snaps, trophies, shrink_k=SHRINK_K):
    """Fit every trophy's hazard line log h = a + b*ln(1+t), shrinking
    each slope toward its tier's mean slope. Returns a list of dicts
    {a, b, resid, se_b, n} per trophy, plus the tier mean slopes."""
    n = len(trophies)
    raw = []
    for j in range(n):
        obs = interval_hazards(snaps, j)
        if len(obs) >= 2:
            fit = wls([o[0] for o in obs], [o[1] for o in obs],
                      [o[2] for o in obs])
        else:
            fit = None
        raw.append((obs, fit))
    # Tier mean slopes from trophies with a real fit, pooled toward the
    # global mean slope so thin tiers (platinum: 1 trophy) stay sane.
    slopes = [f[1] for _, f in raw if f is not None]
    global_b = sum(slopes) / len(slopes) if slopes else 0.0
    tier_b = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        ts = [raw[j][1][1] for j in range(n)
              if trophies[j]["tier"] == tier and raw[j][1] is not None]
        tier_b[tier] = ((sum(ts) + 2.0 * global_b) / (len(ts) + 2.0)
                        if ts else global_b)
    out = []
    for j in range(n):
        obs, fit = raw[j]
        tier = trophies[j]["tier"]
        if fit is not None:
            a, b, resid, se_b = fit
            nj = len(obs)
            b2 = (nj * b + shrink_k * tier_b[tier]) / (nj + shrink_k)
            # Re-anchor the intercept at the shrunk slope.
            sw = sum(o[2] for o in obs)
            mx = sum(o[2] * o[0] for o in obs) / sw
            my = sum(o[2] * o[1] for o in obs) / sw
            a2 = my - b2 * mx
            out.append({"a": a2, "b": b2, "resid": resid, "se_b": se_b,
                        "n": nj})
        elif obs:
            # One interval: level from it, slope from the tier.
            x, y, _ = obs[0]
            out.append({"a": y - tier_b[tier] * x, "b": tier_b[tier],
                        "resid": 0.3, "se_b": 0.5, "n": 1})
        else:
            # No usable interval: a near-zero hazard at the tier slope.
            out.append({"a": math.log(0.5 / 1e6), "b": tier_b[tier],
                        "resid": 0.5, "se_b": 0.8, "n": 0})
    return out, tier_b


def fit_players(snaps):
    """Players tracked: log P ~ a + b*ln(1+t) over the refresh states,
    weighted toward recent states (weight halves every 24h), since
    launch growth is faster than the current regime. Returns the
    (a, b, 0.0) coefficients (the integral takes the quadratic form;
    a concave quadratic was tried and dropped — it overfits refresh
    noise in the players series)."""
    t_last = snaps[-1]["t"]
    us = [math.log1p(max(0.0, s["t"])) for s in snaps]
    ys = [math.log(max(1.0, s["P"])) for s in snaps]
    ws = [0.5 ** ((t_last - s["t"]) / 24.0) for s in snaps]
    lin = wls(us, ys, ws)
    if lin is None:
        return (math.log(max(1.0, snaps[-1]["P"])), 0.0, 0.0)
    return (lin[0], min(2.0, max(0.0, lin[1])), 0.0)


def players_at(quad, t_hours):
    """Players tracked at t hours since release under the players fit."""
    a, b, c = quad
    u = math.log1p(max(0.0, t_hours))
    return math.exp(a + b * u + c * u * u)


def hazard_integral(a, b, quad, t_from, t_to):
    """Integral of h(t)*P(t) dt from t_from to t_to (hours since
    release), with h(t) = exp(a)*(1+t)^b and P(t) the quadratic players
    fit. Computed by Simpson's rule in u = ln(1+t) space, where the
    integrand is a smooth exponential: exp((a+A) + (b+B+1)*u + C*u^2),
    dt = e^u du. Deterministic, 200 steps."""
    aq, bq, cq = quad
    u0 = math.log1p(max(0.0, t_from))
    u1 = math.log1p(max(0.0, t_to))
    if u1 <= u0:
        return 0.0
    n = 200
    h = (u1 - u0) / n
    total = 0.0
    for i in range(n + 1):
        u = u0 + i * h
        z = (a + aq) + (b + bq + 1.0) * u + cq * u * u
        # Clamp the exponent: the integrand is bounded in practice, and
        # a wild fit must not overflow the forecast.
        z = max(-50.0, min(50.0, z))
        w = 1.0 if i in (0, n) else (4.0 if i % 2 == 1 else 2.0)
        total += w * math.exp(z)
    return total * h / 3.0


def last_earners(snaps, j):
    """Most recent non-missing earners for trophy j (walks back over
    snapshots), or 0.5 when never observed."""
    for s in reversed(snaps):
        if s["E"][j] is not None:
            return s["E"][j]
    return 0.5


def forecast_earners(snaps, trophies, t_target, hazard_fits, players_fit,
                     rescale_w):
    """Median earners at the target for every trophy, via the hazard
    integral from the last refresh state, then pulled toward the fitted
    total trophies earned (count consistency). The pull's strength is
    rescale_w (0.25..1), set by the caller from the median LOSO error:
    when one-refresh-ahead predictions are unreliable the aggregate
    anchor is worth more. (A blend with a level-based log-E fit was
    tried and reverted: it hurt every synthetic-final case and made the
    2-snapshot case unstable.)"""
    t_last = snaps[-1]["t"]
    p_target = max(players_at(players_fit, t_target), snaps[-1]["P"])
    earners = []
    for j in range(len(trophies)):
        e_last = last_earners(snaps, j)
        f = hazard_fits[j]
        inc = hazard_integral(f["a"], f["b"], players_fit, t_last, t_target)
        e = e_last + max(0.0, inc)
        earners.append(min(e, p_target))
    xs = [math.log1p(max(0.0, s["t"])) for s in snaps]
    ys = [math.log(max(1.0, s["total"])) for s in snaps]
    fit = ols(xs, ys)
    if fit is not None and sum(earners) > 0:
        total_t = math.exp(fit[0] + fit[1] * math.log1p(max(0.0, t_target)))
        factor = total_t / sum(earners)
        factor = 1.0 + rescale_w * (factor - 1.0)
        factor = min(RESCALE_HI, max(RESCALE_LO, factor))
        earners = [min(e * factor, p_target) for e in earners]
    return earners, p_target


# --- spread: leave-one-refresh-out calibration ------------------------------

def loso_errors(snaps, trophies):
    """One-refresh-ahead logit errors: for each snapshot k >= 3, refit
    the hazards on the other snapshots and predict k's rates from k-1's
    state. Returns a list of (tier, logit_error)."""
    errs = []
    if len(snaps) < 4:
        return errs
    for k in range(3, len(snaps)):
        sub = snaps[:k] + snaps[k + 1:]
        # Refit on the reduced set, predict snapshot k from state k-1.
        fits, _ = fit_hazards(sub, trophies)
        pf = fit_players(sub)
        # Predict earners at snapshot k's time, starting from k-1.
        base = snaps[k - 1]
        for j, t in enumerate(trophies):
            y = snaps[k]["E"][j]
            if y is None or base["E"][j] is None:
                continue
            f = fits[j]
            inc = hazard_integral(f["a"], f["b"], pf,
                                  base["t"], snaps[k]["t"])
            e_pred = base["E"][j] + max(0.0, inc)
            p_pred = max(players_at(pf, snaps[k]["t"]), base["P"])
            r_pred = 100.0 * min(e_pred, p_pred) / p_pred
            r_true = 100.0 * y / snaps[k]["P"]
            errs.append((t["tier"], logit(r_true) - logit(r_pred)))
    return errs


def tier_sigma(errs):
    """Per-tier one-step-ahead scale: the TRIMMED RMS of the LOSO logit
    errors (worst 10% dropped per tier), pooled toward the global
    trimmed RMS (POOL_K pseudo-trophies). Trimming matters: a few
    near-zero trophies have catastrophic logit errors that would
    otherwise set every trophy's band. Returns {tier: sigma}."""
    def trimmed_rms(sq):
        if not sq:
            return None
        sq = sorted(sq)
        keep = sq[:max(1, int(len(sq) * 0.9))]
        return math.sqrt(sum(keep) / len(keep))
    all_sq = [e * e for _, e in errs]
    glob = trimmed_rms(all_sq)
    if glob is None:
        glob = 0.30
    out = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        sq = [e * e for ty, e in errs if ty == tier]
        raw = trimmed_rms(sq)
        if raw is None:
            out[tier] = glob
        else:
            out[tier] = math.sqrt((len(sq) * raw * raw + POOL_K * glob * glob)
                                  / (len(sq) + POOL_K))
    return out, glob


# --- flat-history branch (refresh prior) ------------------------------------

def flat_branch(case, snaps, trophies, t_target):
    """No refresh observed yet: the next event is a refresh, and the
    kit documents what the launch-night refresh looked like. Players
    scale by the documented ratio; every newcomer holds the one 100%
    trophy; the leftover trophy budget splits by budget weight."""
    last = snaps[-1]
    p0 = last["P"]
    p_new = p0 * REFRESH_P_RATIO
    newcomers = p_new - p0
    new_total = last["total"] * (REFRESH_E_RATIO - 1.0)
    budget = max(0.0, new_total - newcomers)
    # The 100% trophy: the one with the highest last rate.
    rates = case["history"][-1]["rates"]
    full_j = max(range(len(trophies)),
                 key=lambda j: rates[j] if rates[j] is not None else -1)
    weights = []
    for j, t in enumerate(trophies):
        if j == full_j:
            weights.append(0.0)
        else:
            e0 = last["E"][j] if last["E"][j] is not None else 0.0
            weights.append(budget_weight(t) * math.sqrt(e0 + 1.0))
    wsum = sum(weights)
    medians = []
    for j in range(len(trophies)):
        e0 = last["E"][j] if last["E"][j] is not None else 0.0
        if j == full_j:
            e_new = e0 + newcomers
        elif wsum > 0:
            e_new = e0 + budget * weights[j] / wsum
        else:
            e_new = e0
        medians.append(100.0 * min(e_new, p_new) / p_new)
    return medians, full_j


# --- main forecast ----------------------------------------------------------

def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not %s" % LEVELS)
    if not case["history"]:
        raise ValueError("case has no history")
    origin = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    t_target = hours_between(origin, target)
    trophies = case["trophies"]
    snaps = collapse_snapshots(case, origin)

    if len(snaps) < 2:
        medians, full_j = flat_branch(case, snaps, trophies, t_target)
        sigmas = [SIGMA_FULL if j == full_j else SIGMA_FLAT
                  for j in range(len(trophies))]
    else:
        fits, _ = fit_hazards(snaps, trophies)
        pf = fit_players(snaps)
        errs = loso_errors(snaps, trophies)
        # How much to trust the per-trophy integrals: the median
        # one-refresh-ahead error. Clean histories (~0.05) get the
        # weakest pull toward the total; noisy ones (~0.15) the full
        # pull. No LOSO errors (too few snapshots) -> full pull.
        if errs:
            med = sorted(abs(e) for _, e in errs)[len(errs) // 2]
            rescale_w = min(1.0, max(0.25, (med - 0.05) / 0.10))
            # Noisy histories: per-trophy slopes are noise-dominated, so
            # shrink them harder toward the tier mean before forecasting.
            noise = min(1.0, max(0.0, (med - 0.05) / 0.10))
            shrink_k = SHRINK_K + 7.0 * noise
            if shrink_k > SHRINK_K + 0.5:
                fits, _ = fit_hazards(snaps, trophies, shrink_k=shrink_k)
        else:
            rescale_w = 1.0
        earners, p_target = forecast_earners(snaps, trophies, t_target,
                                             fits, pf, rescale_w)
        medians = [100.0 * e / p_target for e in earners]
        sig_tier, _ = tier_sigma(errs)
        span = math.log1p(max(0.0, t_target)) - math.log1p(max(0.0, snaps[-1]["t"]))
        sigmas = []
        for j, t in enumerate(trophies):
            f = fits[j]
            # Trend uncertainty: the slope's standard error carried over
            # the forecast span, in log-hazard units.
            trend = f["se_b"] * max(0.0, span)
            # Level uncertainty: the fit residual, averaged over intervals.
            level = f["resid"] / math.sqrt(max(1, f["n"]))
            var = sig_tier[t["tier"]] ** 2 + trend ** 2 + level ** 2
            s = SIGMA_TIER.get(t["tier"], 1.4) * math.sqrt(var)
            sigmas.append(min(SIGMA_MAX, max(SIGMA_MIN, s)))
        # A trophy already at ~100% stays there: narrow band, median 99.99.
        last_rates = case["history"][-1]["rates"]
        for j in range(len(trophies)):
            r = last_rates[j]
            if r is not None and r >= 99.9:
                medians[j] = 99.99
                sigmas[j] = SIGMA_FULL

    rows = []
    for j, t in enumerate(trophies):
        lg = logit(medians[j])
        s = sigmas[j]
        qs = [round4(inv_logit_pct(lg + z * s)) for z in Z]
        # Keep the quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / ("%s.out" % path.stem)).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
