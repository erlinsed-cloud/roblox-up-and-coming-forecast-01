#!/usr/bin/env python3
"""
Roblox Up-and-Coming chart forecast (make.fun bounty roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (snapshots of Roblox's Up-and-Coming explore chart) and
/input/task.json, and writes /output/forecast.json: one forecast for the whole scored
window (72 hourly reads), scored place by place with a log score.

Approach
--------
The chart ("Top experiences published within the last 28 days sorted by user growth")
moves fast: its #1 changed 13 times in 45 archive reads and the top 10 changed in 42
of 44 consecutive read pairs (median 6 places), so a forecast that simply persists the
closing order decays quickly.  This program:

1. measures each game's rank velocity (weighted least-squares slope of its chart rank
   against time, recent reads weighted more) and its recent player-count growth
   (relative log growth per hour with the chart-wide common growth removed, which
   cancels the daily activity cycle);
2. builds empirical transition tables from the history itself: for every pair of reads
   (i, j) at most ~76 h apart it records where every game ranked in read i was ranked
   at read j, keyed by lag bucket, rank bucket and velocity bucket, with
   hierarchical backoff when a cell is thin;
3. for every hour of the scored window, predicts each candidate game's rank
   distribution from those tables -- interpolating between lag buckets, and blending
   in exact persistence for the first hours of the window, where the chart barely
   moves -- then averages the per-hour distributions over the window;
4. applies the chart's structural rule that only games published within the last 28
   days may appear on it, when the history carries each game's creation date
   (`details`), fading a game out as it reaches that limit;
5. shares each place between candidate games in proportion to those averaged
   probabilities, but deliberately leaves a large part of each place unnamed: the
   unnamed share is spread by the scorer over every chart game not named at that
   place, the only channel that gives probability to a game published after the
   history ends;
6. guesses that the games with the highest probability of holding a top-10 place
   will hold it, ordered by their expected place.

Everything is deterministic (no randomness) and uses only the standard library.
With MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set it runs on a local kit copy:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# tuning constants (fitted by backtesting on the practice history: a forecast
# is made at each read and scored against every later read within 72 h, with
# each 8 h lag bin weighted equally, mirroring the scored window's hourly read
# mix; the chosen values sit on broad plateaus of that backtest)
# ---------------------------------------------------------------------------
VEL_WINDOW_H = 36.0      # use reads within this many hours to estimate velocity
VEL_TAU_H = 34.0         # exponential weight decay for older reads (hours)
VEL_CLIP = 2.5           # max |velocity| in chart places per hour
GROW_WINDOW_H = 26.0     # window for the slow player-growth feature
GROW_RECENT_MIN_H = 3.0  # minimum spacing for the recent growth reference
GROW_SCALE_SLOW = 4.0    # slow growth/h -> equivalent rank velocity, places/h
GROW_SCALE_RECENT = 8.0  # recent growth/h -> equivalent rank velocity, places/h
LAG_BUCKETS = (          # (lo, hi, representative lag in hours)
    (0.0, 2.0, 1.0),
    (2.0, 8.0, 5.0),
    (8.0, 24.0, 16.0),
    (24.0, 48.0, 36.0),
    (48.0, 76.0, 62.0),
)
PERSIST_BLEND = True     # blend towards no movement for the earliest hours
PERSIST_BLEND_LAG = 6.0  # blend the empirical distribution towards exact
                         # persistence below this lag (hours)
PERSIST_EXP = 3.0        # shape of the persistence blend: (lag/LAG)**PERSIST_EXP
RANK_BUCKETS = ((1, 3), (4, 6), (7, 10), (11, 15), (16, 20),
                (21, 30), (31, 45), (46, 60), (61, 78), (79, 97))
V_EDGES = (0.40, 0.15, 0.05, -0.05, -0.15, -0.40)  # descending; + = improving
CELL_PRIOR = 24.0        # backoff strength: parent counts mixed into each cell
CAND_WINDOW_H = 44.0     # a game is a candidate if seen in a read this recent
UNNAMED_EXTRA = 0.0     # margin added to the estimated unknowable share
UNNAMED_FLOOR = 0.26     # minimum share of a place left unnamed
UNNAMED_CAP = 0.50       # maximum share of a place left unnamed
FLATTEN_T = 1.25         # >1 flattens the per-place distribution (calibration)
MIN_ROW_P = 0.01       # rows below this probability are dropped (their mass
                         # goes to the unnamed share, which is spread by the
                         # scorer over the chart games not named)
MAX_ROWS_PER_PLACE = 70
RANK_MAX = 120           # ranks 1..120 are resolved; beyond that is "off chart"
DIST_SIZE = RANK_MAX + 1  # index r-1 for rank r, index RANK_MAX for off chart
AGE_FADE_FROM = 27.0     # a game starts fading out of the chart at this age (days)
AGE_FADE_TO = 29.0       # ... and reaches the floor at this age
AGE_FLOOR = 0.02         # residual probability once past the 28-day chart limit


def parse_ts(text):
    """ISO 8601 UTC timestamp -> seconds since epoch (float)."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()


def rank_bucket(rank):
    for b, (lo, hi) in enumerate(RANK_BUCKETS):
        if lo <= rank <= hi:
            return b
    return len(RANK_BUCKETS) - 1


def v_bucket(v):
    for b, edge in enumerate(V_EDGES):
        if v >= edge:
            return b
    return len(V_EDGES)


def load_reads(path):
    """Parse history.ndjson into normalized reads (oldest first)."""
    reads = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict) or not obj.get("ok", True):
                continue
            rows = obj.get("up_and_coming")
            if not rows:
                continue
            try:
                at = parse_ts(obj.get("at"))
            except (TypeError, ValueError):
                continue
            entries = []
            names = {}
            for row in rows:
                try:
                    uid = int(row.get("universe_id"))
                    rk = int(row.get("rank"))
                except (TypeError, ValueError):
                    continue
                entries.append((rk, uid, row.get("playing")))
                names[uid] = str(row.get("name") or "")
            entries.sort(key=lambda e: (e[0], e[1]))
            rank = {}
            playing = {}
            name = {}
            for rk, uid, pl in entries:
                rank[uid] = rk
                if isinstance(pl, (int, float)):
                    playing[uid] = float(pl)
                name[uid] = names[uid]
            created = {}
            det = obj.get("details")
            if isinstance(det, dict):
                for key, val in det.items():
                    try:
                        uid = int(key)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(val, dict):
                        continue
                    c = val.get("created")
                    if c:
                        try:
                            created[uid] = parse_ts(c)
                        except (TypeError, ValueError):
                            pass
            reads.append({
                "at": at,
                "rank": rank,
                "playing": playing,
                "name": name,
                "created": created,
            })
    reads.sort(key=lambda r: r["at"])
    return reads


def compute_velocity(reads, idx):
    """Weighted least-squares rank slope (places per hour, positive = improving)
    for every game seen within VEL_WINDOW_H before reads[idx]."""
    if idx <= 0:
        return {}
    t0 = reads[idx]["at"]
    samples = {}
    for j in range(idx, -1, -1):
        dt = (t0 - reads[j]["at"]) / 3600.0
        if dt > VEL_WINDOW_H:
            break
        for uid, rk in reads[j]["rank"].items():
            samples.setdefault(uid, []).append((dt, rk))
    out = {}
    for uid, pts in samples.items():
        if len(pts) < 2:
            out[uid] = 0.0
            continue
        sw = sy = sx = sxx = sxy = 0.0
        for dt, rk in pts:
            w = math.exp(-dt / VEL_TAU_H)
            x = -dt
            sw += w
            sy += w * rk
            sx += w * x
            sxx += w * x * x
            sxy += w * x * rk
        den = sw * sxx - sx * sx
        slope = 0.0 if abs(den) < 1e-9 else (sw * sxy - sx * sy) / den
        v = -slope
        if v > VEL_CLIP:
            v = VEL_CLIP
        elif v < -VEL_CLIP:
            v = -VEL_CLIP
        out[uid] = v
    return out


def compute_growth(reads, idx):
    """Relative log player growth per hour, with the chart-wide common growth
    over the same interval removed (which cancels the diurnal activity cycle).
    Positive = growing faster than the chart as a whole.

    Returns (g_recent, g_slow): growth over a recent window 3-12 h back
    (catches launches and spikes, a game whose player count multiplies in hours)
    and a slow window 18-40 h back (the steady trend)."""
    if idx <= 0:
        return {}, {}
    t0 = reads[idx]["at"]
    r_recent = None
    r_slow = None
    for k in range(idx - 1, -1, -1):
        dt = (t0 - reads[k]["at"]) / 3600.0
        if dt > GROW_WINDOW_H + 8.0:
            break
        if r_recent is None and dt >= GROW_RECENT_MIN_H:
            r_recent = k
        if dt <= GROW_WINDOW_H + 8.0:
            r_slow = k
    return _growth_vs(reads, idx, r_recent), _growth_vs(reads, idx, r_slow)


def _growth_vs(reads, idx, base_idx):
    """Day-adjusted log player growth per hour between reads[base_idx] and
    reads[idx]."""
    if base_idx is None or base_idx >= idx:
        return {}
    dt = (reads[idx]["at"] - reads[base_idx]["at"]) / 3600.0
    if dt < 1.0:
        return {}
    base = reads[base_idx]
    logs = []
    for uid, p1 in reads[idx]["playing"].items():
        p0 = base["playing"].get(uid)
        if p0 and p1 and p0 > 0 and p1 > 0:
            logs.append(math.log(p1 / p0))
    if len(logs) < 5:
        return {}
    logs.sort()
    med = logs[len(logs) // 2]
    out = {}
    for uid, p1 in reads[idx]["playing"].items():
        p0 = base["playing"].get(uid)
        if p0 and p1 and p0 > 0 and p1 > 0:
            out[uid] = (math.log(p1 / p0) - med) / dt
    return out


def lag_bucket_index(lag):
    for b, (lo, hi, mid) in enumerate(LAG_BUCKETS):
        if lo <= lag < hi:
            return b
    return len(LAG_BUCKETS) - 1


def build_tables(reads):
    """Sample counts over (lag bucket, rank bucket, velocity bucket) -> future
    rank histogram, plus per (lag, place) unknowable counts."""
    counts = {}
    n_lag = len(LAG_BUCKETS)
    unknown = [[0] * (RANK_MAX + 2) for _ in range(n_lag)]
    place_tot = [[0] * (RANK_MAX + 2) for _ in range(n_lag)]
    lag_tot = [0] * n_lag
    vel_cache = {}
    n_reads = len(reads)
    for i in range(n_reads):
        vel = vel_cache.get(i)
        if vel is None:
            vel = compute_velocity(reads, i)
            vel_cache[i] = vel
        ri = reads[i]
        for j in range(i + 1, n_reads):
            lag = (reads[j]["at"] - ri["at"]) / 3600.0
            if lag >= LAG_BUCKETS[-1][1]:
                break
            lb = lag_bucket_index(lag)
            rj = reads[j]["rank"]
            for uid, rk in rj.items():
                if rk <= 10:
                    place_tot[lb][rk] += 1
                    r0 = ri["rank"].get(uid)
                    if r0 is None or r0 > 60:
                        unknown[lb][rk] += 1
            for uid, rk in ri["rank"].items():
                if rk > RANK_BUCKETS[-1][1]:
                    continue
                rb = rank_bucket(rk)
                vb = v_bucket(vel.get(uid, 0.0))
                fj = rj.get(uid)
                key = (lb, rb, vb)
                arr = counts.get(key)
                if arr is None:
                    arr = counts[key] = [0.0] * DIST_SIZE
                arr[fj - 1 if fj and fj <= RANK_MAX else RANK_MAX] += 1.0
                lag_tot[lb] += 1
    return counts, unknown, place_tot, lag_tot


def cell_dist(counts, lb, rb, vb):
    """Hierarchical backoff: (lb, rb, vb) -> (lb, rb) -> (lb, *) -> flat."""
    size = DIST_SIZE

    def raw(key):
        return counts.get(key)

    own = raw((lb, rb, vb))
    n_own = sum(own) if own else 0.0
    agg1 = [0.0] * size
    n_agg1 = 0.0
    for vb2 in range(len(V_EDGES) + 1):
        if vb2 == vb:
            continue
        c = raw((lb, rb, vb2))
        if not c:
            continue
        for t in range(size):
            if c[t]:
                agg1[t] += c[t]
                n_agg1 += c[t]
    if n_own and n_agg1:
        w = n_own / (n_own + CELL_PRIOR)
        out = [0.0] * size
        for t in range(size):
            out[t] = w * (own[t] / n_own) + (1.0 - w) * (agg1[t] / n_agg1)
        return out
    if n_own:
        return [c / n_own for c in own]
    if n_agg1:
        return [c / n_agg1 for c in agg1]
    # fall back: all rank buckets at this lag
    agg2 = [0.0] * size
    n2 = 0.0
    for rb2 in range(len(RANK_BUCKETS)):
        for vb2 in range(len(V_EDGES) + 1):
            c = raw((lb, rb2, vb2))
            if c:
                for t in range(size):
                    if c[t]:
                        agg2[t] += c[t]
                        n2 += c[t]
    if n2 <= 0.0:
        out = [0.0] * size
        out[RANK_MAX] = 1.0
        return out
    return [c / n2 for c in agg2]


def lag_distribution(counts, lag_h, rb, vb, active_lags, persist_dist=None, cache=None):
    """Rank distribution for a horizon of lag_h hours, interpolating the
    empirical distributions of the lag buckets that hold samples."""
    def cd(lb):
        if cache is None:
            return cell_dist(counts, lb, rb, vb)
        key = (lb, rb, vb)
        d = cache.get(key)
        if d is None:
            d = cell_dist(counts, lb, rb, vb)
            cache[key] = d
        return d

    mids = [LAG_BUCKETS[b][2] for b in active_lags]
    if lag_h <= mids[0]:
        lo_i = hi_i = 0
        w = 0.0
    elif lag_h >= mids[-1]:
        lo_i = hi_i = len(mids) - 1
        w = 0.0
    else:
        lo_i = 0
        while lo_i < len(mids) - 1 and lag_h > mids[lo_i + 1]:
            lo_i += 1
        hi_i = lo_i + 1
        span = mids[hi_i] - mids[lo_i]
        w = (lag_h - mids[lo_i]) / span if span > 0 else 0.0
    if lo_i == hi_i:
        base = cd(active_lags[lo_i])
    else:
        d_lo = cd(active_lags[lo_i])
        d_hi = cd(active_lags[hi_i])
        base = [d_lo[t] * (1.0 - w) + d_hi[t] * w for t in range(DIST_SIZE)]
    blend_lag = max(mids[0], PERSIST_BLEND_LAG)
    if persist_dist is not None and lag_h < blend_lag:
        lam = (lag_h / blend_lag) ** PERSIST_EXP
        base = [base[t] * lam + persist_dist[t] * (1.0 - lam) for t in range(DIST_SIZE)]
    return base


def age_factor(age_days):
    """The chart lists only games published within the last 28 days; fade a game
    out as it approaches (and passes) that limit."""
    if age_days <= AGE_FADE_FROM:
        return 1.0
    if age_days >= AGE_FADE_TO:
        return AGE_FLOOR
    f = 1.0 - (age_days - AGE_FADE_FROM) / (AGE_FADE_TO - AGE_FADE_FROM)
    return AGE_FLOOR + (1.0 - AGE_FLOOR) * f


def build_forecast(reads, task):
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    slots = int(window.get("slots", 72))
    slot_minutes = int(window.get("slot_minutes", 60))
    slot_h = slot_minutes / 60.0

    if not reads:
        return {"probabilities": [], "top10": []}
    try:
        win_from = parse_ts(window.get("from"))
    except (TypeError, ValueError):
        win_from = reads[-1]["at"]

    cutoff = reads[-1]
    t_cut = cutoff["at"]
    cutoff_rank = cutoff["rank"]

    # ---- candidates: games seen in a read within CAND_WINDOW_H of the cutoff ----
    candidates = set()
    created_at = {}
    for rd in reads:
        if (t_cut - rd["at"]) / 3600.0 > CAND_WINDOW_H:
            continue
        candidates.update(rd["rank"].keys())
        for uid, ts in rd["created"].items():
            if uid not in created_at:
                created_at[uid] = ts
    for uid in cutoff_rank:
        candidates.add(uid)

    counts, unknown, place_tot, lag_tot = build_tables(reads)
    active_lags = [b for b in range(len(LAG_BUCKETS)) if lag_tot[b] > 0]
    if not active_lags:
        active_lags = [len(LAG_BUCKETS) - 1]
    vel = compute_velocity(reads, len(reads) - 1)
    grow_recent, grow_slow = compute_growth(reads, len(reads) - 1)

    cand_feat = {}
    for uid in candidates:
        rk = cutoff_rank.get(uid)
        if rk is None or rk > RANK_MAX:
            pos = RANK_MAX
            rb = len(RANK_BUCKETS) - 1
        else:
            pos = rk - 1
            rb = rank_bucket(rk)
        v = vel.get(uid, 0.0)
        g = GROW_SCALE_SLOW * grow_slow.get(uid, 0.0) + \
            GROW_SCALE_RECENT * grow_recent.get(uid, 0.0)
        v_eff = v + g
        if v_eff > VEL_CLIP:
            v_eff = VEL_CLIP
        elif v_eff < -VEL_CLIP:
            v_eff = -VEL_CLIP
        cand_feat[uid] = (rb, v_bucket(v_eff), pos)

    # ---- per-slot distributions, averaged over the window ----
    n_slots = max(1, slots)
    q = {}
    top_mass = {}
    exp_place = {}
    dist_cache = {}
    named_candidates = sorted(candidates)
    for s in range(n_slots):
        t_slot = win_from + s * slot_h * 3600.0
        lag = (t_slot - t_cut) / 3600.0
        if lag < 0.2:
            lag = 0.2
        for uid in named_candidates:
            rb, vb, pos = cand_feat[uid]
            persist = None
            if PERSIST_BLEND and lag < PERSIST_BLEND_LAG:
                persist = [0.0] * DIST_SIZE
                persist[pos] = 1.0
            dist = lag_distribution(counts, lag, rb, vb, active_lags, persist, dist_cache)
            cts = created_at.get(uid)
            if cts is not None:
                age_days = (t_slot - cts) / 86400.0
                f = age_factor(age_days)
                if f < 1.0:
                    dropped = sum(dist[:RANK_MAX]) * (1.0 - f)
                    dist = [d * f for d in dist]
                    dist[RANK_MAX] += dropped
            for k in range(places):
                p = dist[k]
                if p > 0.0:
                    q[(uid, k + 1)] = q.get((uid, k + 1), 0.0) + p
            tot_top = 0.0
            ep = 0.0
            for k in range(places):
                tot_top += dist[k]
                ep += (k + 1) * dist[k]
            top_mass[uid] = top_mass.get(uid, 0.0) + tot_top
            exp_place[uid] = exp_place.get(uid, 0.0) + ep

    # ---- unknowable share per place (hours of the window in each lag bucket) ----
    unk_share = []
    for k in range(1, places + 1):
        num = 0.0
        den = 0.0
        for lb in range(len(LAG_BUCKETS)):
            lo, hi, _ = LAG_BUCKETS[lb]
            t_span = hi - lo
            if t_span <= 0.0:
                continue
            num += t_span * unknown[lb][k]
            den += t_span * place_tot[lb][k]
        unk_share.append(num / den if den > 0 else 0.25)

    # ---- per place: flatten, normalise, leave the unnamed share ----
    probabilities = []
    for k in range(1, places + 1):
        vals = []
        tot = 0.0
        for uid in named_candidates:
            p = q.get((uid, k), 0.0) / n_slots
            if p > 0.0:
                vals.append((uid, p))
                tot += p
        if tot <= 0.0:
            continue
        unnamed = unk_share[k - 1] + UNNAMED_EXTRA
        if unnamed < UNNAMED_FLOOR:
            unnamed = UNNAMED_FLOOR
        elif unnamed > UNNAMED_CAP:
            unnamed = UNNAMED_CAP
        target = 1.0 - unnamed
        flat = []
        ftot = 0.0
        for uid, p in vals:
            fp = p ** (1.0 / FLATTEN_T)
            flat.append((uid, fp))
            ftot += fp
        rows = []
        for uid, fp in flat:
            pp = fp / ftot * target
            if pp >= MIN_ROW_P and pp < 1.0:
                rows.append([uid, pp])
        rows.sort(key=lambda r: (-r[1], r[0]))
        rows = rows[:MAX_ROWS_PER_PLACE]
        game_tot = {}
        for uid, pp in rows:
            game_tot[uid] = game_tot.get(uid, 0.0) + pp
        for uid, pp in rows:
            if game_tot[uid] > 1.0:
                pp *= 1.0 / game_tot[uid]
            probabilities.append({"universe_id": uid, "place": k, "p": round(pp, 6)})

    # ---- top 10: highest top-10 mass, ordered by expected place ----
    ranked = sorted(candidates, key=lambda u: (-top_mass.get(u, 0.0), u))
    top = ranked[:places]
    order = sorted(top, key=lambda u: (
        exp_place.get(u, 0.0) / max(top_mass.get(u, 1e-9), 1e-9),
        cutoff_rank.get(u, 999),
        u,
    ))
    top10 = [int(u) for u in order][:places]
    return {"probabilities": probabilities, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = load_reads(os.path.join(input_dir, "history.ndjson"))
    forecast = build_forecast(reads, task)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    os.makedirs(output_dir, exist_ok=True)
    text = json.dumps(forecast, separators=(",", ":"), ensure_ascii=False)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
