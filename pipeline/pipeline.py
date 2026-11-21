#!/usr/bin/env python3
"""
Shopping Sessions: High-speed Polars implementation with exact rule parity.
"""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import glob
import json
import os
import re

NCPU = os.environ.get("NCPU")
if NCPU:
    os.environ.setdefault("POLARS_MAX_THREADS", NCPU)

import polars as pl  # noqa: E402

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")
SCRATCH = os.environ.get("SCRATCH_DIR", "/scratch")

MS_PER_SEC = 1_000
MS_PER_MIN = 60_000
MS_PER_HOUR = 3_600_000
MS_PER_DAY = 86_400_000
SESSION_GAP_MS = 30 * MS_PER_MIN
LATE_LIMIT_MS = 24 * MS_PER_HOUR
LOOKBACK_7D_MS = 7 * MS_PER_DAY

ASCII_DIGITS_RE = re.compile(r"^[0-9]+$")
F1_RE = re.compile(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z$")
F2_RE = re.compile(
    r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})([+-])([0-9]{2}):([0-9]{2})$"
)
VALID_EVENT_TYPES = {"view", "cart", "purchase"}

USER_DAILY = {
    "user_id": pl.String,
    "day": pl.Date,
    "sessions": pl.Int64,
    "events": pl.Int64,
    "revenue_cents": pl.Int64,
    "top_category": pl.String,
}
CAMPAIGN_DAILY = {
    "campaign_id": pl.String,
    "channel": pl.String,
    "day": pl.Date,
    "sessions": pl.Int64,
    "purchasing_sessions": pl.Int64,
    "revenue_cents": pl.Int64,
    "assisted_sessions": pl.Int64,
}


def parse_csv_line(line: str, expected_cols: int) -> list[str] | None:
    if line == "":
        return None
    raw_fields = line.split(",")
    if len(raw_fields) != expected_cols:
        return None
    out: list[str] = []
    for raw in raw_fields:
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            inner = raw[1:-1]
            if '"' in inner:
                return None
            out.append(inner)
        else:
            if '"' in raw:
                return None
            out.append(raw)
    return out


def iter_csv_rows(path: str, expected_cols: int):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if not text:
        return
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) <= 1:
        return
    for line in lines[1:]:
        parsed = parse_csv_line(line, expected_cols)
        if parsed is not None:
            yield parsed


def parse_ts_ms(val: str) -> int | None:
    if not val:
        return None
    if ASCII_DIGITS_RE.match(val):
        try:
            ms = int(val)
        except ValueError:
            return None
        if ms < -(1 << 63) or ms > (1 << 63) - 1:
            return None
        return ms

    m1 = F1_RE.match(val)
    if m1:
        y, mo, d, h, mi, s = (int(x) for x in m1.groups())
        try:
            dt = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
        except ValueError:
            return None
        epoch_sec = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)) // timedelta(seconds=1)
        return epoch_sec * MS_PER_SEC

    m2 = F2_RE.match(val)
    if m2:
        y, mo, d, h, mi, s = (int(x) for x in m2.groups()[:6])
        sign, off_h_s, off_m_s = m2.group(7), m2.group(8), m2.group(9)
        off_h = int(off_h_s)
        off_m = int(off_m_s)
        if off_h > 23 or off_m > 59:
            return None
        try:
            dt_local_as_utc = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
        except ValueError:
            return None
        offset = timedelta(hours=off_h, minutes=off_m)
        dt_utc = dt_local_as_utc - offset if sign == "+" else dt_local_as_utc + offset
        epoch_sec = (dt_utc - datetime(1970, 1, 1, tzinfo=timezone.utc)) // timedelta(seconds=1)
        return epoch_sec * MS_PER_SEC

    return None


def parse_price_cents(val: str) -> int | None:
    if not val or not ASCII_DIGITS_RE.match(val):
        return None
    price = int(val)
    if 1 <= price <= 10_000_000:
        return price
    return None


def build_person_lookup(link_files: list[str]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra.encode("utf-8") < rb.encode("utf-8"):
                parent[rb] = ra
            else:
                parent[ra] = rb

    for lf in link_files:
        for uid_a, uid_b in iter_csv_rows(lf, 2):
            if uid_a != "" and uid_b != "":
                union(uid_a, uid_b)

    comp_min_bytes: dict[str, bytes] = {}
    comp_min_str: dict[str, str] = {}
    for uid in parent:
        r = find(uid)
        b = uid.encode("utf-8")
        if r not in comp_min_bytes or b < comp_min_bytes[r]:
            comp_min_bytes[r] = b
            comp_min_str[r] = uid

    return {uid: comp_min_str[find(uid)] for uid in parent}


def main():
    with open(os.path.join(DATA, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    win_start_ms = parse_ts_ms(manifest["window_start"])
    win_end_ms = parse_ts_ms(manifest["window_end"])
    assert win_start_ms is not None and win_end_ms is not None
    click_start_ms = win_start_ms - LOOKBACK_7D_MS

    products: dict[str, str] = {}
    for pid, cat in iter_csv_rows(os.path.join(DATA, "products.csv"), 2):
        products[pid] = cat

    campaigns: dict[str, str] = {}
    for cid, ch in iter_csv_rows(os.path.join(DATA, "campaigns.csv"), 2):
        campaigns[cid] = ch

    event_files = sorted(glob.glob(os.path.join(DATA, "events", "*.csv")))
    click_files = sorted(glob.glob(os.path.join(DATA, "clicks", "*.csv")))
    link_files = sorted(glob.glob(os.path.join(DATA, "links", "*.csv")))

    person_map = build_person_lookup(link_files)

    def resolve_person(uid: str) -> str:
        return person_map.get(uid, uid)

    # 1. Events: parse -> dedup on event_id (smallest ingest_ts, before validation)
    #    -> validate -> drop late (ingest_ts - ts > 24h)
    kept_by_event_id: dict[str, tuple[int, list[str]]] = {}
    for ef in event_files:
        for row in iter_csv_rows(ef, 7):
            event_id, _uid, _ts, ingest_ts_raw, _etype, _pid, _price = row
            if event_id == "":
                continue
            ingest_ms = parse_ts_ms(ingest_ts_raw)
            if ingest_ms is None:
                continue
            prev = kept_by_event_id.get(event_id)
            if prev is None or ingest_ms < prev[0]:
                kept_by_event_id[event_id] = (ingest_ms, row)

    valid_events = []
    for _event_id, (ingest_ms, row) in kept_by_event_id.items():
        _ev_id, uid, ts_raw, _ingest_raw, etype, pid, price_raw = row
        if uid == "" or pid == "" or etype not in VALID_EVENT_TYPES:
            continue
        ts_ms = parse_ts_ms(ts_raw)
        if ts_ms is None or not (win_start_ms <= ts_ms < win_end_ms):
            continue

        is_purchase = etype == "purchase"
        price_cents = 0
        if is_purchase:
            p = parse_price_cents(price_raw)
            if p is None:
                continue
            price_cents = p

        if (ingest_ms - ts_ms) > LATE_LIMIT_MS:
            continue

        valid_events.append({
            "person": resolve_person(uid),
            "ts_ms": ts_ms,
            "category": products.get(pid, "unknown"),
            "is_purchase": is_purchase,
            "rev_cents": price_cents,
        })

    # 2. Clicks: parse -> dedup on click_id -> validate
    seen_click_ids: set[str] = set()
    valid_clicks = []
    for cf in click_files:
        for click_id, uid, cid, ts_raw in iter_csv_rows(cf, 4):
            if click_id == "" or click_id in seen_click_ids:
                continue
            seen_click_ids.add(click_id)

            if uid == "" or cid not in campaigns:
                continue
            ts_ms = parse_ts_ms(ts_raw)
            if ts_ms is None or not (click_start_ms <= ts_ms < win_end_ms):
                continue

            valid_clicks.append({
                "person": resolve_person(uid),
                "ts_ms": ts_ms,
                "click_id": click_id,
                "campaign_id": cid,
            })

    ev_schema = {"person": pl.String, "ts_ms": pl.Int64, "category": pl.String, "is_purchase": pl.Boolean, "rev_cents": pl.Int64}
    cl_schema = {"person": pl.String, "ts_ms": pl.Int64, "click_id": pl.String, "campaign_id": pl.String}

    ev_df = pl.DataFrame(valid_events, schema=ev_schema) if valid_events else pl.DataFrame(schema=ev_schema)
    cl_df = pl.DataFrame(valid_clicks, schema=cl_schema) if valid_clicks else pl.DataFrame(schema=cl_schema)

    # 3. Sessionize per person in Polars
    if ev_df.height > 0:
        ev_sorted = ev_df.sort(["person", "ts_ms"])
        prev_ts = pl.col("ts_ms").shift(1).over("person")
        distinct_ts = (
            pl.when(prev_ts.is_null() | (pl.col("ts_ms") != prev_ts))
            .then(pl.col("ts_ms"))
            .otherwise(None)
            .forward_fill()
            .over("person")
        )
        prev_distinct = distinct_ts.shift(1).over("person")
        new_sess_flag = (
            prev_distinct.is_null()
            | ((distinct_ts != prev_distinct) & ((distinct_ts - prev_distinct) > SESSION_GAP_MS))
        ).cast(pl.Int64)
        ev_sess = ev_sorted.with_columns(new_sess_flag.cum_sum().over("person").alias("sess_num"))

        sessions_df = (
            ev_sess.group_by(["person", "sess_num"])
            .agg([
                pl.col("ts_ms").min().alias("session_start_ts"),
                pl.len().cast(pl.Int64).alias("s_events"),
                pl.col("is_purchase").cast(pl.Int64).sum().alias("s_purchases"),
                pl.col("rev_cents").sum().alias("s_revenue"),
            ])
            .with_columns([
                (pl.col("session_start_ts") // MS_PER_DAY)
                .cast(pl.Int32)
                .cast(pl.Date)
                .alias("session_day"),
            ])
        )

        cat_rev_df = (
            ev_sess.join(sessions_df.select(["person", "sess_num", "session_day"]), on=["person", "sess_num"], how="inner")
            .filter(pl.col("is_purchase") & (pl.col("rev_cents") > 0))
            .group_by(["person", "session_day", "category"])
            .agg(pl.col("rev_cents").sum().alias("cat_rev"))
        )
        if cat_rev_df.height > 0:
            top_cat_map: dict[tuple[str, object], str] = {}
            best_score: dict[tuple[str, object], tuple[int, bytes]] = {}
            for row in cat_rev_df.iter_rows(named=True):
                key = (row["person"], row["session_day"])
                score = (-int(row["cat_rev"]), row["category"].encode("utf-8"))
                if key not in best_score or score < best_score[key]:
                    best_score[key] = score
                    top_cat_map[key] = row["category"]
            top_cat_df = pl.DataFrame(
                [{"person": k[0], "session_day": k[1], "top_category": v} for k, v in top_cat_map.items()],
                schema={"person": pl.String, "session_day": pl.Date, "top_category": pl.String},
            )
        else:
            top_cat_df = pl.DataFrame(schema={"person": pl.String, "session_day": pl.Date, "top_category": pl.String})

        user_df = (
            sessions_df.group_by(["person", "session_day"])
            .agg([
                pl.len().cast(pl.Int64).alias("sessions"),
                pl.col("s_events").sum().alias("events"),
                pl.col("s_revenue").sum().alias("revenue_cents"),
            ])
            .join(top_cat_df, on=["person", "session_day"], how="left")
            .with_columns([
                pl.when(pl.col("revenue_cents") > 0).then(pl.col("top_category")).otherwise(None).alias("top_category"),
            ])
            .select([
                pl.col("person").alias("user_id"),
                pl.col("session_day").alias("day"),
                "sessions",
                "events",
                "revenue_cents",
                "top_category",
            ])
            .cast(USER_DAILY)
        )
    else:
        sessions_df = pl.DataFrame(
            schema={"person": pl.String, "sess_num": pl.Int64, "session_start_ts": pl.Int64, "s_events": pl.Int64, "s_purchases": pl.Int64, "s_revenue": pl.Int64, "session_day": pl.Date}
        )
        user_df = pl.DataFrame(schema=USER_DAILY)

    # 4. Attribution & 7-day assists per session
    camp_acc: dict[tuple[str, object], dict] = {}

    def get_camp(cid: str, day):
        key = (cid, day)
        entry = camp_acc.get(key)
        if entry is None:
            ch = "organic" if cid == "organic" else campaigns[cid]
            entry = {
                "campaign_id": cid,
                "channel": ch,
                "day": day,
                "sessions": 0,
                "purchasing_sessions": 0,
                "revenue_cents": 0,
                "assisted_sessions": 0,
            }
            camp_acc[key] = entry
        return entry

    if sessions_df.height > 0:
        clicks_by_p: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for row in cl_df.iter_rows(named=True):
            clicks_by_p[row["person"]].append((int(row["ts_ms"]), row["click_id"], row["campaign_id"]))

        for s_row in sessions_df.iter_rows(named=True):
            person = s_row["person"]
            s_start = int(s_row["session_start_ts"])
            s_day = s_row["session_day"]
            s_purch = int(s_row["s_purchases"])
            s_rev = int(s_row["s_revenue"])

            min_ts = s_start - LOOKBACK_7D_MS
            p_clicks = clicks_by_p.get(person, [])
            qualifying = [c for c in p_clicks if min_ts <= c[0] <= s_start]

            if qualifying:
                latest_ts = max(c[0] for c in qualifying)
                best_click = min(
                    (c for c in qualifying if c[0] == latest_ts),
                    key=lambda c: c[1].encode("utf-8"),
                )
                credited_cid = best_click[2]
                assisting_cids = {c[2] for c in qualifying}
            else:
                credited_cid = "organic"
                assisting_cids = set()

            c_entry = get_camp(credited_cid, s_day)
            c_entry["sessions"] += 1
            if s_purch > 0:
                c_entry["purchasing_sessions"] += 1
            c_entry["revenue_cents"] += s_rev

            for a_cid in assisting_cids:
                get_camp(a_cid, s_day)["assisted_sessions"] += 1

    user_folder = os.path.join(OUT, "user_daily")
    camp_folder = os.path.join(OUT, "campaign_daily")
    os.makedirs(user_folder, exist_ok=True)
    os.makedirs(camp_folder, exist_ok=True)

    camp_rows = [
        {
            "campaign_id": c["campaign_id"],
            "channel": c["channel"],
            "day": c["day"],
            "sessions": c["sessions"],
            "purchasing_sessions": c["purchasing_sessions"],
            "revenue_cents": c["revenue_cents"],
            "assisted_sessions": c["assisted_sessions"],
        }
        for c in camp_acc.values()
    ]
    camp_df = pl.DataFrame(camp_rows, schema=CAMPAIGN_DAILY) if camp_rows else pl.DataFrame(schema=CAMPAIGN_DAILY)

    user_df.write_parquet(os.path.join(user_folder, "part-0.parquet"))
    camp_df.write_parquet(os.path.join(camp_folder, "part-0.parquet"))
    print(f"wrote {len(user_df)} user_daily rows and {len(camp_df)} campaign_daily rows")


if __name__ == "__main__":
    main()
