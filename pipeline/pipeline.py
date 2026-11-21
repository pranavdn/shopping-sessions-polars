#!/usr/bin/env python3
"""
Shopping Sessions: Hybrid Polars pipeline (exact small-file line parser + fast large-file Polars reader).
"""
from bisect import bisect_left, bisect_right
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
EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

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
    """Read LF-delimited CSV records one at a time, preserving CR as field data.

    The input contract drops malformed physical lines independently. Polars'
    parallel CSV reader can lose valid rows after malformed quoting in a chunk,
    so use the exact line parser for every shard.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        header = f.readline()
        if header == "":
            return
        for line in f:
            # Remove LF only. A preceding CR is ordinary field content by spec.
            if line.endswith("\n"):
                line = line[:-1]
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
        epoch_sec = (dt - EPOCH_UTC) // timedelta(seconds=1)
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
        epoch_sec = (dt_utc - EPOCH_UTC) // timedelta(seconds=1)
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


def ts_expr_to_ms(col_name: str) -> pl.Expr:
    s = pl.col(col_name).cast(pl.String, strict=False).fill_null("")
    digits_ms = (
        pl.when(s.str.contains(r"^[0-9]+$"))
        .then(s.str.to_integer(dtype=pl.Int64, strict=False))
        .otherwise(None)
    )
    z_dt = (
        pl.when(s.str.contains(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"))
        .then(s.str.strptime(pl.Datetime("ms", "UTC"), format="%Y-%m-%dT%H:%M:%SZ", strict=False))
        .otherwise(None)
    )
    m2 = s.str.extract_groups(
        r"^([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})([+-])([0-9]{2}):([0-9]{2})$"
    )
    base_dt = m2.struct.field("1").str.strptime(pl.Datetime("ms", "UTC"), format="%Y-%m-%dT%H:%M:%S", strict=False)
    sign = m2.struct.field("2")
    off_h = m2.struct.field("3").cast(pl.Int64, strict=False)
    off_m = m2.struct.field("4").cast(pl.Int64, strict=False)
    valid_off = base_dt.is_not_null() & off_h.is_not_null() & off_m.is_not_null() & (off_h <= 23) & (off_m <= 59)
    offset_ms = (off_h * 3_600_000 + off_m * 60_000) * pl.when(sign == "+").then(1).otherwise(-1)
    tz_ms = pl.when(valid_off).then(base_dt.dt.epoch("ms") - offset_ms).otherwise(None)

    return pl.coalesce([digits_ms, z_dt.dt.epoch("ms"), tz_ms])


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
    ev_schema = {"person": pl.String, "ts_ms": pl.Int64, "category": pl.String, "is_purchase": pl.Boolean, "rev_cents": pl.Int64}
    cl_schema = {"person": pl.String, "ts_ms": pl.Int64, "click_id": pl.String, "campaign_id": pl.String}

    # Always use the exact line parser. The chunked Polars CSV path silently
    # loses valid dev rows when malformed quoting desynchronizes a read chunk.
    use_exact_line_parser = True

    if use_exact_line_parser:
        kept_by_event_id: dict[str, tuple[int, list[str]]] = {}
        for ef in event_files:
            for row in iter_csv_rows(ef, 7):
                event_id = row[0]
                if event_id == "":
                    continue
                ingest_ms = parse_ts_ms(row[3])
                if ingest_ms is None:
                    continue
                prev = kept_by_event_id.get(event_id)
                if prev is None or ingest_ms < prev[0]:
                    kept_by_event_id[event_id] = (ingest_ms, row)

        valid_events = []
        for _event_id, (ingest_ms, row) in kept_by_event_id.items():
            uid, etype, pid = row[1], row[4], row[5]
            if uid == "" or pid == "" or etype not in VALID_EVENT_TYPES:
                continue
            ts_ms = parse_ts_ms(row[2])
            if ts_ms is None or not (win_start_ms <= ts_ms < win_end_ms):
                continue

            is_purchase = etype == "purchase"
            price_cents = 0
            if is_purchase:
                p = parse_price_cents(row[6])
                if p is None:
                    continue
                price_cents = p

            if (ingest_ms - ts_ms) > LATE_LIMIT_MS:
                continue

            valid_events.append({
                "person": person_map.get(uid, uid),
                "ts_ms": ts_ms,
                "category": products.get(pid, "unknown"),
                "is_purchase": is_purchase,
                "rev_cents": price_cents,
            })

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
                    "person": person_map.get(uid, uid),
                    "ts_ms": ts_ms,
                    "click_id": click_id,
                    "campaign_id": cid,
                })

        ev_df = pl.DataFrame(valid_events, schema=ev_schema) if valid_events else pl.DataFrame(schema=ev_schema)
        cl_df = pl.DataFrame(valid_clicks, schema=cl_schema) if valid_clicks else pl.DataFrame(schema=cl_schema)
    else:
        ev_frames = []
        expected_ev_cols = ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"]
        for ef in event_files:
            try:
                df = pl.read_csv(
                    ef,
                    has_header=True,
                    infer_schema=False,
                    ignore_errors=True,
                    truncate_ragged_lines=False,
                )
            except Exception:
                continue
            if all(c in df.columns for c in expected_ev_cols):
                ev_frames.append(df.select(expected_ev_cols))

        if ev_frames:
            ev_raw = pl.concat(ev_frames, how="vertical")
            str_cols = [pl.col(c).cast(pl.String, strict=False).fill_null("") for c in expected_ev_cols]
            ev_df = (
                ev_raw.with_columns(str_cols)
                .filter(~pl.col("event_id").str.contains("[\r\n\"]"))
                .with_columns([
                    ts_expr_to_ms("ingest_ts").alias("ingest_ms"),
                    ts_expr_to_ms("ts").alias("ts_ms"),
                ])
                .filter((pl.col("event_id") != "") & pl.col("ingest_ms").is_not_null())
                .sort("ingest_ms")
                .unique(subset=["event_id"], keep="first")
                .with_columns([
                    pl.when(pl.col("price_cents").str.contains(r"^[0-9]+$"))
                    .then(pl.col("price_cents").str.to_integer(dtype=pl.Int64, strict=False))
                    .otherwise(None)
                    .alias("price_int"),
                ])
                .filter(
                    (pl.col("user_id") != "")
                    & (pl.col("product_id") != "")
                    & (pl.col("event_type").is_in(list(VALID_EVENT_TYPES)))
                    & pl.col("ts_ms").is_not_null()
                    & (pl.col("ts_ms") >= win_start_ms)
                    & (pl.col("ts_ms") < win_end_ms)
                    & ((pl.col("ingest_ms") - pl.col("ts_ms")) <= LATE_LIMIT_MS)
                    & (
                        (pl.col("event_type") != "purchase")
                        | (pl.col("price_int").is_not_null() & (pl.col("price_int") >= 1) & (pl.col("price_int") <= 10_000_000))
                    )
                )
                .with_columns([
                    pl.col("user_id").replace_strict(person_map, default=None, return_dtype=pl.String).fill_null(pl.col("user_id")).alias("person"),
                    pl.col("product_id").replace_strict(products, default=None, return_dtype=pl.String).fill_null("unknown").alias("category"),
                    (pl.col("event_type") == "purchase").alias("is_purchase"),
                    pl.when(pl.col("event_type") == "purchase")
                    .then(pl.col("price_int").fill_null(0))
                    .otherwise(0)
                    .cast(pl.Int64)
                    .alias("rev_cents"),
                ])
                .select(["person", "ts_ms", "category", "is_purchase", "rev_cents"])
            )
        else:
            ev_df = pl.DataFrame(schema=ev_schema)

        cl_frames = []
        expected_cl_cols = ["click_id", "user_id", "campaign_id", "ts"]
        for cf in click_files:
            try:
                df = pl.read_csv(
                    cf,
                    has_header=True,
                    infer_schema=False,
                    ignore_errors=True,
                    truncate_ragged_lines=False,
                )
            except Exception:
                continue
            if all(c in df.columns for c in expected_cl_cols):
                cl_frames.append(df.select(expected_cl_cols))

        if cl_frames:
            cl_raw = pl.concat(cl_frames, how="vertical")
            cl_df = (
                cl_raw.with_columns([pl.col(c).cast(pl.String, strict=False).fill_null("") for c in expected_cl_cols])
                .filter(~pl.col("click_id").str.contains("[\r\n\"]"))
                .with_columns(ts_expr_to_ms("ts").alias("ts_ms"))
                .filter(pl.col("click_id") != "")
                .unique(subset=["click_id"], keep="first", maintain_order=True)
                .filter(
                    (pl.col("user_id") != "")
                    & (pl.col("campaign_id").is_in(list(campaigns.keys())))
                    & pl.col("ts_ms").is_not_null()
                    & (pl.col("ts_ms") >= click_start_ms)
                    & (pl.col("ts_ms") < win_end_ms)
                )
                .with_columns([
                    pl.col("user_id").replace_strict(person_map, default=None, return_dtype=pl.String).fill_null(pl.col("user_id")).alias("person"),
                ])
                .select(["person", "ts_ms", "click_id", "campaign_id"])
                .sort(["person", "ts_ms"])
            )
        else:
            cl_df = pl.DataFrame(schema=cl_schema)

    clicks_by_p: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for row in cl_df.iter_rows(named=True):
        clicks_by_p[row["person"]].append((int(row["ts_ms"]), row["click_id"], row["campaign_id"]))

    click_ts_by_p: dict[str, list[int]] = {}
    for p_key, p_list in clicks_by_p.items():
        p_list.sort(key=lambda x: x[0])
        click_ts_by_p[p_key] = [x[0] for x in p_list]

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
        for s_row in sessions_df.iter_rows(named=True):
            person = s_row["person"]
            s_start = int(s_row["session_start_ts"])
            s_day = s_row["session_day"]
            s_purch = int(s_row["s_purchases"])
            s_rev = int(s_row["s_revenue"])

            p_clicks = clicks_by_p.get(person)
            if p_clicks:
                p_ts = click_ts_by_p[person]
                lo = bisect_left(p_ts, s_start - LOOKBACK_7D_MS)
                hi = bisect_right(p_ts, s_start)
                if lo < hi:
                    latest_ts = p_ts[hi - 1]
                    best_click_id: str | None = None
                    best_click_bytes: bytes | None = None
                    credited_cid = ""
                    assisting_cids: set[str] = set()
                    for idx_c in range(lo, hi):
                        c_ts, c_id, c_cid = p_clicks[idx_c]
                        assisting_cids.add(c_cid)
                        if c_ts == latest_ts:
                            c_bytes = c_id.encode("utf-8")
                            if best_click_bytes is None or c_bytes < best_click_bytes:
                                best_click_bytes = c_bytes
                                best_click_id = c_id
                                credited_cid = c_cid
                    assert best_click_id is not None
                else:
                    credited_cid = "organic"
                    assisting_cids = set()
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


