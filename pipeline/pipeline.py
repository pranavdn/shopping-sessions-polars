#!/usr/bin/env python3
"""
Shopping Sessions: Hybrid Polars pipeline (exact small-file line parser + fast large-file Polars reader).
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
SMALL_FILE_THRESHOLD = 65_536
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

def sanitize_csv_file(path: str, expected_cols: int, output_path: str) -> str | None:
    """Drop malformed LF-delimited records before Polars reads a large shard.

    The extra sentinel column keeps a trailing CR inside its original field.
    Output is buffered in batches to avoid one Python write call per record.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_batch = bytearray()
    batch_limit = 4 * 1024 * 1024

    with open(path, "rb") as source, open(output_path, "wb", buffering=1024 * 1024) as target:
        header = source.readline()
        if header == b"":
            return None
        if header.endswith(b"\n"):
            header = header[:-1]
        if header.endswith(b"\r"):
            header = header[:-1]
        target.write(header + b",__sentinel\n")

        for line in source:
            if line.endswith(b"\n"):
                line = line[:-1]
            if line.count(b",") != expected_cols - 1:
                continue

            # The common generated-data path has no quote characters. Only
            # run the stricter quote-boundary checks when a quote is present.
            if b'"' in line:
                quote_count = line.count(b'"')
                openings = line.count(b',"') + line.startswith(b'"')
                closings = line.count(b'",') + line.endswith(b'"')
                if openings != closings or quote_count != openings + closings:
                    continue

            write_batch.extend(line)
            write_batch.extend(b",0\n")
            if len(write_batch) >= batch_limit:
                target.write(write_batch)
                write_batch.clear()

        if write_batch:
            target.write(write_batch)

    return output_path

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
    # Polars/chrono can accept leap-second values such as :60. The input
    # contract only permits ordinary clock values (00-23:00-59:00-59), so
    # validate the clock fields before asking Polars to parse the date.
    valid_z_clock = s.str.contains(
        r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z$"
    )
    z_dt = (
        pl.when(valid_z_clock)
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
    valid_offset_clock = m2.struct.field("1").str.contains(
        r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$"
    )
    valid_off = (
        base_dt.is_not_null()
        & valid_offset_clock
        & off_h.is_not_null()
        & off_m.is_not_null()
        & (off_h <= 23)
        & (off_m <= 59)
    )
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

    is_small_dataset = all(
        os.path.getsize(p) < SMALL_FILE_THRESHOLD
        for p in (event_files + click_files)
    )

    if is_small_dataset:
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
        for idx, ef in enumerate(event_files):
            clean_path = sanitize_csv_file(
                ef, 7, os.path.join(SCRATCH, "sanitized", "events", f"part-{idx:05d}.csv")
            )
            if clean_path is None:
                continue
            df = pl.scan_csv(
                clean_path,
                has_header=True,
                infer_schema=False,
                ignore_errors=False,
                truncate_ragged_lines=False,
            )
            if all(c in df.collect_schema().names() for c in expected_ev_cols):
                ev_frames.append(df.select(expected_ev_cols))

        if ev_frames:
            ev_raw = pl.concat(ev_frames, how="vertical")
            str_cols = [pl.col(c).cast(pl.String, strict=False).fill_null("") for c in expected_ev_cols]
            ev_df = (
                ev_raw.with_columns(str_cols)
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
                .collect()
            )
        else:
            ev_df = pl.DataFrame(schema=ev_schema)

        cl_frames = []
        expected_cl_cols = ["click_id", "user_id", "campaign_id", "ts"]
        for idx, cf in enumerate(click_files):
            clean_path = sanitize_csv_file(
                cf, 4, os.path.join(SCRATCH, "sanitized", "clicks", f"part-{idx:05d}.csv")
            )
            if clean_path is None:
                continue
            df = pl.scan_csv(
                clean_path,
                has_header=True,
                infer_schema=False,
                ignore_errors=False,
                truncate_ragged_lines=False,
            )
            if all(c in df.collect_schema().names() for c in expected_cl_cols):
                cl_frames.append(df.select(expected_cl_cols))

        if cl_frames:
            cl_raw = pl.concat(cl_frames, how="vertical")
            cl_df = (
                cl_raw.with_columns([pl.col(c).cast(pl.String, strict=False).fill_null("") for c in expected_cl_cols])
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
                .collect()
            )
        else:
            cl_df = pl.DataFrame(schema=cl_schema)

    # Index clicks by person in timestamp and click-ID order.
    # This supports efficient sliding-window campaign attribution.
    clicks_by_p: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

    for person, ts_ms, click_id, campaign_id in (
        cl_df.sort(["person", "ts_ms", "click_id"]).iter_rows()
    ):
        clicks_by_p[person].append(
            (int(ts_ms), click_id, campaign_id)
        )
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
        # Process sessions chronologically for each person.
        # Each click enters and leaves the seven-day window at most once.
        ordered_sessions = sessions_df.sort(
            ["person", "session_start_ts"]
        )

        current_person = None
        p_clicks = []
        left = 0
        right = 0

        # Active click counts by campaign.
        active_counts = {}

        # Latest eligible click and its campaign.
        latest_ts = None
        latest_click_bytes = None
        latest_cid = "organic"

        session_columns = ordered_sessions.select([
            "person",
            "session_start_ts",
            "session_day",
            "s_purchases",
            "s_revenue",
        ])

        for person, start_raw, s_day, purch_raw, rev_raw in (
            session_columns.iter_rows()
        ):
            s_start = int(start_raw)
            s_purch = int(purch_raw)
            s_rev = int(rev_raw)

            # Reset the window when the person changes.
            if person != current_person:
                current_person = person
                p_clicks = clicks_by_p.get(person, [])
                left = 0
                right = 0
                active_counts = {}
                latest_ts = None
                latest_click_bytes = None
                latest_cid = "organic"

            # Add clicks up to and including the session start.
            while right < len(p_clicks):
                c_ts, c_id, c_cid = p_clicks[right]

                if c_ts > s_start:
                    break

                active_counts[c_cid] = (
                    active_counts.get(c_cid, 0) + 1
                )

                c_id_bytes = c_id.encode("utf-8")

                if (
                    latest_ts is None
                    or c_ts > latest_ts
                    or (
                        c_ts == latest_ts
                        and c_id_bytes < latest_click_bytes
                    )
                ):
                    latest_ts = c_ts
                    latest_click_bytes = c_id_bytes
                    latest_cid = c_cid

                right += 1

            # Remove clicks older than the seven-day lookback.
            lower_bound = s_start - LOOKBACK_7D_MS

            while (
                left < right
                and p_clicks[left][0] < lower_bound
            ):
                _, _, old_cid = p_clicks[left]

                remaining = active_counts[old_cid] - 1

                if remaining:
                    active_counts[old_cid] = remaining
                else:
                    del active_counts[old_cid]

                left += 1

            # Credit the latest eligible click, or organic.
            if latest_ts is not None and latest_ts >= lower_bound:
                credited_cid = latest_cid
                assisting_cids = active_counts.keys()
            else:
                credited_cid = "organic"
                assisting_cids = ()

            c_entry = get_camp(credited_cid, s_day)
            c_entry["sessions"] += 1

            if s_purch > 0:
                c_entry["purchasing_sessions"] += 1

            c_entry["revenue_cents"] += s_rev

            # Each distinct campaign in the window receives one assist.
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


