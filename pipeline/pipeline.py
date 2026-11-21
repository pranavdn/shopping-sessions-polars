#!/usr/bin/env python3
"""
Shopping Sessions: Polars starter — Complete rule-accurate implementation.
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
SESSION_GAP_MS = 30 * MS_PER_MIN        # 1,800,000 ms (new session if > this)
LATE_LIMIT_MS = 24 * MS_PER_HOUR        # 86,400,000 ms (drop if ingest_ts - ts > this)
LOOKBACK_7D_MS = 7 * MS_PER_DAY         # 604,800,000 ms

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
    """
    Union-Find over links/*.csv where each person is named by the byte-wise smallest
    user_id in their connected component. Unlinked user_ids resolve to themselves.
    """
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


def utc_date_from_ms(ts_ms: int):
    days = ts_ms // MS_PER_DAY
    return (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=days)).date()


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

    events_by_person: dict[str, list[tuple[int, str, int, bool]]] = defaultdict(list)

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

        person = resolve_person(uid)
        category = products.get(pid, "unknown")
        events_by_person[person].append((ts_ms, category, price_cents, is_purchase))

    # 2. Clicks: parse -> dedup on click_id -> validate
    seen_click_ids: set[str] = set()
    clicks_by_person: dict[str, list[tuple[int, str, str]]] = defaultdict(list)

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

            person = resolve_person(uid)
            clicks_by_person[person].append((ts_ms, click_id, cid))

    for p_clicks in clicks_by_person.values():
        p_clicks.sort(key=lambda x: x[0])

    # 3. Sessionize per person, attribute to click, count 7-day assists, roll up daily tables
    user_acc: dict[tuple[str, object], dict] = {}
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

    for person, p_events in events_by_person.items():
        p_events.sort(key=lambda x: x[0])
        p_clicks = clicks_by_person.get(person, [])

        n = len(p_events)
        idx = 0
        while idx < n:
            session_start_ts = p_events[idx][0]
            session_day = utc_date_from_ms(session_start_ts)
            prev_distinct_ts = session_start_ts

            s_events = 0
            s_purchases = 0
            s_revenue = 0
            s_cat_rev: dict[str, int] = defaultdict(int)

            while idx < n:
                ev_ts, ev_cat, ev_price, ev_is_purch = p_events[idx]
                if ev_ts != prev_distinct_ts:
                    if (ev_ts - prev_distinct_ts) > SESSION_GAP_MS:
                        break
                    prev_distinct_ts = ev_ts

                s_events += 1
                if ev_is_purch:
                    s_purchases += 1
                    s_revenue += ev_price
                    s_cat_rev[ev_cat] += ev_price
                idx += 1

            ukey = (person, session_day)
            u_entry = user_acc.get(ukey)
            if u_entry is None:
                u_entry = {
                    "user_id": person,
                    "day": session_day,
                    "sessions": 0,
                    "events": 0,
                    "revenue_cents": 0,
                    "cat_rev": defaultdict(int),
                }
                user_acc[ukey] = u_entry
            u_entry["sessions"] += 1
            u_entry["events"] += s_events
            u_entry["revenue_cents"] += s_revenue
            for c_name, c_amt in s_cat_rev.items():
                u_entry["cat_rev"][c_name] += c_amt

            min_ts = session_start_ts - LOOKBACK_7D_MS
            qualifying = [
                c for c in p_clicks if min_ts <= c[0] <= session_start_ts
            ]

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

            c_entry = get_camp(credited_cid, session_day)
            c_entry["sessions"] += 1
            if s_purchases > 0:
                c_entry["purchasing_sessions"] += 1
            c_entry["revenue_cents"] += s_revenue

            for a_cid in assisting_cids:
                get_camp(a_cid, session_day)["assisted_sessions"] += 1

    user_rows = []
    for u in user_acc.values():
        rev = u["revenue_cents"]
        if rev > 0 and u["cat_rev"]:
            top_cat = min(
                u["cat_rev"].items(),
                key=lambda kv: (-kv[1], kv[0].encode("utf-8")),
            )[0]
        else:
            top_cat = None
        user_rows.append({
            "user_id": u["user_id"],
            "day": u["day"],
            "sessions": u["sessions"],
            "events": u["events"],
            "revenue_cents": rev,
            "top_category": top_cat,
        })

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

    user_folder = os.path.join(OUT, "user_daily")
    camp_folder = os.path.join(OUT, "campaign_daily")
    os.makedirs(user_folder, exist_ok=True)
    os.makedirs(camp_folder, exist_ok=True)

    user_df = pl.DataFrame(user_rows, schema=USER_DAILY) if user_rows else pl.DataFrame(schema=USER_DAILY)
    camp_df = pl.DataFrame(camp_rows, schema=CAMPAIGN_DAILY) if camp_rows else pl.DataFrame(schema=CAMPAIGN_DAILY)

    user_df.write_parquet(os.path.join(user_folder, "part-0.parquet"))
    camp_df.write_parquet(os.path.join(camp_folder, "part-0.parquet"))
    print(f"wrote {len(user_df)} user_daily rows and {len(camp_df)} campaign_daily rows")


if __name__ == "__main__":
    main()