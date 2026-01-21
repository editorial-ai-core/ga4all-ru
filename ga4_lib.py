# ga4_lib.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from collections.abc import Mapping as MappingABC
from urllib.parse import urlparse

import pandas as pd

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    Filter,
    FilterExpression,
    FilterExpressionList,
    InListFilter,
    OrderBy,
    RunReportRequest,
)
from google.oauth2 import service_account


# =========================
# Errors / Config
# =========================

class GA4ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GA4Config:
    property_id: str
    service_account_info: Dict[str, Any]


def _require_str(secrets: Mapping[str, Any], key: str) -> str:
    v = secrets.get(key, None)
    if v is None:
        raise GA4ConfigError(f"Missing secret: {key}")
    if not isinstance(v, str) or not v.strip():
        raise GA4ConfigError(f"Invalid secret: {key} (must be non-empty string)")
    return v.strip()


def _require_mapping(secrets: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """
    Streamlit returns st.secrets sections as SecretDict / Mapping-like objects,
    not necessarily plain dict. We accept any Mapping and cast to dict.
    """
    v = secrets.get(key, None)
    if not isinstance(v, MappingABC):
        raise GA4ConfigError(f"Missing secret: {key} (dict)")
    return dict(v)


def build_config_from_streamlit_secrets(secrets: Mapping[str, Any]) -> GA4Config:
    """
    Expected secrets.toml:

    GA4_PROPERTY_ID = "123456789"

    [gcp_service_account]
    type="service_account"
    project_id="..."
    private_key_id="..."
    private_key="""-----BEGIN PRIVATE KEY-----..."""
    client_email="..."
    token_uri="https://oauth2.googleapis.com/token"
    """
    property_id = _require_str(secrets, "GA4_PROPERTY_ID")
    sa_info = _require_mapping(secrets, "gcp_service_account")

    # Minimal sanity checks (do not over-restrict)
    for k in ("type", "project_id", "private_key", "client_email", "token_uri"):
        if not sa_info.get(k):
            raise GA4ConfigError(f"gcp_service_account missing field: {k}")

    # Streamlit TOML triple-quoted key comes with real newlines - good.
    # But if someone pasted \n escapes, normalize.
    pk = sa_info.get("private_key", "")
    if isinstance(pk, str) and "\\n" in pk and "BEGIN PRIVATE KEY" in pk:
        sa_info["private_key"] = pk.replace("\\n", "\n")

    return GA4Config(property_id=property_id, service_account_info=sa_info)


def make_client(cfg: GA4Config) -> BetaAnalyticsDataClient:
    creds = service_account.Credentials.from_service_account_info(
        cfg.service_account_info,
        scopes=["https://www.googleapis.com/auth/analytics.readonly"],
    )
    return BetaAnalyticsDataClient(credentials=creds)


# =========================
# URL parsing helpers
# =========================

def _normalize_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return "/"
    if not p.startswith("/"):
        p = "/" + p
    # remove trailing whitespace; keep trailing slash as-is (GA4 paths are exact)
    return p


def collect_paths_hosts(lines: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Input: list of strings (URLs or paths), one per line.
    Output:
      unique_paths: list[str] unique pagePath values
      hostnames: list[str] unique hostName values (only from full URLs)
      order_paths: list[str] paths in the same order user entered (normalized)
    """
    paths_order: List[str] = []
    hosts: List[str] = []

    for raw in lines:
        s = (raw or "").strip()
        if not s:
            continue

        # If looks like URL without scheme, add https:// for parsing
        if "://" not in s and (s.startswith("www.") or "." in s.split("/")[0]):
            # treat as domain/path
            s_for_parse = "https://" + s
        else:
            s_for_parse = s

        if "://" in s_for_parse:
            u = urlparse(s_for_parse)
            host = (u.netloc or "").strip()
            path = _normalize_path(u.path)
            if host:
                hosts.append(host)
            paths_order.append(path)
        else:
            # plain path
            paths_order.append(_normalize_path(s))

    # unique while preserving order
    def uniq(seq: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    unique_paths = uniq(paths_order)
    hostnames = uniq(hosts)
    order_paths = paths_order[:]  # keep duplicates for ordering if needed

    return unique_paths, hostnames, order_paths


# =========================
# GA4 report helpers
# =========================

def _chunks(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _in_list_filter(field_name: str, values: Sequence[str]) -> FilterExpression:
    return FilterExpression(
        filter=Filter(
            field_name=field_name,
            in_list_filter=InListFilter(values=list(values), case_sensitive=False),
        )
    )


def _and_filters(filters: List[FilterExpression]) -> Optional[FilterExpression]:
    filters = [f for f in filters if f is not None]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return FilterExpression(
        and_group=FilterExpressionList(expressions=filters)
    )


def _run_report(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
    dimensions: Sequence[str],
    metrics: Sequence[str],
    dimension_filter: Optional[FilterExpression] = None,
    order_bys: Optional[List[OrderBy]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    req = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name=d) for d in dimensions],
        metrics=[Metric(name=m) for m in metrics],
    )
    if dimension_filter is not None:
        req.dimension_filter = dimension_filter
    if order_bys:
        req.order_bys = order_bys
    if limit is not None:
        req.limit = int(limit)

    resp = client.run_report(req)

    dim_headers = [h.name for h in resp.dimension_headers]
    met_headers = [h.name for h in resp.metric_headers]

    rows = []
    for r in resp.rows:
        dvals = [dv.value for dv in r.dimension_values]
        mvals = [mv.value for mv in r.metric_values]
        row = dict(zip(dim_headers, dvals))
        row.update(dict(zip(met_headers, mvals)))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def _to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# Public API for Streamlit app
# =========================

def fetch_ga4_by_paths(
    client: BetaAnalyticsDataClient,
    property_id: str,
    paths_in: Sequence[str],
    hosts_in: Optional[Sequence[str]],
    start_date: str,
    end_date: str,
    order_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    For URL Analytics tab.
    Returns columns:
      pagePath, pageTitle,
      screenPageViews (Views),
      activeUsers,
      sessions,
      engagementRate,
      avgEngagementTime_sec  (userEngagementDuration / sessions)
    """
    paths = [p for p in paths_in if p]
    if not paths:
        return pd.DataFrame(
            columns=[
                "pagePath", "pageTitle",
                "screenPageViews", "activeUsers", "sessions",
                "engagementRate", "avgEngagementTime_sec",
            ]
        )

    hosts = [h for h in (hosts_in or []) if h]

    dims = ["pagePath", "pageTitle"]
    mets = ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"]

    dfs: List[pd.DataFrame] = []

    # Chunk paths to avoid request size limits
    for path_chunk in _chunks(paths, 50):
        filters: List[FilterExpression] = [_in_list_filter("pagePath", path_chunk)]
        if hosts:
            filters.append(_in_list_filter("hostName", hosts))

        dim_filter = _and_filters(filters)

        df = _run_report(
            client=client,
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
            dimensions=dims + (["hostName"] if hosts else []),
            metrics=mets,
            dimension_filter=dim_filter,
        )
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if df_all.empty:
        out = pd.DataFrame(
            columns=[
                "pagePath", "pageTitle",
                "screenPageViews", "activeUsers", "sessions",
                "engagementRate", "avgEngagementTime_sec",
            ]
        )
        return out

    # Normalize and aggregate in case multiple chunks returned duplicates
    df_all = _to_numeric(df_all, ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"])

    group_cols = ["pagePath", "pageTitle"]
    agg = {
        "screenPageViews": "sum",
        "activeUsers": "sum",
        "sessions": "sum",
        # engagementRate is a ratio. If aggregating, do weighted by sessions:
        "engagementRate": "mean",
        "userEngagementDuration": "sum",
    }
    df_g = df_all.groupby(group_cols, dropna=False, as_index=False).agg(agg)

    # Weighted engagementRate by sessions (better than mean)
    # engagementRate in API is typically already computed, but per-row can vary.
    # We recompute weighted if possible.
    if "engagementRate" in df_all.columns and "sessions" in df_all.columns:
        # Recompute engagementRate = engagedSessions / sessions
        # But we don't have engagedSessions metric here; approximate using weighted average:
        # sum(rate * sessions)/sum(sessions)
        tmp = df_all.copy()
        tmp["w"] = tmp["sessions"].fillna(0)
        tmp["wr"] = tmp["engagementRate"].fillna(0) * tmp["w"]
        wr = tmp.groupby(group_cols, dropna=False, as_index=False)[["wr", "w"]].sum()
        df_g = df_g.merge(wr, on=group_cols, how="left")
        df_g["engagementRate"] = (df_g["wr"] / df_g["w"]).where(df_g["w"] > 0, 0)
        df_g = df_g.drop(columns=["wr", "w"])

    # Avg session duration (seconds) = userEngagementDuration / sessions
    df_g["avgEngagementTime_sec"] = (df_g["userEngagementDuration"] / df_g["sessions"]).where(df_g["sessions"] > 0, 0)
    df_g = df_g.drop(columns=["userEngagementDuration"])

    # Order by user input order (first occurrence)
    if order_keys:
        order = list(order_keys)
        pos = {p: i for i, p in enumerate(order) if p}
        df_g["__ord"] = df_g["pagePath"].map(lambda x: pos.get(x, 10**9))
        df_g = df_g.sort_values(["__ord", "screenPageViews"], ascending=[True, False]).drop(columns="__ord")
    else:
        df_g = df_g.sort_values("screenPageViews", ascending=False)

    # Ensure column order
    df_g = df_g.reindex(
        columns=[
            "pagePath", "pageTitle",
            "screenPageViews", "activeUsers", "sessions",
            "engagementRate", "avgEngagementTime_sec",
        ]
    )

    return df_g


def fetch_top_materials(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    For Top materials tab.
    Returns:
      pagePath, pageTitle,
      screenPageViews, activeUsers, sessions, engagementRate,
      avgEngagementTime_sec (userEngagementDuration / sessions)
    """
    dims = ["pagePath", "pageTitle"]
    mets = ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"]

    df = _run_report(
        client=client,
        property_id=property_id,
        start_date=start_date,
        end_date=end_date,
        dimensions=dims,
        metrics=mets,
        order_bys=[
            OrderBy(
                metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"),
                desc=True,
            )
        ],
        limit=int(limit),
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "pagePath", "pageTitle",
                "screenPageViews", "activeUsers", "sessions",
                "engagementRate", "avgEngagementTime_sec",
            ]
        )

    df = _to_numeric(df, ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"])
    df["avgEngagementTime_sec"] = (df["userEngagementDuration"] / df["sessions"]).where(df["sessions"] > 0, 0)
    df = df.drop(columns=["userEngagementDuration"])

    df = df.reindex(
        columns=[
            "pagePath", "pageTitle",
            "screenPageViews", "activeUsers", "sessions",
            "engagementRate", "avgEngagementTime_sec",
        ]
    )
    return df


def fetch_site_totals(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
) -> Dict[str, int]:
    """
    For Site totals tab.
    Metrics:
      screenPageViews (Views / Page Views),
      sessions,
      totalUsers,
      activeUsers
    """
    df = _run_report(
        client=client,
        property_id=property_id,
        start_date=start_date,
        end_date=end_date,
        dimensions=[],
        metrics=["screenPageViews", "sessions", "totalUsers", "activeUsers"],
    )
    if df.empty:
        return {"screenPageViews": 0, "sessions": 0, "totalUsers": 0, "activeUsers": 0}

    df = _to_numeric(df, ["screenPageViews", "sessions", "totalUsers", "activeUsers"])
    row = df.iloc[0].to_dict()

    def i(key: str) -> int:
        v = row.get(key, 0)
        try:
            return int(v) if pd.notna(v) else 0
        except Exception:
            return 0

    return {
        "screenPageViews": i("screenPageViews"),
        "sessions": i("sessions"),
        "totalUsers": i("totalUsers"),
        "activeUsers": i("activeUsers"),
    }
