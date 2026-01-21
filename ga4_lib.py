from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Dict, List
from collections.abc import Mapping
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import pandas as pd


# ----------------------------
# Errors / Config
# ----------------------------

class GA4ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class GA4Config:
    property_id: str
    service_account_info: dict[str, Any]
    scopes: tuple[str, ...] = ("https://www.googleapis.com/auth/analytics.readonly",)


def _as_mapping(x: Any) -> Optional[Mapping]:
    return x if isinstance(x, Mapping) else None


def build_config_from_streamlit_secrets(secrets: Any) -> GA4Config:
    """
    Expected secrets.toml:

    GA4_PROPERTY_ID = "123456789"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = """-----BEGIN PRIVATE KEY-----
    ...
    -----END PRIVATE KEY-----"""
    client_email = "....iam.gserviceaccount.com"
    token_uri = "https://oauth2.googleapis.com/token"
    """
    if not hasattr(secrets, "get"):
        raise GA4ConfigError("Secrets must be Mapping-like (st.secrets).")

    property_id = str(secrets.get("GA4_PROPERTY_ID", "")).strip()
    if not property_id:
        raise GA4ConfigError("Missing secret: GA4_PROPERTY_ID")

    sa = secrets.get("gcp_service_account", None)
    sa_map = _as_mapping(sa)
    if sa_map is None:
        raise GA4ConfigError("Missing secret: gcp_service_account (dict/table)")

    # Streamlit returns AttrDict/Mapping; convert to plain dict
    sa_dict = dict(sa_map)

    # minimal sanity
    if not sa_dict.get("client_email") or not sa_dict.get("private_key"):
        raise GA4ConfigError("gcp_service_account must contain at least client_email and private_key")

    return GA4Config(property_id=property_id, service_account_info=sa_dict)


def make_client(cfg: GA4Config):
    """
    Returns BetaAnalyticsDataClient.
    Imports are inside to keep module import light.
    """
    from google.oauth2 import service_account
    from google.analytics.data_v1beta import BetaAnalyticsDataClient

    creds = service_account.Credentials.from_service_account_info(
        cfg.service_account_info,
        scopes=list(cfg.scopes),
    )
    return BetaAnalyticsDataClient(credentials=creds)


# ----------------------------
# URL / Path parsing
# ----------------------------

INVISIBLE = ("\ufeff", "\u200b", "\u2060", "\u00a0")


def _clean(s: Any) -> str:
    s = "" if s is None else str(s)
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    return s.strip()


def _strip_utm_and_fragment(raw_url: str) -> str:
    p = urlparse(raw_url)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))


def _looks_like_domain_no_scheme(s: str) -> bool:
    s = s.strip()
    if not s or s.startswith("/"):
        return False
    head = s.split("/")[0]
    return (" " not in s) and ("." in head) and (":" not in head)


def _normalize_input_to_path_host(raw: str) -> Tuple[str, Optional[str]]:
    """
    Accepts:
      - https://domain/path
      - domain/path
      - /path
      - path
    Returns:
      ("/path", "domain") OR ("/path", None)
    """
    s = _clean(raw)
    if not s:
        return "", None

    if _looks_like_domain_no_scheme(s):
        s = "https://" + s

    if s.lower().startswith(("http://", "https://")):
        s2 = _strip_utm_and_fragment(s)
        p = urlparse(s2)
        host = p.hostname or None
        path = p.path or "/"
        if not path.startswith("/"):
            path = "/" + path
        return path, host

    # treat as path
    if not s.startswith("/"):
        s = "/" + s
    return s, None


def collect_paths_hosts(lines: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      unique_paths: list[str]  (for GA4 filter)
      hosts: list[str]         (if extracted from URLs)
      order_paths: list[str]   (original unique order for stable output ordering)
    """
    seen_path = set()
    order_paths: List[str] = []
    hosts_set = set()

    for raw in lines:
        path, host = _normalize_input_to_path_host(raw)
        if not path:
            continue
        if path not in seen_path:
            seen_path.add(path)
            order_paths.append(path)
        if host:
            hosts_set.add(host.lower())

    unique_paths = list(seen_path)
    hosts = sorted(hosts_set)
    return unique_paths, hosts, order_paths


# ----------------------------
# GA4 Data API helpers
# ----------------------------

def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _run_report_df(client, property_id: str, dimensions: List[str], metrics: List[str],
                   start_date: str, end_date: str,
                   dim_filter_expression=None,
                   order_by_metric: Optional[str] = None,
                   desc: bool = True,
                   limit: Optional[int] = None) -> pd.DataFrame:
    from google.analytics.data_v1beta.types import (
        DateRange, Dimension, Metric, RunReportRequest, OrderBy
    )

    req = RunReportRequest(
        property=f"properties/{property_id}",
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name=d) for d in dimensions],
        metrics=[Metric(name=m) for m in metrics],
    )

    if dim_filter_expression is not None:
        req.dimension_filter = dim_filter_expression

    if order_by_metric:
        req.order_bys = [OrderBy(metric=OrderBy.MetricOrderBy(metric_name=order_by_metric), desc=desc)]

    if limit is not None:
        req.limit = int(limit)

    resp = client.run_report(req)

    cols = [d.name for d in resp.dimension_headers] + [m.name for m in resp.metric_headers]
    rows = []
    for r in resp.rows:
        dim_vals = [v.value for v in r.dimension_values]
        met_vals = [v.value for v in r.metric_values]
        rows.append(dim_vals + met_vals)

    df = pd.DataFrame(rows, columns=cols)

    # Cast numeric metrics
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    return df


def _make_inlist_filter(dim_name: str, values: List[str]):
    from google.analytics.data_v1beta.types import Filter, FilterExpression
    return FilterExpression(
        filter=Filter(
            field_name=dim_name,
            in_list_filter=Filter.InListFilter(values=values),
        )
    )


def _make_and_filter(exprs: List[Any]):
    from google.analytics.data_v1beta.types import FilterExpression
    if len(exprs) == 1:
        return exprs[0]
    return FilterExpression(and_group=FilterExpression.List(expressions=exprs))


def fetch_ga4_by_paths(
    client,
    property_id: str,
    paths_in: List[str],
    hosts_in: List[str],
    start_date: str,
    end_date: str,
    order_keys: Optional[List[str]] = None,
    chunk_size: int = 50,
) -> pd.DataFrame:
    """
    URL Analytics:
    Returns DataFrame with:
      pagePath, pageTitle, views, activeUsers, sessions, engagementRate, engagementTime,
      avgSessionDuration_sec
    """
    # What we collect (as requested)
    dimensions = ["pagePath", "pageTitle"]
    metrics = ["views", "activeUsers", "sessions", "engagementRate", "engagementTime"]

    dfs: List[pd.DataFrame] = []

    # Build host filter once (optional)
    host_expr = _make_inlist_filter("hostName", hosts_in) if hosts_in else None

    # Run in chunks for many paths
    paths_in = [p for p in paths_in if p]
    if not paths_in:
        return pd.DataFrame(columns=dimensions + metrics + ["avgSessionDuration_sec"])

    for part in _chunks(paths_in, chunk_size):
        path_expr = _make_inlist_filter("pagePath", part)
        exprs = [path_expr]
        if host_expr is not None:
            exprs.append(host_expr)
        dim_filter = _make_and_filter(exprs)

        df = _run_report_df(
            client=client,
            property_id=property_id,
            dimensions=dimensions,
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            dim_filter_expression=dim_filter,
            order_by_metric=None,
            limit=None,
        )
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=dimensions + metrics + ["avgSessionDuration_sec"])

    out = pd.concat(dfs, ignore_index=True)

    # Aggregate duplicates (can happen across chunks or because titles change)
    agg = {
        "views": "sum",
        "activeUsers": "sum",
        "sessions": "sum",
        "engagementTime": "sum",
        # engagementRate нельзя суммировать напрямую — пересчитаем позже
        "engagementRate": "mean",
        "pageTitle": "last",
    }
    out = out.groupby("pagePath", as_index=False).agg(agg)

    # Recompute engagementRate more reasonably if possible:
    # engagedSessions not requested; keep mean as pragmatic.
    # (If you want exact: request engagedSessions and compute engagedSessions/sessions)
    # Compute avg session duration
    out["avgSessionDuration_sec"] = (out["engagementTime"] / out["sessions"]).replace([float("inf"), -float("inf")], 0).fillna(0)

    # Stable ordering by input order if provided
    if order_keys:
        order_map = {p: i for i, p in enumerate(order_keys)}
        out["_ord"] = out["pagePath"].map(order_map).fillna(10**9).astype(int)
        out = out.sort_values(["_ord", "views"], ascending=[True, False]).drop(columns=["_ord"])
    else:
        out = out.sort_values("views", ascending=False)

    return out.reset_index(drop=True)


def fetch_top_materials(
    client,
    property_id: str,
    start_date: str,
    end_date: str,
    limit: int = 10,
    host_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Top materials:
    Returns DataFrame with:
      pagePath, views, activeUsers, sessions, engagementRate, engagementTime, avgSessionDuration_sec
    """
    dimensions = ["pagePath"]
    metrics = ["views", "activeUsers", "sessions", "engagementRate", "engagementTime"]

    dim_filter = None
    if host_filter:
        dim_filter = _make_inlist_filter("hostName", host_filter)

    df = _run_report_df(
        client=client,
        property_id=property_id,
        dimensions=dimensions,
        metrics=metrics,
        start_date=start_date,
        end_date=end_date,
        dim_filter_expression=dim_filter,
        order_by_metric="views",
        desc=True,
        limit=int(limit),
    )

    if df.empty:
        df["avgSessionDuration_sec"] = []
        return df

    df["avgSessionDuration_sec"] = (df["engagementTime"] / df["sessions"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    return df


def fetch_site_totals(
    client,
    property_id: str,
    start_date: str,
    end_date: str,
) -> Dict[str, float]:
    """
    Totals:
    Must return keys:
      views, sessions, totalUsers, activeUsers
    """
    df = _run_report_df(
        client=client,
        property_id=property_id,
        dimensions=[],
        metrics=["views", "sessions", "totalUsers", "activeUsers"],
        start_date=start_date,
        end_date=end_date,
        dim_filter_expression=None,
        order_by_metric=None,
        limit=1,
    )

    if df.empty:
        return {"views": 0, "sessions": 0, "totalUsers": 0, "activeUsers": 0}

    row = df.iloc[0].to_dict()
    # ensure numeric
    return {
        "views": float(row.get("views") or 0),
        "sessions": float(row.get("sessions") or 0),
        "totalUsers": float(row.get("totalUsers") or 0),
        "activeUsers": float(row.get("activeUsers") or 0),
    }
