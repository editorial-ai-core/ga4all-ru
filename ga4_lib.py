# ga4_lib.py
# GA4 helper library for Streamlit apps (Service Account + GA4 Data API)
# Contract: always expose "views" to the app/UI.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Filter,
    FilterExpression,
    Metric,
    OrderBy,
    RunReportRequest,
)
from google.oauth2 import service_account


# ---------------------------
# Errors / Config
# ---------------------------

class GA4ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class GA4Config:
    property_id: str
    service_account_info: Dict[str, Any]
    scopes: Tuple[str, ...] = ("https://www.googleapis.com/auth/analytics.readonly",)


def build_config_from_streamlit_secrets(secrets: Any) -> GA4Config:
    '''
    Expected Streamlit secrets keys:

    GA4_PROPERTY_ID = "123456789"
    [gcp_service_account]
    type="service_account"
    project_id="..."
    private_key_id="..."
    private_key="""-----BEGIN PRIVATE KEY-----
    ...
    -----END PRIVATE KEY-----"""
    client_email="..."
    token_uri="https://oauth2.googleapis.com/token"

    Notes:
    - st.secrets behaves like a dict-like object.
    - private_key can be multi-line (preferred) OR contain \\n escapes.
    '''
    try:
        root = dict(secrets)
    except Exception:
        root = secrets  # type: ignore

    prop = root.get("GA4_PROPERTY_ID") or root.get("ga4_property_id") or root.get("property_id")
    if not prop or not isinstance(prop, str):
        raise GA4ConfigError("Missing secret: GA4_PROPERTY_ID (str)")
    prop = prop.strip()

    sa = root.get("gcp_service_account")
    if sa is None:
        raise GA4ConfigError("Missing secret: gcp_service_account (dict)")
    if not isinstance(sa, dict):
        try:
            sa = dict(sa)
        except Exception:
            raise GA4ConfigError("Bad secret type: gcp_service_account must be dict")

    pk = sa.get("private_key")
    if isinstance(pk, str):
        if "\\n" in pk and "-----BEGIN PRIVATE KEY-----" in pk:
            sa["private_key"] = pk.replace("\\n", "\n")

    for k in ("type", "client_email", "token_uri", "private_key"):
        if k not in sa or not isinstance(sa.get(k), str) or not str(sa.get(k)).strip():
            raise GA4ConfigError(f"Missing secret inside [gcp_service_account]: {k} (str)")

    return GA4Config(property_id=prop, service_account_info=sa)


# ---------------------------
# Client
# ---------------------------

def make_client(cfg: GA4Config) -> BetaAnalyticsDataClient:
    creds = service_account.Credentials.from_service_account_info(
        cfg.service_account_info,
        scopes=list(cfg.scopes),
    )
    return BetaAnalyticsDataClient(credentials=creds)


# ---------------------------
# URL parsing helpers
# ---------------------------

def collect_paths_hosts(lines: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Takes list of strings (URLs or paths), returns:
      unique_paths: unique pagePath values for filtering
      hostnames: unique hostName values (if URLs were provided)
      order_paths: original order (normalized paths) to restore output ordering
    """
    paths: List[str] = []
    hosts: List[str] = []

    for raw in lines:
        s = (raw or "").strip()
        if not s:
            continue

        if s.startswith("http://") or s.startswith("https://"):
            u = urlparse(s)
            h = (u.hostname or "").strip()
            if h:
                hosts.append(h.lower())
            p = u.path or "/"
            if not p.startswith("/"):
                p = "/" + p
            paths.append(p)
        else:
            p2 = s
            if not p2.startswith("/"):
                p2 = "/" + p2
            paths.append(p2)

    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    order_paths = paths[:]  # can include duplicates
    return uniq(paths), uniq(hosts), order_paths


# ---------------------------
# GA4 Data API helpers
# ---------------------------

def _chunks(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    buf: List[str] = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _inlist_expr(field_name: str, values: Sequence[str]) -> FilterExpression:
    """
    InListFilter type differs across google-analytics-data versions:
    - sometimes it's only Filter.InListFilter (no top-level InListFilter import)
    - case_sensitive may or may not exist
    """
    try:
        in_list = Filter.InListFilter(values=list(values), case_sensitive=False)
    except TypeError:
        in_list = Filter.InListFilter(values=list(values))

    return FilterExpression(
        filter=Filter(
            field_name=field_name,
            in_list_filter=in_list,
        )
    )


def _and_expr(parts: List[Optional[FilterExpression]]) -> Optional[FilterExpression]:
    parts2 = [p for p in parts if p is not None]
    if not parts2:
        return None
    if len(parts2) == 1:
        return parts2[0]
    return FilterExpression(and_group=FilterExpression.ListExpression(expressions=parts2))


def _map_order_bys_metric(order_bys: Optional[List[OrderBy]], metric_map: Dict[str, str]) -> Optional[List[OrderBy]]:
    if not order_bys:
        return order_bys

    out: List[OrderBy] = []
    for ob in order_bys:
        if ob.metric and ob.metric.metric_name in metric_map:
            out.append(
                OrderBy(
                    metric=OrderBy.MetricOrderBy(metric_name=metric_map[ob.metric.metric_name]),
                    desc=ob.desc,
                )
            )
        else:
            out.append(ob)
    return out


def _run_report(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
    dimensions: List[str],
    metrics: List[str],
    dimension_filter: Optional[FilterExpression] = None,
    limit: int = 100000,
    order_bys: Optional[List[OrderBy]] = None,
) -> pd.DataFrame:
    """
    Always returns DataFrame with exactly the metric names requested in `metrics`.
    If the backend/SDK doesn't accept 'views', we transparently retry with the compatible name,
    but map the output column back to 'views'.
    """
    def run_once(metrics_for_request: List[str], order_bys_for_request: Optional[List[OrderBy]]) -> pd.DataFrame:
        req = RunReportRequest(
            property=f"properties/{property_id}",
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            dimensions=[Dimension(name=d) for d in dimensions],
            metrics=[Metric(name=m) for m in metrics_for_request],
            limit=limit,
        )
        if dimension_filter is not None:
            req.dimension_filter = dimension_filter
        if order_bys_for_request:
            req.order_bys = order_bys_for_request

        resp = client.run_report(req)

        cols = list(dimensions) + list(metrics_for_request)
        rows: List[List[Any]] = []
        for r in resp.rows:
            dvals = [dv.value for dv in r.dimension_values]
            mvals = [mv.value for mv in r.metric_values]
            rows.append(dvals + mvals)

        df_ = pd.DataFrame(rows, columns=cols)

        for m in metrics_for_request:
            if m in df_.columns:
                df_[m] = pd.to_numeric(df_[m], errors="coerce")
        return df_

    # First try with requested names (we want 'views' to stay 'views')
    try:
        df = run_once(metrics, order_bys)
        return df
    except Exception as e:
        msg = str(e).lower()

        # Fallback: some environments accept screenPageViews instead of views
        if "views" in metrics and ("metric" in msg or "invalid" in msg or "not valid" in msg or "unknown" in msg):
            metric_map = {"views": "screenPageViews"}
            metrics2 = [metric_map.get(m, m) for m in metrics]
            order_bys2 = _map_order_bys_metric(order_bys, metric_map)

            df2 = run_once(metrics2, order_bys2)

            # Map back to requested metric names
            df2 = df2.rename(columns={"screenPageViews": "views"})
            return df2

        # If it's not about views metric, bubble up
        raise


# ---------------------------
# Public API (contract aligned with streamlit_app.py)
# ---------------------------

def fetch_ga4_by_paths(
    client: BetaAnalyticsDataClient,
    property_id: str,
    paths_in: Sequence[str],
    hosts_in: Sequence[str],
    start_date: str,
    end_date: str,
    order_keys: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Output columns (UI expects):
      pagePath, views, activeUsers, sessions, engagementRate, avgSessionDuration_sec
    """
    dims = ["pagePath"]
    mets = ["views", "activeUsers", "sessions", "engagedSessions", "averageSessionDuration"]

    host_expr: Optional[FilterExpression] = None
    if hosts_in:
        host_expr = _inlist_expr("hostName", list(hosts_in))

    frames: List[pd.DataFrame] = []
    for part in _chunks(list(paths_in), 100):
        path_expr = _inlist_expr("pagePath", part)
        filt = _and_expr([path_expr, host_expr])

        df = _run_report(
            client=client,
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
            dimensions=dims,
            metrics=mets,
            dimension_filter=filt,
            limit=100000,
        )
        frames.append(df)

    if not frames:
        out = pd.DataFrame(columns=dims + mets)
    else:
        out = pd.concat(frames, ignore_index=True)

    if not out.empty:
        out["_asd_x_sess"] = (out["averageSessionDuration"] * out["sessions"]).fillna(0)

        agg = {
            "views": "sum",
            "sessions": "sum",
            "engagedSessions": "sum",
            "activeUsers": "max",
            "_asd_x_sess": "sum",
        }
        out = out.groupby("pagePath", as_index=False).agg(agg)

        denom_sessions = out["sessions"].where(out["sessions"] > 0, 1)
        out["avgSessionDuration_sec"] = (out["_asd_x_sess"] / denom_sessions).fillna(0)
        out["engagementRate"] = (out["engagedSessions"] / denom_sessions).fillna(0)

        out = out.drop(columns=["_asd_x_sess"])

    if order_keys:
        order_index = {p: i for i, p in enumerate(list(order_keys))}
        out["_ord"] = out["pagePath"].map(lambda x: order_index.get(x, 10**9))
        out = out.sort_values(["_ord", "views"], ascending=[True, False]).drop(columns=["_ord"])
    else:
        if "views" in out.columns:
            out = out.sort_values("views", ascending=False)

    keep = ["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"]
    return out.reindex(columns=keep)


def fetch_top_materials(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Output columns (UI expects):
      pagePath, views, activeUsers, sessions, engagementRate, avgSessionDuration_sec
    """
    dims = ["pagePath"]
    mets = ["views", "activeUsers", "sessions", "engagedSessions", "averageSessionDuration"]

    order_bys = [OrderBy(metric=OrderBy.MetricOrderBy(metric_name="views"), desc=True)]

    df = _run_report(
        client=client,
        property_id=property_id,
        start_date=start_date,
        end_date=end_date,
        dimensions=dims,
        metrics=mets,
        dimension_filter=None,
        limit=int(limit),
        order_bys=order_bys,
    )

    if df.empty:
        out = pd.DataFrame(columns=["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"])
        return out

    denom = df["sessions"].where(df["sessions"] > 0, 1)
    df["engagementRate"] = (df["engagedSessions"] / denom).fillna(0)
    df["avgSessionDuration_sec"] = df["averageSessionDuration"].fillna(0)

    keep = ["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"]
    return df.reindex(columns=keep)


def fetch_site_totals(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
) -> Dict[str, float]:
    """
    Output keys (UI expects):
      views, sessions, totalUsers, activeUsers
    """
    df = _run_report(
        client=client,
        property_id=property_id,
        start_date=start_date,
        end_date=end_date,
        dimensions=[],
        metrics=["views", "sessions", "totalUsers", "activeUsers"],
        dimension_filter=None,
        limit=1,
    )

    if df.empty:
        return {"views": 0.0, "sessions": 0.0, "totalUsers": 0.0, "activeUsers": 0.0}

    row = df.iloc[0].to_dict()
    return {
        "views": float(row.get("views") or 0),
        "sessions": float(row.get("sessions") or 0),
        "totalUsers": float(row.get("totalUsers") or 0),
        "activeUsers": float(row.get("activeUsers") or 0),
    }


# Optional: UI-friendly Russian labels for contract fields
RUS_METRIC_LABELS = {
    "views": "Просмотры",
    "activeUsers": "Активные пользователи",
    "sessions": "Сессии",
    "engagementRate": "Доля вовлечённых сессий",
    "avgSessionDuration_sec": "Средняя длительность сеанса",
}
