# ga4_lib.py
# GA4 helper library for Streamlit apps (Service Account + GA4 Data API)

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
    InListFilter,
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


def _require_key(d: Any, key: str, expected_type: type, hint: str = "") -> Any:
    if not isinstance(d, dict) or key not in d:
        raise GA4ConfigError(f"Missing secret: {key} ({expected_type.__name__})")
    v = d.get(key)
    if not isinstance(v, expected_type):
        raise GA4ConfigError(f"Bad secret type: {key} must be {expected_type.__name__}")
    if hint:
        # hint is not shown by default, only useful during debugging
        pass
    return v


def build_config_from_streamlit_secrets(secrets: Any) -> GA4Config:
    """
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
    - st.secrets behaves like a dict-like object. We only rely on dict operations.
    - private_key can be multi-line (preferred) OR contain \n escapes.
    """
    # secrets may be a special Streamlit object; convert shallowly via dict()
    try:
        root = dict(secrets)
    except Exception:
        # last resort: assume it's already dict-like
        root = secrets  # type: ignore

    # property id
    prop = root.get("GA4_PROPERTY_ID") or root.get("ga4_property_id") or root.get("property_id")
    if not prop or not isinstance(prop, str):
        raise GA4ConfigError("Missing secret: GA4_PROPERTY_ID (str)")
    prop = prop.strip()

    # service account dict
    sa = root.get("gcp_service_account")
    if sa is None:
        raise GA4ConfigError("Missing secret: gcp_service_account (dict)")
    if not isinstance(sa, dict):
        # sometimes Streamlit keeps nested objects; try converting
        try:
            sa = dict(sa)
        except Exception:
            raise GA4ConfigError("Bad secret type: gcp_service_account must be dict")

    # normalize private_key (allow either real multiline or \n escapes)
    pk = sa.get("private_key")
    if isinstance(pk, str):
        # If user pasted with "\n" escapes, convert to real newlines
        if "\\n" in pk and "-----BEGIN PRIVATE KEY-----" in pk:
            sa["private_key"] = pk.replace("\\n", "\n")

    # minimal fields validation (don't be too strict)
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
            if u.netloc:
                hosts.append(u.netloc.lower())
            p = u.path or "/"
            if not p.startswith("/"):
                p = "/" + p
            paths.append(p)
        else:
            # treat as path
            p2 = s
            if not p2.startswith("/"):
                p2 = "/" + p2
            paths.append(p2)

    # unique while preserving order
    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    order_paths = paths[:]  # keep original ordering
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
    return FilterExpression(
        filter=Filter(
            field_name=field_name,
            in_list_filter=InListFilter(values=list(values), case_sensitive=False),
        )
    )


def _and_expr(parts: List[FilterExpression]) -> Optional[FilterExpression]:
    parts2 = [p for p in parts if p is not None]
    if not parts2:
        return None
    if len(parts2) == 1:
        return parts2[0]
    return FilterExpression(and_group=FilterExpression.ListExpression(expressions=parts2))


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
    req = RunReportRequest(
        property="properties/%s" % property_id,
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name=d) for d in dimensions],
        metrics=[Metric(name=m) for m in metrics],
        limit=limit,
    )
    if dimension_filter is not None:
        req.dimension_filter = dimension_filter
    if order_bys:
        req.order_bys = order_bys

    resp = client.run_report(req)

    cols = list(dimensions) + list(metrics)
    rows: List[List[Any]] = []

    for r in resp.rows:
        dvals = [dv.value for dv in r.dimension_values]
        mvals = [mv.value for mv in r.metric_values]
        rows.append(dvals + mvals)

    df = pd.DataFrame(rows, columns=cols)

    # coerce numeric metrics
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    return df


# ---------------------------
# Public API
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
    Returns per-page table for given paths (+ optional hostName filter).
    Metrics:
      - screenPageViews  (Views)
      - activeUsers      (Active users)
      - sessions         (Sessions)
      - engagementRate   (Engaged sessions / Sessions)
      - userEngagementDuration (seconds) -> to compute engagementTime/sessions
    """

    dims = ["pagePath", "pageTitle"]
    mets = ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"]

    # GA4 InListFilter has practical limits; chunk paths
    all_frames: List[pd.DataFrame] = []

    # host filter (if provided)
    host_expr: Optional[FilterExpression] = None
    if hosts_in:
        host_expr = _inlist_expr("hostName", list(hosts_in))

    # IMPORTANT: use pagePath filter; if you need query string, switch to pagePathPlusQueryString
    for part in _chunks(list(paths_in), 100):
        path_expr = _inlist_expr("pagePath", part)
        filt = _and_expr([path_expr, host_expr] if host_expr is not None else [path_expr])

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
        all_frames.append(df)

    if not all_frames:
        out = pd.DataFrame(columns=dims + mets)
    else:
        out = pd.concat(all_frames, ignore_index=True)
        # aggregate duplicates across chunks (same path/title)
        grp = out.groupby(["pagePath", "pageTitle"], as_index=False)[mets].sum(numeric_only=True)
        out = grp

    # compute engagementTime/sessions (Average engagement time per session, seconds)
    out["avgEngagementTime_sec"] = (out["userEngagementDuration"] / out["sessions"]).replace([pd.NA, pd.NaT], 0)
    out["avgEngagementTime_sec"] = out["avgEngagementTime_sec"].fillna(0)

    # optional: restore ordering by order_keys (paths)
    if order_keys:
        order_index = {p: i for i, p in enumerate(list(order_keys))}
        out["_ord"] = out["pagePath"].map(lambda x: order_index.get(x, 10**9))
        out = out.sort_values(["_ord", "screenPageViews"], ascending=[True, False]).drop(columns=["_ord"])
    else:
        out = out.sort_values("screenPageViews", ascending=False)

    # keep only columns we care about
    keep = ["pagePath", "pageTitle", "screenPageViews", "activeUsers", "sessions", "engagementRate", "avgEngagementTime_sec"]
    out = out.reindex(columns=keep)

    return out


def fetch_top_materials(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Top pages by Views with required metrics.
    """
    dims = ["pagePath", "pageTitle"]
    mets = ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"]

    order_bys = [
        OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)
    ]

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

    df["avgEngagementTime_sec"] = (df["userEngagementDuration"] / df["sessions"]).fillna(0)
    keep = ["pagePath", "pageTitle", "screenPageViews", "activeUsers", "sessions", "engagementRate", "avgEngagementTime_sec"]
    return df.reindex(columns=keep)


def fetch_site_totals(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
) -> Dict[str, float]:
    """
    Site totals for the period.
    You said you need (left->right):
      Views (Page Views) = screenPageViews
      Sessions           = sessions
      totalUsers         = totalUsers (Unique Users)
      activeUsers        = activeUsers
    """
    df = _run_report(
        client=client,
        property_id=property_id,
        start_date=start_date,
        end_date=end_date,
        dimensions=[],
        metrics=["screenPageViews", "sessions", "totalUsers", "activeUsers"],
        dimension_filter=None,
        limit=1,
    )

    if df.empty:
        return {"screenPageViews": 0, "sessions": 0, "totalUsers": 0, "activeUsers": 0}

    row = df.iloc[0].to_dict()
    # ensure keys exist
    out = {
        "screenPageViews": float(row.get("screenPageViews") or 0),
        "sessions": float(row.get("sessions") or 0),
        "totalUsers": float(row.get("totalUsers") or 0),
        "activeUsers": float(row.get("activeUsers") or 0),
    }
    return out


# ---------------------------
# Optional: UI-friendly Russian column mapping
# ---------------------------

RUS_METRIC_LABELS = {
    "screenPageViews": "Просмотры",
    "activeUsers": "Активные пользователи",
    "sessions": "Сессии",
    "engagementRate": "Доля вовлечённых сессий",
    "avgEngagementTime_sec": "Средняя длительность сеанса",
}

