# ga4_lib.py
# GA4 helper library for Streamlit apps (Service Account + GA4 Data API)
#
# Goal: match GA4 UI "Путь к странице и класс экрана" + "Просмотры" (Views)
# - Dimension in API: unifiedPagePathScreen
# - Views metric in API: screenPageViews
# Output contract for the app:
#   pagePath, views, activeUsers, sessions, engagementRate, avgSessionDuration_sec

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
        # If user pasted with "\n" escapes, convert to real newlines
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
      unique_paths: unique path-like strings for filtering
      hostnames: unique hostName values (if URLs were provided)
      order_paths: original order (normalized paths) to restore output ordering

    Note: we keep hostnames for future use, but for "match GA4 UI" mode we do NOT apply host filter,
    because GA4 UI table in your screenshot isn't filtered by host.
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

    order_paths = paths[:]  # can include duplicates; used only for UI ordering
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
        property=f"properties/{property_id}",
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
