import re
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    FilterExpression,
    Filter,
    RunReportRequest,
    OrderBy,
)
from google.oauth2 import service_account


# ----------------------------
# Errors / Config
# ----------------------------
class GA4ConfigError(Exception):
    pass


@dataclass
class GA4Config:
    property_id: str
    service_account_info: Dict[str, Any]


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Streamlit secrets иногда дают table/dict, иногда строку JSON."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # пробуем JSON
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _fix_private_key(sa: Dict[str, Any]) -> Dict[str, Any]:
    """Приводим private_key к виду с реальными переводами строк."""
    sa = dict(sa)
    pk = sa.get("private_key")
    if isinstance(pk, str):
        # если прилетело с \n внутри строки
        sa["private_key"] = pk.replace("\\n", "\n")
    return sa


def build_config_from_streamlit_secrets(secrets: Any) -> GA4Config:
    """
    Ожидаем в secrets:
      GA4_PROPERTY_ID = "123"
      [gcp_service_account]
      type="service_account"
      ...
    Дополнительно принимаем варианты ключей property id:
      ga4_property_id / property_id / GA4_PROPERTY_ID
    """
    if secrets is None:
        raise GA4ConfigError("Secrets пустые: добавь GA4_PROPERTY_ID и [gcp_service_account].")

    # Property ID: несколько возможных имён
    pid = None
    for k in ("GA4_PROPERTY_ID", "ga4_property_id", "property_id", "GA4PropertyID"):
        try:
            v = secrets.get(k)  # streamlit secrets behaves like dict
        except Exception:
            v = None
        if isinstance(v, str) and v.strip():
            pid = v.strip()
            break

    if not pid:
        raise GA4ConfigError("Missing secret: GA4_PROPERTY_ID (string)")

    # Service account table
    try:
        raw_sa = secrets.get("gcp_service_account")
    except Exception:
        raw_sa = None

    sa = _as_dict(raw_sa)
    if not sa:
        raise GA4ConfigError("Missing secret: gcp_service_account (dict)")

    sa = _fix_private_key(sa)

    if sa.get("type") != "service_account":
        raise GA4ConfigError("gcp_service_account.type должен быть 'service_account'")

    if not sa.get("client_email") or not sa.get("private_key"):
        raise GA4ConfigError("gcp_service_account должен содержать минимум client_email и private_key")

    return GA4Config(property_id=pid, service_account_info=sa)


def make_client(cfg: GA4Config) -> BetaAnalyticsDataClient:
    scopes = ["https://www.googleapis.com/auth/analytics.readonly"]
    creds = service_account.Credentials.from_service_account_info(
        cfg.service_account_info, scopes=scopes
    )
    return BetaAnalyticsDataClient(credentials=creds)


# ----------------------------
# URL parsing helpers
# ----------------------------
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _extract_host_and_path(line: str) -> Tuple[Optional[str], str]:
    """
    Принимаем:
      - полный URL https://example.com/path?a=b
      - путь /path
      - путь без слэша path
    Возвращаем (hostname|None, /path)
    """
    s = (line or "").strip()
    if not s:
        return None, ""

    if _URL_RE.match(s):
        # простая разборка без urlparse чтобы избежать “кривых” случаев — достаточно для GA4
        s2 = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
        parts = s2.split("/", 1)
        host = parts[0].strip().lower()
        path = "/" + parts[1] if len(parts) > 1 else "/"
        # убираем query/fragment
        path = path.split("?", 1)[0].split("#", 1)[0]
        if not path.startswith("/"):
            path = "/" + path
        return host, path

    # не URL — считаем что это path
    path = s.split("?", 1)[0].split("#", 1)[0]
    if not path.startswith("/"):
        path = "/" + path
    return None, path


def collect_paths_hosts(lines: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Возвращает:
      unique_paths: уникальные пути
      hostnames: уникальные хосты (если были URL)
      order_paths: порядок путей как ввели (для сортировки результата)
    """
    order_paths: List[str] = []
    paths_set = set()
    hosts_set = set()

    for line in lines:
        host, path = _extract_host_and_path(line)
        if not path:
            continue
        order_paths.append(path)

        if path not in paths_set:
            paths_set.add(path)
        if host:
            hosts_set.add(host)

    unique_paths = list(paths_set)
    hostnames = list(hosts_set)

    return unique_paths, hostnames, order_paths


# ----------------------------
# GA4 Query helpers
# ----------------------------
def _in_list_filter(field_name: str, values: Sequence[str]) -> FilterExpression:
    # OR-цепочка: field == v1 OR field == v2 ...
    exprs: List[FilterExpression] = []
    for v in values:
        exprs.append(
            FilterExpression(
                filter=Filter(
                    field_name=field_name,
                    string_filter=Filter.StringFilter(
                        match_type=Filter.StringFilter.MatchType.EXACT,
                        value=v,
                        case_sensitive=False,
                    ),
                )
            )
        )
    return FilterExpression(or_group=FilterExpression.ListExpression(expressions=exprs))


def _and_filters(filters: List[FilterExpression]) -> Optional[FilterExpression]:
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return FilterExpression(and_group=FilterExpression.ListExpression(expressions=filters))


def _rows_to_df(response, dim_names: List[str], metric_names: List[str]) -> pd.DataFrame:
    rows = []
    for r in response.rows:
        dvals = [dv.value for dv in r.dimension_values]
        mvals = [mv.value for mv in r.metric_values]
        row = {}
        for k, v in zip(dim_names, dvals):
            row[k] = v
        for k, v in zip(metric_names, mvals):
            # все метрики приходят строками
            try:
                if "." in v:
                    row[k] = float(v)
                else:
                    row[k] = int(v)
            except Exception:
                row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)


def _add_avg_session_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    avgSessionDuration_sec = userEngagementDuration / sessions
    userEngagementDuration — в секундах (GA4 Data API).
    """
    if "userEngagementDuration" in df.columns and "sessions" in df.columns:
        def safe_div(a, b):
            try:
                b = float(b)
                if b == 0:
                    return 0.0
                return float(a) / b
            except Exception:
                return 0.0

        df["avgSessionDuration_sec"] = [
            safe_div(a, b) for a, b in zip(df["userEngagementDuration"], df["sessions"])
        ]
    else:
        df["avgSessionDuration_sec"] = 0.0
    return df


# ----------------------------
# Public API
# ----------------------------
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
    Для URL Analytics:
      dimensions: pagePath
      metrics: views, activeUsers, sessions, engagementRate, userEngagementDuration
    """
    if not property_id:
        raise GA4ConfigError("property_id пуст")
    if not paths_in:
        return pd.DataFrame(
            columns=["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"]
        )

    dims = [Dimension(name="pagePath")]
    metrics = [
        Metric(name="screenPageViews"),     # Views
        Metric(name="activeUsers"),
        Metric(name="sessions"),
        Metric(name="engagementRate"),
        Metric(name="userEngagementDuration"),
    ]

    filters: List[FilterExpression] = []
    filters.append(_in_list_filter("pagePath", list(paths_in)))

    # hostName фильтр делаем только если реально передали hosts
    if hosts_in:
        hosts = [h.strip().lower() for h in hosts_in if h and h.strip()]
        if hosts:
            filters.append(_in_list_filter("hostName", hosts))

    dimension_filter = _and_filters(filters)

    req = RunReportRequest(
        property="properties/%s" % str(property_id),
        dimensions=dims,
        metrics=metrics,
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimension_filter=dimension_filter,
    )

    resp = client.run_report(req)

    df = _rows_to_df(
        resp,
        dim_names=["pagePath"],
        metric_names=["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"],
    )

    if df.empty:
        df = pd.DataFrame(
            columns=["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"]
        )
        return df

    # Переименовываем screenPageViews -> views
    df = df.rename(columns={"screenPageViews": "views"})

    df = _add_avg_session_duration(df)

    # Переупорядочим согласно вводу
    if order_keys:
        order_map = {}
        idx = 0
        for p in order_keys:
            if p not in order_map:
                order_map[p] = idx
                idx += 1
        df["_ord"] = df["pagePath"].map(lambda x: order_map.get(x, 10**9))
        df = df.sort_values(["_ord", "pagePath"]).drop(columns=["_ord"], errors="ignore")

    # финальный набор
    cols = ["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]


def fetch_top_materials(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Для вкладки "Топ материалов":
      dimension: pagePath
      metrics: views, activeUsers, sessions, engagementRate, userEngagementDuration
      sort: views desc
    """
    dims = [Dimension(name="pagePath")]
    metrics = [
        Metric(name="screenPageViews"),
        Metric(name="activeUsers"),
        Metric(name="sessions"),
        Metric(name="engagementRate"),
        Metric(name="userEngagementDuration"),
    ]

    req = RunReportRequest(
        property="properties/%s" % str(property_id),
        dimensions=dims,
        metrics=metrics,
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        order_bys=[
            OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)
        ],
        limit=int(limit),
    )

    resp = client.run_report(req)

    df = _rows_to_df(
        resp,
        dim_names=["pagePath"],
        metric_names=["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"],
    )

    if df.empty:
        return pd.DataFrame(
            columns=["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"]
        )

    df = df.rename(columns={"screenPageViews": "views"})
    df = _add_avg_session_duration(df)

    cols = ["pagePath", "views", "activeUsers", "sessions", "engagementRate", "avgSessionDuration_sec"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]


def fetch_site_totals(
    client: BetaAnalyticsDataClient,
    property_id: str,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Для вкладки "Общие данные по сайту":
      metrics: views (screenPageViews), sessions, totalUsers, activeUsers
    Возвращаем:
      {"views":..., "sessions":..., "totalUsers":..., "activeUsers":...}
    """
    metrics = [
        Metric(name="screenPageViews"),
        Metric(name="sessions"),
        Metric(name="totalUsers"),
        Metric(name="activeUsers"),
    ]

    req = RunReportRequest(
        property="properties/%s" % str(property_id),
        dimensions=[],
        metrics=metrics,
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
    )

    resp = client.run_report(req)

    if not resp.rows:
        return {"views": 0, "sessions": 0, "totalUsers": 0, "activeUsers": 0}

    row = resp.rows[0]
    m = [mv.value for mv in row.metric_values]

    def as_int(x):
        try:
            return int(float(x))
        except Exception:
            return 0

    return {
        "views": as_int(m[0]),
        "sessions": as_int(m[1]),
        "totalUsers": as_int(m[2]),
        "activeUsers": as_int(m[3]),
    }
