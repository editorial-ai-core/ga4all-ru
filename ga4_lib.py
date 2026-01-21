# -*- coding: utf-8 -*-
"""
ga4_lib.py — reusable GA4 helpers (NO Streamlit dependency)

Ключевая фишка для Streamlit Cloud:
- все google-импорты сделаны "ленивыми" (внутри функций),
  чтобы приложение не падало на старте из-за зависимости.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import numpy as np
import pandas as pd


# ----------------------------
# Config / Auth
# ----------------------------

GA4_READONLY_SCOPES = ("https://www.googleapis.com/auth/analytics.readonly",)


class GA4ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class GA4Config:
    property_id: str
    service_account_info: dict[str, Any]
    scopes: tuple[str, ...] = GA4_READONLY_SCOPES


def build_config_from_secrets(secrets: Any) -> GA4Config:
    """
    secrets: Mapping-like (например, streamlit.secrets)
    Требует:
      GA4_PROPERTY_ID
      gcp_service_account (dict)
    """
    getter = getattr(secrets, "get", None)
    if not callable(getter):
        raise GA4ConfigError("Secrets object must be Mapping-like (have .get())")

    pid = str(getter("GA4_PROPERTY_ID", "")).strip()
    if not pid:
        raise GA4ConfigError("Missing secret: GA4_PROPERTY_ID")

    sa = getter("gcp_service_account", None)
    if not sa or not isinstance(sa, dict):
        raise GA4ConfigError("Missing secret: gcp_service_account (dict)")

    return GA4Config(property_id=pid, service_account_info=dict(sa))


def make_client(cfg: GA4Config):
    """
    Возвращает BetaAnalyticsDataClient. Импорты внутри функции.
    """
    from google.oauth2 import service_account
    from google.analytics.data_v1beta import BetaAnalyticsDataClient

    creds = service_account.Credentials.from_service_account_info(
        cfg.service_account_info,
        scopes=list(cfg.scopes),
    )
    return BetaAnalyticsDataClient(credentials=creds)


# ----------------------------
# URL / Path utils
# ----------------------------

INVISIBLE = ("\ufeff", "\u200b", "\u2060", "\u00a0")

METRICS_PAGE = ("screenPageViews", "activeUsers", "userEngagementDuration")


def clean_line(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    for ch in INVISIBLE:
        s = s.replace(ch, "")
    return s.strip()


def strip_utm_and_fragment(raw_url: str) -> str:
    p = urlparse(raw_url)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))


def looks_like_domain_no_scheme(s: str) -> bool:
    s = s.strip()
    if not s or s.startswith("/"):
        return False
    head = s.split("/")[0]
    return (" " not in s) and ("." in head) and (":" not in head)


def normalize_any_input_to_path_and_host(raw: str) -> tuple[str, Optional[str]]:
    """
    Accepts:
      - https://domain/path...
      - http://domain/path...
      - www.domain/path...
      - domain/path...
      - /path...
      - path...
    Returns:
      path (always starting with "/"), host (if detected)
    """
    s = clean_line(raw)
    if not s:
        return "", None

    # "www.domain/.." or "domain.tld/.." -> add scheme for parsing
    if looks_like_domain_no_scheme(s):
        s = "https://" + s

    if s.lower().startswith(("http://", "https://")):
        s2 = strip_utm_and_fragment(s)
        p = urlparse(s2)
        path = p.path or "/"
        if not path.startswith("/"):
            path = "/" + path
        host = p.hostname or None
        return path, host

    # treat as path
    if not s.startswith("/"):
        s = "/" + s
    return s, None


def collect_paths_hosts(raw_list: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Возвращает:
      unique_paths: уникальные pagePath (для запроса)
      hosts: найденные hostName (если юзер вставил домены)
      order_list: пути в исходном порядке (для reindex)
    """
    seen = set()
    unique_paths: list[str] = []
    hosts = set()
    order_list: list[str] = []

    for raw in raw_list:
        path, host = normalize_any_input_to_path_and_host(raw)
        if not path:
            continue
        order_list.append(path)
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
        if host:
            hosts.add(host)

    return unique_paths, sorted(hosts), order_list


def read_uploaded_lines(uploaded) -> list[str]:
    """
    uploaded: объект типа Streamlit UploadedFile или file-like.
    Поддержка:
      - .txt (по строкам)
      - .csv (берём 1 столбец)
    """
    if uploaded is None:
        return []
    try:
        name = getattr(uploaded, "name", "") or ""
        if name.lower().endswith(".txt"):
            return [clean_line(b.decode("utf-8", errors="ignore")) for b in uploaded.readlines()]
        dfu = pd.read_csv(uploaded, header=None)
        return [clean_line(x) for x in dfu.iloc[:, 0].tolist()]
    except Exception:
        return []


# ----------------------------
# GA4 Queries
# ----------------------------

def _empty_paths_df(with_host: bool) -> pd.DataFrame:
    cols = ["pagePath", "pageTitle"] + (["hostName"] if with_host else []) + list(METRICS_PAGE)
    return pd.DataFrame(columns=cols)


def make_path_filter(paths_batch: list[str]):
    """
    Build FilterExpression OR-group for pagePath begins_with for each path.
    Импорты внутри функции.
    """
    from google.analytics.data_v1beta.types import Filter, FilterExpression, FilterExpressionList

    exprs = [
        FilterExpression(
            filter=Filter(
                field_name="pagePath",
                string_filter=Filter.StringFilter(
                    value=pth,
                    match_type=Filter.StringFilter.MatchType.BEGINS_WITH,
                    case_sensitive=False,
                )
            )
        )
        for pth in paths_batch
    ]
    return FilterExpression(or_group=FilterExpressionList(expressions=exprs))


def fetch_ga4_by_paths(
    client,
    property_id: str,
    paths_in: list[str],
    hosts_in: list[str],
    start_date: str,
    end_date: str,
    order_keys: list[str],
    batch_size: int = 25,
    limit: int = 100000,
) -> pd.DataFrame:
    """
    Возвращает DataFrame по заданным pagePath (с сохранением исходного порядка order_keys).
    """
    from google.analytics.data_v1beta.types import (
        RunReportRequest, Dimension, Metric, Filter, FilterExpression, FilterExpressionList
    )

    if not property_id:
        raise ValueError("property_id is empty")

    want_host = bool(hosts_in)
    if not paths_in:
        return _empty_paths_df(want_host)

    rows: list[dict[str, Any]] = []

    for i in range(0, len(paths_in), batch_size):
        batch = paths_in[i:i + batch_size]
        base = make_path_filter(batch)
        dim_filter = base

        dims = [Dimension(name="pagePath"), Dimension(name="pageTitle")]

        # если юзер вставил домены — ограничим по hostName
        if want_host:
            host_expr = FilterExpression(
                filter=Filter(
                    field_name="hostName",
                    in_list_filter=Filter.InListFilter(values=hosts_in[:50])
                )
            )
            dim_filter = FilterExpression(and_group=FilterExpressionList(expressions=[base, host_expr]))
            dims.append(Dimension(name="hostName"))

        req = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=dims,
            metrics=[Metric(name=m) for m in METRICS_PAGE],
            date_ranges=[{"start_date": start_date, "end_date": end_date}],
            dimension_filter=dim_filter,
            limit=int(limit),
        )

        resp = client.run_report(req)
        for r in resp.rows:
            rec: dict[str, Any] = {}
            idx = 0
            rec["pagePath"] = r.dimension_values[idx].value; idx += 1
            rec["pageTitle"] = r.dimension_values[idx].value; idx += 1
            if want_host:
                rec["hostName"] = r.dimension_values[idx].value
            for j, m in enumerate(METRICS_PAGE):
                rec[m] = r.metric_values[j].value
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        df = _empty_paths_df(want_host)

    # типы
    for m in METRICS_PAGE:
        if m not in df.columns:
            df[m] = 0
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    if "pagePath" not in df.columns:
        df["pagePath"] = ""
    if "pageTitle" not in df.columns:
        df["pageTitle"] = ""

    # агрегируем по pagePath
    agg = {m: "sum" for m in METRICS_PAGE}
    agg["pageTitle"] = "first"
    if want_host:
        if "hostName" not in df.columns:
            df["hostName"] = ""
        agg["hostName"] = "first"

    if (not df.empty) and (df["pagePath"].astype(str).str.len().sum() > 0):
        df = df.groupby(["pagePath"], as_index=False).agg(agg)
    else:
        df = _empty_paths_df(want_host)

    # добавляем нулевые строки для всех запрошенных путей
    present = set(df["pagePath"].tolist()) if not df.empty else set()
    missing_unique = [p for p in paths_in if p not in present]
    if missing_unique:
        base_zero = {
            "pagePath": None,
            "pageTitle": "",
            "screenPageViews": 0,
            "activeUsers": 0,
            "userEngagementDuration": 0,
        }
        zeros = pd.DataFrame([dict(base_zero, **{"pagePath": p}) for p in missing_unique])
        if want_host:
            zeros["hostName"] = hosts_in[0] if hosts_in else ""
        df = pd.concat([df, zeros], ignore_index=True)

    # исходный порядок
    df = df.set_index("pagePath").reindex(order_keys).reset_index()

    # производные метрики
    den = pd.to_numeric(df["activeUsers"], errors="coerce").replace(0, np.nan).astype(float)
    df["viewsPerActiveUser"] = (
        pd.to_numeric(df["screenPageViews"], errors="coerce").astype(float) / den
    ).fillna(0).round(2)
    df["avgEngagementTime_sec"] = (
        pd.to_numeric(df["userEngagementDuration"], errors="coerce").astype(float) / den
    ).fillna(0).round(1)

    return df


def fetch_top_materials(
    client,
    property_id: str,
    start_date: str,
    end_date: str,
    limit: int,
) -> pd.DataFrame:
    """
    Топ материалов по screenPageViews.
    """
    from google.analytics.data_v1beta.types import RunReportRequest, Dimension, Metric, OrderBy

    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath"), Dimension(name="pageTitle")],
        metrics=[Metric(name="screenPageViews"), Metric(name="activeUsers"), Metric(name="userEngagementDuration")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)],
        limit=int(limit),
    )
    resp = client.run_report(req)

    rows = []
    for r in resp.rows:
        views = int(float(r.metric_values[0].value or 0))
        users = int(float(r.metric_values[1].value or 0))
        eng = float(r.metric_values[2].value or 0)
        rows.append({
            "Путь": r.dimension_values[0].value,
            "Заголовок": r.dimension_values[1].value,
            "Просмотры": views,
            "Уникальные пользователи": users,
            "Average engagement time (сек)": round(eng / max(users, 1), 1),
        })
    return pd.DataFrame(rows)


def fetch_site_totals(
    client,
    property_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Общие итоги по сайту (sessions, totalUsers, screenPageViews)
    """
    from google.analytics.data_v1beta.types import RunReportRequest, Metric

    req = RunReportRequest(
        property=f"properties/{property_id}",
        metrics=[Metric(name="sessions"), Metric(name="totalUsers"), Metric(name="screenPageViews")],
        date_ranges=[{"start_date": start_date, "end_date": end_date}],
        limit=1,
    )
    resp = client.run_report(req)

    if not resp.rows:
        return pd.DataFrame([{"sessions": 0, "totalUsers": 0, "screenPageViews": 0}])

    mv = resp.rows[0].metric_values
    return pd.DataFrame([{
        "sessions": int(float(mv[0].value or 0)),
        "totalUsers": int(float(mv[1].value or 0)),
        "screenPageViews": int(float(mv[2].value or 0)),
    }])
