# ga4_lib.py
# GA4 helper library for Streamlit apps (Service Account + GA4 Data API)
# максимально совместимая версия (без dataclass/typing/__future__)

import pandas as pd
from urllib.parse import urlparse

from google.oauth2 import service_account
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


class GA4ConfigError(RuntimeError):
    pass


def build_config_from_streamlit_secrets(secrets):
    """
    Ожидается в Streamlit Secrets:

    GA4_PROPERTY_ID = "318772986"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = '''-----BEGIN PRIVATE KEY-----
    ...
    -----END PRIVATE KEY-----'''
    client_email = "..."
    token_uri = "https://oauth2.googleapis.com/token"
    """
    # st.secrets иногда не совсем dict — пробуем привести
    try:
        root = dict(secrets)
    except Exception:
        root = secrets

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

    # нормализуем private_key: если вставлен как \n — превращаем в реальные переводы строк
    pk = sa.get("private_key")
    if not isinstance(pk, str) or not pk.strip():
        raise GA4ConfigError("Missing secret inside [gcp_service_account]: private_key (str)")
    if "\\n" in pk and "-----BEGIN PRIVATE KEY-----" in pk:
        sa["private_key"] = pk.replace("\\n", "\n")

    # минимальная валидация обязательных строковых полей
    for k in ("type", "client_email", "token_uri", "private_key"):
        v = sa.get(k)
        if not isinstance(v, str) or not v.strip():
            raise GA4ConfigError("Missing secret inside [gcp_service_account]: %s (str)" % k)

    return {"property_id": prop, "service_account_info": sa}


def make_client(cfg):
    scopes = ["https://www.googleapis.com/auth/analytics.readonly"]
    creds = service_account.Credentials.from_service_account_info(
        cfg["service_account_info"],
        scopes=scopes,
    )
    return BetaAnalyticsDataClient(credentials=creds)


def collect_paths_hosts(lines):
    """
    Вход: список строк (URL или пути)
    Выход:
      unique_paths, hostnames, order_paths
    """
    paths = []
    hosts = []

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
            p2 = s
            if not p2.startswith("/"):
                p2 = "/" + p2
            paths.append(p2)

    def uniq(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return uniq(paths), uniq(hosts), list(paths)


def _chunks(seq, size):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _inlist_expr(field_name, values):
    return FilterExpression(
        filter=Filter(
            field_name=field_name,
            in_list_filter=InListFilter(values=list(values), case_sensitive=False),
        )
    )


def _and_expr(parts):
    parts2 = [p for p in parts if p is not None]
    if not parts2:
        return None
    if len(parts2) == 1:
        return parts2[0]
    return FilterExpression(and_group=FilterExpression.ListExpression(expressions=parts2))


def _run_report(client, property_id, start_date, end_date, dimensions, metrics, dimension_filter=None, limit=100000, order_bys=None):
    req = RunReportRequest(
        property="properties/%s" % property_id,
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name=d) for d in dimensions],
        metrics=[Metric(name=m) for m in metrics],
        limit=int(limit),
    )
    if dimension_filter is not None:
        req.dimension_filter = dimension_filter
    if order_bys:
        req.order_bys = order_bys

    resp = client.run_report(req)

    cols = list(dimensions) + list(metrics)
    rows = []

    for r in resp.rows:
        dvals = [dv.value for dv in r.dimension_values]
        mvals = [mv.value for mv in r.metric_values]
        rows.append(dvals + mvals)

    df = pd.DataFrame(rows, columns=cols)

    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    return df


def fetch_ga4_by_paths(client, property_id, paths_in, hosts_in, start_date, end_date, order_keys=None):
    """
    Таблица по конкретным путям.
    Метрики (как ты просил):
      Views         -> screenPageViews
      Active users  -> activeUsers
      Sessions      -> sessions
      engagementRate -> engagementRate
      engagementTime/sessions -> userEngagementDuration / sessions (сек)
    """
    dims = ["pagePath", "pageTitle"]
    mets = ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"]

    all_frames = []

    host_expr = None
    if hosts_in:
        host_expr = _inlist_expr("hostName", list(hosts_in))

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
        # если один и тот же путь пришёл из разных чанков — суммируем
        out = out.groupby(["pagePath", "pageTitle"], as_index=False)[mets].sum(numeric_only=True)

    # engagementTime/sessions
    out["avgEngagementTime_sec"] = (out["userEngagementDuration"] / out["sessions"]).fillna(0)

    # сортировка
    if order_keys:
        idx = {}
        i = 0
        for p in list(order_keys):
            if p not in idx:
                idx[p] = i
                i += 1
        out["_ord"] = out["pagePath"].map(lambda x: idx.get(x, 10**9))
        out = out.sort_values(["_ord", "screenPageViews"], ascending=[True, False]).drop(columns=["_ord"])
    else:
        out = out.sort_values("screenPageViews", ascending=False)

    keep = ["pagePath", "pageTitle", "screenPageViews", "activeUsers", "sessions", "engagementRate", "avgEngagementTime_sec"]
    return out.reindex(columns=keep)


def fetch_top_materials(client, property_id, start_date, end_date, limit=10):
    """
    Топ материалов по Views (screenPageViews) с теми же метриками.
    """
    dims = ["pagePath", "pageTitle"]
    mets = ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"]

    order_bys = [OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)]

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


def fetch_site_totals(client, property_id, start_date, end_date):
    """
    Итоги для вкладки "Общие данные по сайту":
    в коде ты хочешь выводить слева направо:
      Views (Page Views) -> screenPageViews
      Sessions          -> sessions
      Unique Users      -> totalUsers
      Active users      -> activeUsers
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
    return {
        "screenPageViews": float(row.get("screenPageViews") or 0),
        "sessions": float(row.get("sessions") or 0),
        "totalUsers": float(row.get("totalUsers") or 0),
        "activeUsers": float(row.get("activeUsers") or 0),
    }


# Для удобства (если захочешь в streamlit_app.py делать rename)
RUS_METRIC_LABELS = {
    "screenPageViews": "Просмотры",
    "activeUsers": "Активные пользователи",
    "sessions": "Сессии",
    "engagementRate": "Доля вовлечённых сессий",
    "avgEngagementTime_sec": "Средняя длительность сеанса",
}
