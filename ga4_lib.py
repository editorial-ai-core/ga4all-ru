# ga4_lib.py
# GA4 helper library for Streamlit apps (Service Account + GA4 Data API)
# ВАЖНО: без InListFilter (в некоторых версиях google-analytics-data его нет)

import pandas as pd
from urllib.parse import urlparse

from google.oauth2 import service_account
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


class GA4ConfigError(RuntimeError):
    pass


def build_config_from_streamlit_secrets(secrets):
    """
    Streamlit Secrets ожидаются так:

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

    pk = sa.get("private_key")
    if not isinstance(pk, str) or not pk.strip():
        raise GA4ConfigError("Missing secret inside [gcp_service_account]: private_key (str)")
    # если ключ пришёл с \n — превращаем в реальные переносы строк
    if "\\n" in pk and "-----BEGIN PRIVATE KEY-----" in pk:
        sa["private_key"] = pk.replace("\\n", "\n")

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


def _exact_or_expr(field_name, values):
    """
    Делает OR-группу из точных совпадений:
      field == v1 OR field == v2 OR ...
    Работает везде, не зависит от InListFilter.
    """
    vals = [v for v in (values or []) if isinstance(v, str) and v.strip()]
    if not vals:
        return None

    exprs = []
    for v in vals:
        exprs.append(
            FilterExpression(
                filter=Filter(
                    field_name=field_name,
                    string_filter=Filter.StringFilter(
                        value=v,
                        match_type=Filter.StringFilter.MatchType.EXACT,
                        case_sensitive=False,
                    ),
                )
            )
        )

    if len(exprs) == 1:
        return exprs[0]

    return FilterExpression(or_group=FilterExpression.ListExpression(expressions=exprs))


def _and_expr(parts):
    parts2 = [p for p in parts if p is not None]
    if not parts2:
        return None
    if len(parts2) == 1:
        return parts2[0]
    return FilterExpression(and_group=FilterExpression.ListExpression(expressions=parts2))


def _run_report(
    client,
    property_id,
    start_date,
    end_date,
    dimensions,
    metrics,
    dimension_filter=None,
    limit=100000,
    order_bys=None,
):
    req = RunReportRequest(
        property="properties/%s" % property_id,
        date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        dimensions=[Dimension(name=d) for d in (dimensions or [])],
        metrics=[Metric(name=m) for m in (metrics or [])],
        limit=int(limit),
    )
    if dimension_filter is not None:
        req.dimension_filter = dimension_filter
    if order_bys:
        req.order_bys = order_bys

    resp = client.run_report(req)

    cols = list(dimensions or []) + list(metrics or [])
    rows = []

    for r in resp.rows:
        dvals = [dv.value for dv in r.dimension_values] if dimensions else []
        mvals = [mv.value for mv in r.metric_values] if metrics else []
        rows.append(dvals + mvals)

    df = pd.DataFrame(rows, columns=cols)

    for m in (metrics or []):
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    return df


def fetch_ga4_by_paths(client, property_id, paths_in, hosts_in, start_date, end_date, order_keys=None):
    """
    URL Analytics (по конкретным путям).

    Метрики:
      Views         -> screenPageViews
      Active users  -> activeUsers
      Sessions      -> sessions
      engagementRate -> engagementRate
      engagementTime/sessions -> userEngagementDuration / sessions (сек)
    """
    dims = ["pagePath", "pageTitle"]
    mets = ["screenPageViews", "activeUsers", "sessions", "engagementRate", "userEngagementDuration"]

    all_frames = []

    host_expr = _exact_or_expr("hostName", list(hosts_in or [])) if hosts_in else None

    for part in _chunks(list(paths_in or []), 50):
        # 50 чтобы OR-фильтр не стал слишком большим
        path_expr = _exact_or_expr("pagePath", part)
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
        out = out.groupby(["pagePath", "pageTitle"], as_index=False)[mets].sum(numeric_only=True)

    out["avgEngagementTime_sec"] = (out["userEngagementDuration"] / out["sessions"]).fillna(0)

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


def fetch_top_materials(client, property_id, start_date, end_date, limit=10
