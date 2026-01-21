import streamlit as st
import pandas as pd
from datetime import date, timedelta

from ga4_lib import (
    build_config_from_secrets,
    make_client,
    collect_paths_hosts,
    fetch_ga4_by_paths,
    fetch_top_materials,
    fetch_site_totals,
    GA4ConfigError,
)

st.set_page_config(page_title="Analytics Console", layout="wide")


def fail_ui(msg: str):
    st.error(msg)
    st.stop()


@st.cache_resource
def get_client_and_pid():
    try:
        cfg = build_config_from_secrets(st.secrets)
    except GA4ConfigError as e:
        fail_ui(str(e))

    client = make_client(cfg)
    return client, cfg.property_id


client, pid_default = get_client_and_pid()

st.title("Analytics Console")

with st.sidebar:
    st.markdown("### Период отчета")
    today = date.today()
    date_from = st.date_input("Дата с", value=today - timedelta(days=30))
    date_to = st.date_input("Дата по", value=today)

    st.divider()
    property_id = st.text_input("GA4 Property ID", value=pid_default).strip()

tab1, tab2, tab3 = st.tabs(["URL Analytics", "Топ материалов", "Общие данные по сайту"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("URL Analytics")
    uinput = st.text_area("Вставьте URL или пути (по одному в строке)", height=200)
    lines = [x.strip() for x in (uinput or "").splitlines() if x.strip()]

    if st.button("Собрать"):
        if date_from > date_to:
            fail_ui("Дата «с» должна быть раньше или равна дате «по».")
        if not property_id:
            fail_ui("GA4 Property ID пуст.")
        if not lines:
            fail_ui("Добавьте хотя бы один URL/путь.")

        unique_paths, hostnames, order_paths = collect_paths_hosts(lines)

        df_p = fetch_ga4_by_paths(
            client=client,
            property_id=property_id,
            paths_in=unique_paths,
            hosts_in=hostnames,
            start_date=str(date_from),
            end_date=str(date_to),
            order_keys=order_paths,
        )

        show = df_p.reindex(
            columns=[
                "pagePath",
                "pageTitle",
                "screenPageViews",
                "activeUsers",
                "viewsPerActiveUser",
                "avgEngagementTime_sec",
            ]
        ).rename(
            columns={
                "pagePath": "Путь",
                "pageTitle": "Заголовок",
                "screenPageViews": "Просмотры",
                "activeUsers": "Уникальные пользователи",
                "viewsPerActiveUser": "Просмотры / пользователь",
                "avgEngagementTime_sec": "Average engagement time (сек)",
            }
        )

        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "Скачать CSV",
            show.to_csv(index=False).encode("utf-8"),
            "ga4_url_analytics.csv",
            "text/csv",
        )

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Топ материалов")
    limit = st.number_input("Лимит", min_value=1, max_value=500, value=10)

    if st.button("Собрать топ"):
        if date_from > date_to:
            fail_ui("Дата «с» должна быть раньше или равна дате «по».")
        if not property_id:
            fail_ui("GA4 Property ID пуст.")

        df_top = fetch_top_materials(
            client=client,
            property_id=property_id,
            start_date=str(date_from),
            end_date=str(date_to),
            limit=int(limit),
        )
        st.dataframe(df_top, use_container_width=True, hide_index=True)

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Общие данные по сайту")

    if st.button("Обновить итоги"):
        if date_from > date_to:
            fail_ui("Дата «с» должна быть раньше или равна дате «по».")
        if not property_id:
            fail_ui("GA4 Property ID пуст.")

        totals = fetch_site_totals(
            client=client,
            property_id=property_id,
            start_date=str(date_from),
            end_date=str(date_to),
        )

        # totals is a 1-row DataFrame
        s = int(totals.loc[0, "sessions"]) if not totals.empty else 0
        u = int(totals.loc[0, "totalUsers"]) if not totals.empty else 0
        v = int(totals.loc[0, "screenPageViews"]) if not totals.empty else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", f"{s:,}")
        c2.metric("Unique Users", f"{u:,}")
        c3.metric("Page Views", f"{v:,}")
