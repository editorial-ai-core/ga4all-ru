# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import base64

from ga4_lib import (
    build_config_from_streamlit_secrets,
    make_client,
    collect_paths_hosts,
    fetch_ga4_by_paths,
    fetch_top_materials,
    fetch_site_totals,
    GA4ConfigError,
)

st.set_page_config(page_title="Analytics Console", layout="wide")


# -------------------------
# Helpers
# -------------------------
ASSETS_DIR = Path(__file__).parent / "assets"
INTERNEWS_LOGO = ASSETS_DIR / "internews.svg"


def fail_ui(msg: str):
    st.error(msg)
    st.stop()


def svg_data_uri(svg_path: Path) -> str:
    svg_bytes = svg_path.read_bytes()
    b64 = base64.b64encode(svg_bytes).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def analytics_icon_svg_uri() -> str:
    # Простой “GA-like” значок (оранжевые бары) как inline-SVG
    svg = """
    <svg width="84" height="84" viewBox="0 0 84 84" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="8" y="48" width="14" height="28" rx="7" fill="#F9AB00"/>
      <rect x="30" y="34" width="14" height="42" rx="7" fill="#F9AB00"/>
      <rect x="52" y="16" width="14" height="60" rx="7" fill="#F9AB00"/>
      <circle cx="15" cy="36" r="7" fill="#F9AB00"/>
    </svg>
    """
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def render_header():
    st.markdown(
        """
        <style>
          .ac-header { margin-top: 8px; margin-bottom: 6px; }
          .ac-title { font-size: 48px; font-weight: 800; line-height: 1.05; margin: 0; }
          .ac-sub { font-size: 16px; color: rgba(0,0,0,0.65); margin-top: 10px; }
          .ac-hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 16px 0 10px; }
          .ac-icon { display:flex; justify-content:flex-end; align-items:center; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
    with c1:
        st.markdown(
            """
            <div class="ac-header">
              <div class="ac-title">Analytics Console</div>
              <div class="ac-sub">Professional content performance and user engagement reporting.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="ac-icon">
              <img src="{analytics_icon_svg_uri()}" width="84" height="84" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="ac-hr" />', unsafe_allow_html=True)


def render_sidebar_branding(default_pid: str):
    with st.sidebar:
        st.markdown("### Период отчета")
        today = date.today()
        date_from = st.date_input("Дата с", value=today - timedelta(days=30), key="date_from")
        date_to = st.date_input("Дата по", value=today, key="date_to")

        st.divider()
        property_id = st.text_input("GA4 Property ID", value=default_pid, key="property_id").strip()

        st.divider()
        st.markdown("### Developed by")
        st.markdown("**Alexey Terekhov**")
        st.markdown("[aterekhov@internews.org](mailto:aterekhov@internews.org)")

        if INTERNEWS_LOGO.exists():
            st.image(str(INTERNEWS_LOGO), width=170)

    return date_from, date_to, property_id


# -------------------------
# Client init
# -------------------------
@st.cache_resource
def get_client_and_pid():
    try:
        cfg = build_config_from_streamlit_secrets(st.secrets)
    except GA4ConfigError as e:
        fail_ui(str(e))
    client = make_client(cfg)
    return client, cfg.property_id


client, pid_default = get_client_and_pid()

# -------------------------
# UI
# -------------------------
render_header()
date_from, date_to, property_id = render_sidebar_branding(pid_default)

tab1, tab2, tab3 = st.tabs(["URL Analytics", "Топ материалов", "Общие данные по сайту"])


# -------------------------
# Tab 1
# -------------------------
with tab1:
    st.subheader("URL Analytics")
    uinput = st.text_area("Вставьте URL или пути (по одному в строке)", height=200)
    lines = [x.strip() for x in (uinput or "").splitlines() if x.strip()]

    if st.button("Собрать", key="btn_collect_urls"):
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


# -------------------------
# Tab 2
# -------------------------
with tab2:
    st.subheader("Топ материалов")
    limit = st.number_input("Лимит", min_value=1, max_value=500, value=10)

    if st.button("Собрать топ", key="btn_collect_top"):
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


# -------------------------
# Tab 3
# -------------------------
with tab3:
    st.subheader("Общие данные по сайту")

    if st.button("Обновить итоги", key="btn_collect_totals"):
        if date_from > date_to:
            fail_ui("Дата «с» должна быть раньше или равна дате «по».")
        if not property_id:
            fail_ui("GA4 Property ID пуст.")

        totals_df = fetch_site_totals(
            client=client,
            property_id=property_id,
            start_date=str(date_from),
            end_date=str(date_to),
        )
        totals = totals_df.iloc[0].to_dict()

        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions", f"{int(totals.get('sessions', 0)):,}")
        c2.metric("Unique Users", f"{int(totals.get('totalUsers', 0)):,}")
        c3.metric("Page Views", f"{int(totals.get('screenPageViews', 0)):,}")
