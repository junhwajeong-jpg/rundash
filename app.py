import math
from datetime import date, datetime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from supabase import create_client


# ---------------------------
# Secrets / Client
# ---------------------------
def get_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_ANON_KEY = get_secret("SUPABASE_ANON_KEY")
USER_ID = get_secret("USER_ID", "jun")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Supabase Secrets가 비어있습니다. Settings → Secrets에 SUPABASE_URL / SUPABASE_ANON_KEY를 넣어주세요.")
    st.stop()

sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

RUN_TYPES = ["Easy", "Long", "Speed", "Tempo", "Trail", "Intervals"]


# ---------------------------
# Helpers
# ---------------------------
def pace_str_from_min_per_km(min_per_km: float) -> str:
    if min_per_km is None or not np.isfinite(min_per_km) or min_per_km <= 0:
        return "-"
    m = int(min_per_km)
    s = int(round((min_per_km - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d} /km"


def avg_speed_kmh(distance_km: float, time_min: float) -> float:
    if distance_km is None or time_min is None or time_min <= 0:
        return float("nan")
    return float(distance_km) / (float(time_min) / 60.0)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def today_range_default():
    end = date.today()
    start = end - timedelta(days=28)
    return start, end


# ---------------------------
# Supabase CRUD
# ---------------------------
@st.cache_data(ttl=10, show_spinner=False)
def db_fetch_runs(user_id: str) -> pd.DataFrame:
    resp = (
        sb.table("runs")
        .select("*")
        .eq("user_id", user_id)
        .order("date", desc=False)
        .execute()
    )
    rows = resp.data or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["id", "user_id", "date", "type", "distance_km", "time_min", "avg_hr", "note", "created_at", "updated_at"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # numeric normalize
    df["distance_km"] = df["distance_km"].apply(safe_float)
    df["time_min"] = df["time_min"].apply(safe_float)
    df["avg_hr"] = df["avg_hr"].apply(lambda v: safe_int(v, None))
    df["note"] = df["note"].fillna("")
    return df


def db_insert_run(payload: dict):
    sb.table("runs").insert(payload).execute()
    st.cache_data.clear()


def db_update_run(run_id: str, payload: dict):
    sb.table("runs").update(payload).eq("id", run_id).execute()
    st.cache_data.clear()


def db_delete_run(run_id: str):
    sb.table("runs").delete().eq("id", run_id).execute()
    st.cache_data.clear()


def db_delete_all_runs(user_id: str):
    sb.table("runs").delete().eq("user_id", user_id).execute()
    st.cache_data.clear()


@st.cache_data(ttl=60, show_spinner=False)
def db_get_goals(user_id: str) -> dict:
    resp = sb.table("goals").select("*").eq("user_id", user_id).execute()
    rows = resp.data or []
    if not rows:
        # create default
        default = {"user_id": user_id, "weekly_km": 30, "monthly_km": 120}
        sb.table("goals").upsert(default).execute()
        return default
    return rows[0]


def db_save_goals(user_id: str, weekly_km: float, monthly_km: float):
    sb.table("goals").upsert({"user_id": user_id, "weekly_km": weekly_km, "monthly_km": monthly_km}).execute()
    st.cache_data.clear()


@st.cache_data(ttl=60, show_spinner=False)
def db_get_profile(user_id: str) -> dict:
    resp = sb.table("profile").select("*").eq("user_id", user_id).execute()
    rows = resp.data or []
    if not rows:
        default = {
            "user_id": user_id,
            "nickname": "Runner",
            "level": "Intermediate",
            "weight_kg": 70,
            "height_cm": 175,
            "dark_mode": True,
            "weekly_report": False,
        }
        sb.table("profile").upsert(default).execute()
        return default
    return rows[0]


def db_save_profile(user_id: str, payload: dict):
    payload = {"user_id": user_id, **payload}
    sb.table("profile").upsert(payload).execute()
    st.cache_data.clear()


# ---------------------------
# UI: Config + Sidebar
# ---------------------------
profile = db_get_profile(USER_ID)
st.set_page_config(page_title="RunDash", layout="wide")

st.sidebar.title("RunDash")
st.sidebar.caption(f"User: {USER_ID}")

page = st.sidebar.radio("화면", ["Home", "Runs", "Analytics", "Goals", "Profile"], index=0)

st.sidebar.divider()

dummy_mode = st.sidebar.toggle("더미 데이터 보기(입력 전)", value=False)

# Filters (shared)
st.sidebar.subheader("기간")
start_d, end_d = today_range_default()
date_range = st.sidebar.date_input(" ", value=(start_d, end_d))
if isinstance(date_range, tuple) and len(date_range) == 2:
    filter_start, filter_end = date_range
else:
    filter_start, filter_end = start_d, end_d

st.sidebar.subheader("러닝 유형")
selected_types = st.sidebar.multiselect(" ", RUN_TYPES, default=["Easy", "Long", "Speed"])

st.sidebar.subheader("최소 거리(km)")
min_km = st.sidebar.slider(" ", 0.0, 50.0, 0.0, 0.5)

st.sidebar.divider()


# ---------------------------
# Data (real or dummy)
# ---------------------------
runs_df = db_fetch_runs(USER_ID)

if dummy_mode and runs_df.empty:
    # create a dummy dataset for UI preview
    np.random.seed(7)
    base = date.today() - timedelta(days=28)
    rows = []
    for i in range(18):
        d = base + timedelta(days=int(i * 1.6))
        t = np.random.choice(["Easy", "Long", "Speed"])
        dist = float(np.round(np.random.uniform(3, 13), 1))
        pace = float(np.random.uniform(5.2, 7.8))
        time_min = float(np.round(dist * pace, 1))
        rows.append({
            "id": f"dummy-{i}",
            "user_id": USER_ID,
            "date": d,
            "type": t,
            "distance_km": dist,
            "time_min": time_min,
            "avg_hr": int(np.random.uniform(120, 170)),
            "note": "",
            "created_at": "",
            "updated_at": "",
        })
    runs_df = pd.DataFrame(rows)

# Apply filters
filtered = runs_df.copy()
if not filtered.empty:
    filtered = filtered[(filtered["date"] >= filter_start) & (filtered["date"] <= filter_end)]
    if selected_types:
        filtered = filtered[filtered["type"].isin(selected_types)]
    filtered = filtered[filtered["distance_km"] >= float(min_km)]
    filtered = filtered.sort_values("date", ascending=True).reset_index(drop=True)

# Derived metrics
def with_derived(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["pace_min_per_km"] = d.apply(lambda r: (r["time_min"] / r["distance_km"]) if r["distance_km"] > 0 else np.nan, axis=1)
    d["pace_str"] = d["pace_min_per_km"].apply(pace_str_from_min_per_km)
    d["avg_speed_kmh"] = d.apply(lambda r: avg_speed_kmh(r["distance_km"], r["time_min"]), axis=1)
    return d

filtered_d = with_derived(filtered)


# ---------------------------
# HOME
# ---------------------------
def render_home(df: pd.DataFrame):
    st.title("RunDash")

    if df.empty:
        st.info("아직 러닝 기록이 없습니다. Runs 탭에서 기록을 추가해보세요.")
        return

    # headline
    first_day = df["date"].min()
    last_day = df["date"].max()
    total_runs = len(df)

    st.caption(f"{profile.get('nickname', 'Runner')} | {first_day} ~ {last_day} | 총 {total_runs}회 러닝")

    total_km = df["distance_km"].sum()
    total_min = df["time_min"].sum()
    avg_pace = (df["time_min"].sum() / df["distance_km"].sum()) if df["distance_km"].sum() > 0 else np.nan
    days_run = df["date"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 거리", f"{total_km:.1f} km")
    c2.metric("총 시간", f"{total_min/60:.1f} h")
    c3.metric("평균 페이스", pace_str_from_min_per_km(avg_pace))
    c4.metric("러닝 일수", f"{days_run} days")
    # streak
    dates = sorted(df["date"].unique())
    streak = 0
    cur = date.today()
    ds = set(dates)
    while cur in ds:
        streak += 1
        cur = cur - timedelta(days=1)
    c5.metric("스트릭", f"{streak} days")

    st.divider()

    # goals
    goals = db_get_goals(USER_ID)
    weekly_target = float(goals.get("weekly_km", 30))
    week_start = date.today() - timedelta(days=date.today().weekday())
    week_end = week_start + timedelta(days=6)
    week_km = df[(df["date"] >= week_start) & (df["date"] <= week_end)]["distance_km"].sum()
    ratio = 0 if weekly_target <= 0 else float(week_km) / float(weekly_target)
    ratio = max(0.0, min(1.0, ratio))

    left, mid, right = st.columns([1.2, 1.0, 1.2])

    with left:
        st.subheader("이번주 목표")
        st.progress(ratio)
        st.caption(f"{ratio*100:.0f}% ({week_km:.1f}/{weekly_target:.1f} km)")

    with mid:
        st.subheader("PR")
        # longest
        longest = float(df["distance_km"].max()) if not df.empty else 0
        st.metric("최장 거리", f"{longest:.1f} km")
        # best pace in 5k+ and 10k+
        best5 = df[df["distance_km"] >= 5].copy()
        best10 = df[df["distance_km"] >= 10].copy()
        b5 = pace_str_from_min_per_km((best5["time_min"] / best5["distance_km"]).min()) if not best5.empty else "-"
        b10 = pace_str_from_min_per_km((best10["time_min"] / best10["distance_km"]).min()) if not best10.empty else "-"
        st.metric("5k 최고 페이스", b5)
        st.metric("10k 최고 페이스", b10)

    with right:
        st.subheader("빠른 설정")
        with st.form("quick_week_goal"):
            new_weekly = st.number_input("주간 목표(km)", min_value=0.0, max_value=500.0, value=float(weekly_target), step=1.0)
            saved = st.form_submit_button("저장")
        if saved:
            db_save_goals(USER_ID, weekly_km=float(new_weekly), monthly_km=float(goals.get("monthly_km", 120)))
            st.success("주간 목표 저장 완료!")
            st.rerun()

    st.divider()

    # Weekly distance trend (integer y-axis)
    st.subheader("주간 거리 추이")
    dd = df.copy()
    dd["date"] = pd.to_datetime(dd["date"])
    dd["week"] = dd["date"].dt.to_period("W-MON").dt.start_time.dt.date
    w = dd.groupby("week", as_index=False)["distance_km"].sum().sort_values("week")

    y_max = float(np.ceil(w["distance_km"].max() + 1))
    y_ticks = list(range(0, int(max(1, y_max)) + 1, 1))

    ch_week = (
        alt.Chart(w)
        .mark_line(point=True)
        .encode(
            x=alt.X("week:T", title="Week", axis=alt.Axis(format="%m/%d")),
            y=alt.Y("distance_km:Q", title="km", scale=alt.Scale(domain=[0, y_max]), axis=alt.Axis(values=y_ticks)),
            tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("distance_km:Q", title="km", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(ch_week, use_container_width=True)

    st.subheader("일별 페이스")
    d2 = df.copy()
    d2["date"] = pd.to_datetime(d2["date"])
    d2["pace_min_per_km"] = d2["time_min"] / d2["distance_km"]
    # daily x-axis (day unit)
    ch_pace = (
        alt.Chart(d2)
        .mark_circle(size=80)
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%m/%d", tickCount="day")),
            y=alt.Y("pace_min_per_km:Q", title="min/km"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("distance_km:Q", title="km", format=".1f"),
                alt.Tooltip("pace_min_per_km:Q", title="pace(min/km)", format=".2f"),
            ],
            color=alt.Color("type:N", legend=alt.Legend(title="Type")),
        )
        .properties(height=260)
    )
    st.altair_chart(ch_pace, use_container_width=True)


# ---------------------------
# RUNS (CRUD)
# ---------------------------
def render_runs(df: pd.DataFrame):
    st.title("Runs")
    st.caption("기록 추가/수정/삭제/초기화")

    c1, c2 = st.columns([1.1, 1.0])

    with c1:
        st.subheader("기록 추가")
        with st.form("add_run", clear_on_submit=True):
            d = st.date_input("날짜", value=date.today())
            t = st.selectbox("러닝 유형", RUN_TYPES, index=0)
            dist = st.number_input("거리(km)", min_value=0.1, max_value=200.0, value=5.0, step=0.1)
            time_m = st.number_input("시간(분)", min_value=1.0, max_value=2000.0, value=30.0, step=1.0)
            # derived
            pace_min_km = float(time_m) / float(dist)
            speed_kmh = avg_speed_kmh(float(dist), float(time_m))
            st.info(f"평균 페이스: {pace_str_from_min_per_km(pace_min_km)}   |   avr speed (km/hr): {speed_kmh:.2f}")

            hr = st.number_input("평균 심박(선택)", min_value=0, max_value=250, value=0, step=1)
            note = st.text_input("메모(선택)", value="")
            ok = st.form_submit_button("저장")
        if ok:
            payload = {
                "user_id": USER_ID,
                "date": str(d),
                "type": t,
                "distance_km": float(dist),
                "time_min": float(time_m),
                "avg_hr": int(hr) if hr and hr > 0 else None,
                "note": note,
            }
            if dummy_mode:
                st.warning("더미 모드에서는 DB 저장을 막아뒀어요. 더미 토글을 끄고 저장하세요.")
            else:
                db_insert_run(payload)
                st.success("저장 완료!")
                st.rerun()

    with c2:
        st.subheader("기록 관리")
        if df.empty:
            st.info("표시할 기록이 없습니다.")
        else:
            view = df.copy()
            view["pace"] = (view["time_min"] / view["distance_km"]).apply(pace_str_from_min_per_km)
            view["avr_speed(km/hr)"] = (view["distance_km"] / (view["time_min"] / 60)).round(2)
            cols = ["date", "type", "distance_km", "time_min", "pace", "avr_speed(km/hr)", "avg_hr", "note", "id"]
            st.dataframe(view[cols], use_container_width=True, hide_index=True)

            ids = df.sort_values("date", ascending=False)["id"].tolist()
            pick = st.selectbox("수정/삭제할 기록 선택", ids, format_func=lambda rid: f"{rid[:8]}…")

            row = df[df["id"] == pick].iloc[0].to_dict()
            with st.form("edit_run"):
                d = st.date_input("날짜", value=row["date"])
                t = st.selectbox("러닝 유형", RUN_TYPES, index=max(0, RUN_TYPES.index(row["type"]) if row["type"] in RUN_TYPES else 0))
                dist = st.number_input("거리(km)", min_value=0.1, max_value=200.0, value=float(row["distance_km"]), step=0.1)
                time_m = st.number_input("시간(분)", min_value=1.0, max_value=2000.0, value=float(row["time_min"]), step=1.0)

                pace_min_km = float(time_m) / float(dist)
                speed_kmh = avg_speed_kmh(float(dist), float(time_m))
                st.info(f"평균 페이스: {pace_str_from_min_per_km(pace_min_km)}   |   avr speed (km/hr): {speed_kmh:.2f}")

                hr = st.number_input("평균 심박(선택)", min_value=0, max_value=250, value=int(row["avg_hr"] or 0), step=1)
                note = st.text_input("메모(선택)", value=str(row.get("note", "")))

                col_a, col_b, col_c = st.columns([1, 1, 1])
                do_update = col_a.form_submit_button("수정 저장")
                do_delete = col_b.form_submit_button("삭제")
                do_reset = col_c.form_submit_button("전체 초기화")

            if dummy_mode and (do_update or do_delete or do_reset):
                st.warning("더미 모드에서는 DB 반영을 막아뒀어요. 더미 토글을 끄고 실행하세요.")
            else:
                if do_update:
                    db_update_run(pick, {
                        "date": str(d),
                        "type": t,
                        "distance_km": float(dist),
                        "time_min": float(time_m),
                        "avg_hr": int(hr) if hr and hr > 0 else None,
                        "note": note,
                    })
                    st.success("수정 완료!")
                    st.rerun()

                if do_delete:
                    db_delete_run(pick)
                    st.success("삭제 완료!")
                    st.rerun()

                if do_reset:
                    # safety confirm
                    st.session_state["_confirm_reset"] = True

            if st.session_state.get("_confirm_reset", False):
                st.warning("정말 전체 초기화할까요? 되돌릴 수 없습니다.")
                col1, col2 = st.columns(2)
                if col1.button("네, 전체 삭제", type="primary"):
                    db_delete_all_runs(USER_ID)
                    st.session_state["_confirm_reset"] = False
                    st.success("전체 초기화 완료!")
                    st.rerun()
                if col2.button("취소"):
                    st.session_state["_confirm_reset"] = False
                    st.rerun()


# ---------------------------
# ANALYTICS
# ---------------------------
def render_analytics(df: pd.DataFrame):
    st.title("Analytics")

    if df.empty:
        st.info("분석할 데이터가 없습니다. Runs에서 기록을 추가해보세요.")
        return

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["pace_min_per_km"] = d["time_min"] / d["distance_km"]
    d["avg_speed_kmh"] = (d["distance_km"] / (d["time_min"] / 60.0))

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("유형별 거리 합계")
        agg = d.groupby("type", as_index=False)["distance_km"].sum().sort_values("distance_km", ascending=False)
        ch = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("distance_km:Q", title="km"),
                y=alt.Y("type:N", sort="-x", title=""),
                tooltip=[alt.Tooltip("type:N"), alt.Tooltip("distance_km:Q", format=".1f")],
            )
            .properties(height=260)
        )
        st.altair_chart(ch, use_container_width=True)

    with c2:
        st.subheader("페이스 분포")
        ch2 = (
            alt.Chart(d)
            .mark_bar()
            .encode(
                x=alt.X("pace_min_per_km:Q", bin=alt.Bin(maxbins=25), title="min/km"),
                y=alt.Y("count():Q", title="count"),
                tooltip=[alt.Tooltip("count():Q")],
            )
            .properties(height=260)
        )
        st.altair_chart(ch2, use_container_width=True)

    st.subheader("거리 vs 페이스")
    ch3 = (
        alt.Chart(d)
        .mark_circle(size=90)
        .encode(
            x=alt.X("distance_km:Q", title="km"),
            y=alt.Y("pace_min_per_km:Q", title="min/km"),
            color=alt.Color("type:N", legend=alt.Legend(title="Type")),
            tooltip=[
                alt.Tooltip("date:T", format="%Y-%m-%d"),
                alt.Tooltip("type:N"),
                alt.Tooltip("distance_km:Q", format=".1f"),
                alt.Tooltip("pace_min_per_km:Q", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(ch3, use_container_width=True)


# ---------------------------
# GOALS
# ---------------------------
def render_goals():
    st.title("Goals")

    goals = db_get_goals(USER_ID)
    weekly = float(goals.get("weekly_km", 30))
    monthly = float(goals.get("monthly_km", 120))

    with st.form("goals_form"):
        st.subheader("목표 설정")
        new_weekly = st.number_input("주간 목표(km)", min_value=0.0, max_value=1000.0, value=weekly, step=1.0)
        new_monthly = st.number_input("월 목표거리(km)", min_value=0.0, max_value=4000.0, value=monthly, step=5.0)
        ok = st.form_submit_button("저장")

    if ok:
        db_save_goals(USER_ID, float(new_weekly), float(new_monthly))
        st.success("목표 저장 완료!")
        st.rerun()

    st.caption("팁: Home의 '빠른 설정'도 동일 목표를 수정합니다.")


# ---------------------------
# PROFILE
# ---------------------------
def render_profile():
    st.title("Profile")

    p = db_get_profile(USER_ID)

    with st.form("profile_form"):
        st.subheader("프로필")
        nickname = st.text_input("닉네임", value=str(p.get("nickname", "Runner")))
        level = st.selectbox("경험 레벨", ["Beginner", "Intermediate", "Advanced"], index=max(0, ["Beginner", "Intermediate", "Advanced"].index(p.get("level", "Intermediate"))))
        weight_kg = st.number_input("체중(kg)", min_value=30.0, max_value=150.0, value=float(p.get("weight_kg", 70)), step=0.5)
        height_cm = st.number_input("키(cm)", min_value=120, max_value=210, value=int(p.get("height_cm", 175)), step=1)

        st.divider()
        st.subheader("설정")
        dark_mode = st.toggle("다크모드(모형)", value=bool(p.get("dark_mode", True)))
        weekly_report = st.toggle("주간 리포트 알림(모형)", value=bool(p.get("weekly_report", False)))

        ok = st.form_submit_button("저장")

    if ok:
        db_save_profile(USER_ID, {
            "nickname": nickname,
            "level": level,
            "weight_kg": float(weight_kg),
            "height_cm": int(height_cm),
            "dark_mode": bool(dark_mode),
            "weekly_report": bool(weekly_report),
        })
        st.success("프로필 저장 완료!")
        st.rerun()


# ---------------------------
# Route
# ---------------------------
if page == "Home":
    render_home(filtered_d)
elif page == "Runs":
    render_runs(filtered_d)
elif page == "Analytics":
    render_analytics(filtered_d)
elif page == "Goals":
    render_goals()
elif page == "Profile":
    render_profile()
