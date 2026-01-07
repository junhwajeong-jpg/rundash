import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="RunDash", layout="wide")
st.title("RunDash")

# -----------------------------
# Session storage (your real data)
# -----------------------------
if "runs" not in st.session_state:
    st.session_state.runs = pd.DataFrame(
        columns=["date", "type", "distance_km", "time_min", "avg_hr", "avg_speed_kmh", "note"]
    )

# Goals saved
if "goal_km" not in st.session_state:
    st.session_state.goal_km = 120.0

# Weekly goal (for ring)
if "weekly_goal_km" not in st.session_state:
    st.session_state.weekly_goal_km = 30.0

# Profile saved
if "profile" not in st.session_state:
    st.session_state.profile = {
        "nickname": "Runner",
        "level": "Intermediate",
        "weight_kg": 70.0,
        "height_cm": 175,
        "dark_mode": True,
        "weekly_report": False,
    }

# -----------------------------
# Optional mock data (only when empty)
# -----------------------------
def make_mock_runs(days=90):
    np.random.seed(7)
    dates = pd.date_range(date.today() - timedelta(days=days - 1), periods=days, freq="D")
    ran = np.random.binomial(1, 0.55, size=days)
    dist = np.where(ran == 1, np.random.gamma(3, 2, size=days), 0).round(2)
    pace = np.where(dist > 0, (np.random.normal(6.0, 0.6, size=days)).clip(4.2, 9.5), np.nan)
    time_min = np.where(dist > 0, (dist * pace).round(1), 0)
    hr = np.where(dist > 0, np.random.normal(148, 10, size=days).clip(110, 185).round(0), np.nan)

    df = pd.DataFrame({
        "date": dates,
        "type": np.where(dist == 0, "Rest",
                         np.where(pace <= 5.2, "Speed",
                                  np.where(dist >= 12, "Long", "Easy"))),
        "distance_km": dist,
        "time_min": time_min,
        "avg_hr": hr,
        "note": ""
    })
    df = df[df["distance_km"] > 0].copy()
    df["avg_speed_kmh"] = np.where(
        df["time_min"] > 0, df["distance_km"] / (df["time_min"] / 60.0), 0.0
    ).round(2)
    return df

use_mock = st.sidebar.toggle("더미 데이터 보기(입력 전)", value=(len(st.session_state.runs) == 0))

# -----------------------------
# Data prep
# -----------------------------
def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # avg_speed_kmh is derived; keep it consistent
    df["avg_speed_kmh"] = np.where(
        df["time_min"] > 0, df["distance_km"] / (df["time_min"] / 60.0), 0.0
    ).round(2)

    # for charts
    df["pace_min_km"] = np.where(df["distance_km"] > 0, df["time_min"] / df["distance_km"], np.nan)
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["run_type"] = df["type"]
    df["day"] = df["date"].dt.floor("D")
    return df

base = make_mock_runs(90) if (use_mock and len(st.session_state.runs) == 0) else st.session_state.runs
df = prep(base) if len(base) else pd.DataFrame(
    columns=["date", "type", "distance_km", "time_min", "avg_hr", "avg_speed_kmh", "note",
             "pace_min_km", "week", "month", "run_type", "day"]
)

def fmt_pace(x):
    if pd.isna(x) or x <= 0:
        return "-"
    m = int(x)
    s = int(round((x - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d} /km"

# -----------------------------
# “App-like” stats
# -----------------------------
def current_streak_days(dates: pd.Series) -> int:
    """dates: datetime series of run days (date part)."""
    if dates.empty:
        return 0
    days = pd.to_datetime(dates.dt.date).drop_duplicates().sort_values()
    dayset = set(days.dt.date.tolist())
    streak = 0
    cur = date.today()
    while cur in dayset:
        streak += 1
        cur = cur - timedelta(days=1)
    return streak

def best_pr(df_runs: pd.DataFrame):
    """Simple PRs: best pace for 5K/10K attempts and longest run."""
    if df_runs.empty:
        return {"best_5k_pace": None, "best_10k_pace": None, "longest_km": None}

    best_5k = df_runs[df_runs["distance_km"] >= 5].sort_values("pace_min_km").head(1)
    best_10k = df_runs[df_runs["distance_km"] >= 10].sort_values("pace_min_km").head(1)
    longest = df_runs.sort_values("distance_km", ascending=False).head(1)

    return {
        "best_5k_pace": None if best_5k.empty else float(best_5k.iloc[0]["pace_min_km"]),
        "best_10k_pace": None if best_10k.empty else float(best_10k.iloc[0]["pace_min_km"]),
        "longest_km": None if longest.empty else float(longest.iloc[0]["distance_km"]),
    }

# -----------------------------
# Sidebar nav + filters
# -----------------------------
st.sidebar.title("RunDash")
view = st.sidebar.radio("화면", ["Home", "Runs", "Analytics", "Goals", "Profile"], index=0)

if len(df) > 0:
    today = df["date"].max().date()
    default_start = today - timedelta(days=29)
else:
    today = date.today()
    default_start = today - timedelta(days=29)

start, end = st.sidebar.date_input("기간", value=(default_start, today))
if isinstance(start, (tuple, list)):
    start, end = start

run_types = sorted([x for x in df["run_type"].dropna().unique()]) if len(df) else ["Easy", "Speed", "Long", "Recovery"]
sel_types = st.sidebar.multiselect("러닝 유형", run_types, default=run_types)
min_km = st.sidebar.slider("최소 거리(km)", 0.0, 30.0, 0.0, 0.5)

if len(df) > 0:
    f = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()
    f = f[f["distance_km"] >= min_km]
    f = f[f["run_type"].isin(sel_types)]
else:
    f = df.copy()

runs = f[f["distance_km"] > 0].copy()

nickname = st.session_state.profile.get("nickname", "Runner")
st.caption(f"{nickname}  |  {start} ~ {end}  |  총 {len(runs)}회 러닝")

# -----------------------------
# HOME
# -----------------------------
if view == "Home":
    if len(runs) == 0:
        st.info("데이터가 없습니다. Runs 탭에서 러닝 기록을 입력하세요.")
    else:
        # KPIs
        total_km = runs["distance_km"].sum()
        total_time = runs["time_min"].sum()
        avg_pace = (runs["time_min"].sum() / runs["distance_km"].sum()) if total_km > 0 else np.nan
        active_days = runs["date"].dt.date.nunique()
        avg_hr = runs["avg_hr"].mean()

        # App-like metrics
        streak = current_streak_days(runs["date"])
        pr = best_pr(runs)

        # Weekly goal progress (based on current ISO week)
        iso_week = date.today().isocalendar().week
        this_week_km = df[df["week"] == iso_week]["distance_km"].sum() if len(df) else 0.0
        weekly_goal = float(st.session_state.weekly_goal_km)
        weekly_progress = 0 if weekly_goal <= 0 else min(this_week_km / weekly_goal, 1.0)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("총 거리", f"{total_km:.1f} km")
        c2.metric("총 시간", f"{total_time/60:.1f} h")
        c3.metric("평균 페이스", fmt_pace(avg_pace))
        c4.metric("러닝 일수", f"{active_days} days")
        c5.metric("스트릭", f"{streak} days")
        c6.metric("이번주 누적", f"{this_week_km:.1f} km")

        st.divider()

        # Weekly goal ring + PR cards
        a, b, c = st.columns([1, 1, 2])
        with a:
            st.subheader("이번주 목표")
            st.progress(weekly_progress, text=f"{weekly_progress*100:.0f}% ({this_week_km:.1f}/{weekly_goal:.0f} km)")

        with b:
            st.subheader("PR")
            st.metric("최장 거리", "-" if pr["longest_km"] is None else f"{pr['longest_km']:.1f} km")
            st.metric("5K 최고 페이스", "-" if pr["best_5k_pace"] is None else fmt_pace(pr["best_5k_pace"]))
            st.metric("10K 최고 페이스", "-" if pr["best_10k_pace"] is None else fmt_pace(pr["best_10k_pace"]))

        with c:
            st.subheader("빠른 설정")
            with st.form("weekly_goal_form"):
                wk_goal = st.number_input("주간 목표(km)", min_value=0.0, value=float(st.session_state.weekly_goal_km), step=1.0)
                save_wk = st.form_submit_button("저장")
            if save_wk:
                st.session_state.weekly_goal_km = float(wk_goal)
                st.success("주간 목표 저장 완료!")
                st.rerun()

        st.divider()

        left, right = st.columns([2, 1])

        with left:
            st.subheader("주간 거리 추이")
            wk = runs.groupby("week", as_index=False)["distance_km"].sum().sort_values("week")
            fig_wk = px.bar(wk, x="week", y="distance_km")
            fig_wk.update_xaxes(dtick=1, tickformat="d")  # ✅ 1단위 정수
            st.plotly_chart(fig_wk, use_container_width=True)

            st.subheader("일별 페이스(러닝일만)")
            d = runs.sort_values("day")
            fig_day = px.line(d, x="day", y="pace_min_km", markers=True)
            fig_day.update_xaxes(dtick="D1", tickformat="%m-%d")  # ✅ 1일 단위
            st.plotly_chart(fig_day, use_container_width=True)

        with right:
            st.subheader("러닝 유형 비중")
            typ = runs["run_type"].value_counts().reset_index()
            typ.columns = ["run_type", "count"]
            st.plotly_chart(px.pie(typ, names="run_type", values="count"), use_container_width=True)

            st.subheader("최근 러닝")
            recent = runs.sort_values("date", ascending=False).head(8)[
                ["date", "run_type", "distance_km", "pace_min_km", "avg_speed_kmh", "time_min", "avg_hr"]
            ].copy()
            recent["pace"] = recent["pace_min_km"].apply(fmt_pace)
            recent = recent.rename(columns={"run_type": "type", "distance_km": "km", "time_min": "min"})[
                ["date", "type", "km", "pace", "avg_speed_kmh", "min", "avg_hr"]
            ]
            st.dataframe(recent, use_container_width=True, hide_index=True)

# -----------------------------
# RUNS (input + edit + reset)
# -----------------------------
elif view == "Runs":
    st.subheader("러닝 기록 입력")

    with st.form("run_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        run_date = c1.date_input("날짜", value=date.today())
        run_type = c2.selectbox("유형", ["Easy", "Speed", "Long", "Recovery"])
        distance_km = c3.number_input("거리(km)", min_value=0.0, step=0.1)

        c4, c5, c6 = st.columns(3)
        time_min = c4.number_input("시간(분)", min_value=0.0, step=1.0)
        avg_hr = c5.number_input("평균 심박(bpm)", min_value=0, step=1)
        note = c6.text_input("메모", value="")

        submitted = st.form_submit_button("저장")

    if submitted:
        avg_speed_kmh = (float(distance_km) / (float(time_min) / 60.0)) if float(time_min) > 0 else 0.0
        new_row = pd.DataFrame([{
            "date": pd.to_datetime(run_date),
            "type": run_type,
            "distance_km": float(distance_km),
            "time_min": float(time_min),
            "avg_hr": int(avg_hr) if avg_hr > 0 else None,
            "avg_speed_kmh": round(avg_speed_kmh, 2),
            "note": note
        }])
        st.session_state.runs = pd.concat([st.session_state.runs, new_row], ignore_index=True)
        st.success("저장 완료!")
        st.rerun()

    st.divider()

    cA, cB = st.columns([1, 3])
    with cA:
        if st.button("🗑️ 전체 초기화", use_container_width=True, type="secondary"):
            st.session_state.runs = st.session_state.runs.iloc[0:0].copy()
            st.success("전체 데이터 초기화 완료")
            st.rerun()
    with cB:
        st.caption("표에서 직접 수정/삭제 후 ‘수정 저장’을 누르세요. (avg_speed_kmh는 자동 계산/읽기전용)")

    st.subheader("기록 관리(수정/삭제)")
    if len(st.session_state.runs) == 0:
        st.info("아직 입력된 데이터가 없습니다.")
    else:
        df_view = st.session_state.runs.copy()
        df_view["date"] = pd.to_datetime(df_view["date"])
        df_view["avg_speed_kmh"] = np.where(
            df_view["time_min"] > 0,
            df_view["distance_km"] / (df_view["time_min"] / 60.0),
            0.0
        ).round(2)

        df_view = df_view.sort_values("date", ascending=False).reset_index(drop=True)

        edited = st.data_editor(
            df_view,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            disabled=["avg_speed_kmh"],  # 읽기전용
            column_config={
                "date": st.column_config.DateColumn("date", format="YYYY-MM-DD"),
                "type": st.column_config.SelectboxColumn("type", options=["Easy", "Speed", "Long", "Recovery"]),
                "distance_km": st.column_config.NumberColumn("distance_km", min_value=0.0, step=0.1),
                "time_min": st.column_config.NumberColumn("time_min", min_value=0.0, step=1.0),
                "avg_hr": st.column_config.NumberColumn("avg_hr", min_value=0, step=1),
                "avg_speed_kmh": st.column_config.NumberColumn("avg_speed_kmh", min_value=0.0, step=0.1),
                "note": st.column_config.TextColumn("note"),
            }
        )

        if st.button("💾 수정 저장", type="primary"):
            edited = edited.copy()
            edited["date"] = pd.to_datetime(edited["date"])
            edited["avg_speed_kmh"] = np.where(
                edited["time_min"] > 0,
                edited["distance_km"] / (edited["time_min"] / 60.0),
                0.0
            ).round(2)
            st.session_state.runs = edited.sort_values("date").reset_index(drop=True)
            st.success("수정 내용 저장 완료")
            st.rerun()

# -----------------------------
# ANALYTICS
# -----------------------------
elif view == "Analytics":
    st.subheader("분석")
    if len(runs) == 0:
        st.info("데이터가 없습니다. Runs에서 기록을 입력하세요.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**거리 vs 페이스**")
            st.plotly_chart(px.scatter(runs, x="distance_km", y="pace_min_km", color="run_type",
                                       hover_data=["date", "avg_hr"]), use_container_width=True)
        with c2:
            st.markdown("**심박 vs 페이스**")
            st.plotly_chart(px.scatter(runs, x="avg_hr", y="pace_min_km", color="run_type",
                                       hover_data=["date", "distance_km"]), use_container_width=True)

        st.divider()
        st.subheader("월별 합계")
        m = runs.groupby("month", as_index=False).agg(
            total_km=("distance_km", "sum"),
            runs=("distance_km", "count"),
            avg_pace=("pace_min_km", "mean"),
            avg_hr=("avg_hr", "mean"),
        ).sort_values("month")
        m["avg_pace"] = m["avg_pace"].apply(fmt_pace)
        st.dataframe(m, use_container_width=True, hide_index=True)

# -----------------------------
# GOALS (saveable)
# -----------------------------
elif view == "Goals":
    st.subheader("목표")

    with st.form("goal_form"):
        goal_km = st.number_input(
            "월 목표 거리(km)",
            min_value=0.0,
            value=float(st.session_state.goal_km),
            step=5.0
        )
        saved = st.form_submit_button("저장")

    if saved:
        st.session_state.goal_km = float(goal_km)
        st.success("목표 저장 완료!")
        st.rerun()

    goal_km = float(st.session_state.goal_km)

    if len(df) == 0:
        st.info("데이터가 없습니다. Runs에서 기록을 입력하세요.")
    else:
        this_month = df[df["month"] == df["month"].max()]
        this_month_km = this_month["distance_km"].sum()
        progress = 0 if goal_km == 0 else min(this_month_km / goal_km, 1.0)

        st.metric("이번 달 누적 거리", f"{this_month_km:.1f} km")
        st.progress(progress, text=f"{progress*100:.0f}% 달성")

# -----------------------------
# PROFILE (saveable)
# -----------------------------
else:
    st.subheader("프로필")

    p = st.session_state.profile

    with st.form("profile_form"):
        nickname = st.text_input("닉네임", value=p["nickname"])
        level = st.selectbox(
            "경험 레벨",
            ["Beginner", "Intermediate", "Advanced"],
            index=["Beginner", "Intermediate", "Advanced"].index(p["level"])
        )
        weight_kg = st.number_input("체중(kg)", min_value=30.0, max_value=150.0, value=float(p["weight_kg"]), step=0.5)
        height_cm = st.number_input("키(cm)", min_value=120, max_value=210, value=int(p["height_cm"]), step=1)

        st.divider()
        st.subheader("설정")
        dark_mode = st.toggle("다크모드(모형)", value=bool(p["dark_mode"]))
        weekly_report = st.toggle("주간 리포트 알림(모형)", value=bool(p["weekly_report"]))

        saved = st.form_submit_button("저장")

    if saved:
        st.session_state.profile = {
            "nickname": nickname,
            "level": level,
            "weight_kg": float(weight_kg),
            "height_cm": int(height_cm),
            "dark_mode": bool(dark_mode),
            "weekly_report": bool(weekly_report),
        }
        st.success("프로필 저장 완료!")
        st.rerun()

