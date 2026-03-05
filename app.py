from __future__ import annotations

import pandas as pd
import streamlit as st

from smart_insole_core import (
    RenderConfig,
    analyze_and_render,
    parse_combined_json,
    parse_single_side_json,
    validate_insole_df,
)

st.set_page_config(page_title="스마트 인솔 분석", page_icon="👣", layout="wide")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

st.title("👣 스마트 인솔 JSON → CSV 변환 및 분석 영상 생성기")
st.caption(
    "왼쪽/오른쪽 인솔 JSON 또는 CSV를 업로드하면, CSV 변환, 기본 통계량 계산, 3D 분석 영상 생성, 결과 다운로드를 한 번에 수행합니다."
)

with st.sidebar:
    st.header("설정")
    input_mode = st.radio(
        "입력 형식",
        ["통합 JSON 1개", "왼쪽/오른쪽 JSON 2개", "이미 변환된 CSV 2개"],
        index=0,
    )
    target_fps = st.slider("영상 FPS", 8, 24, 15, 1)
    width = st.select_slider("영상 너비", options=[1200, 1366, 1480, 1600, 1700], value=1480)
    step_width = st.slider("좌우 발 간격(m)", 0.12, 0.30, 0.20, 0.01)
    lift_max = st.slider("스윙 최대 상승량(m)", 0.01, 0.06, 0.035, 0.001)
    st.markdown("---")
    autoplay = st.toggle("영상 자동재생", value=False)
    show_preview_table = st.toggle("CSV 미리보기 표시", value=False)
    st.info(
        "압력 패널은 ps_0 ~ ps_8 값을 3×3 도식 배치로 보여줍니다. 실제 센서 물리 배치가 있으면 그에 맞게 바꿀 수 있습니다."
    )

cfg = RenderConfig(
    target_fps=target_fps,
    width=width,
    height=860,
    motion_width=int(width * 0.61),
    step_width=step_width,
    lift_max=lift_max,
)

left_df = None
right_df = None
parse_meta = {}

try:
    if input_mode == "통합 JSON 1개":
        up = st.file_uploader("통합 JSON 업로드", type=["json"])
        if up is not None:
            left_df, right_df, parse_meta = parse_combined_json(up.getvalue())

    elif input_mode == "왼쪽/오른쪽 JSON 2개":
        c1, c2 = st.columns(2)
        with c1:
            up_left = st.file_uploader("왼쪽 JSON", type=["json"], key="left_json")
        with c2:
            up_right = st.file_uploader("오른쪽 JSON", type=["json"], key="right_json")
        if up_left is not None and up_right is not None:
            left_df = parse_single_side_json(up_left.getvalue(), "left")
            right_df = parse_single_side_json(up_right.getvalue(), "right")

    else:
        c1, c2 = st.columns(2)
        with c1:
            up_left = st.file_uploader("왼쪽 CSV", type=["csv"], key="left_csv")
        with c2:
            up_right = st.file_uploader("오른쪽 CSV", type=["csv"], key="right_csv")
        if up_left is not None and up_right is not None:
            left_df = validate_insole_df(pd.read_csv(up_left), "left")
            right_df = validate_insole_df(pd.read_csv(up_right), "right")

except Exception as e:
    st.error(f"파일 파싱 중 오류: {e}")

if left_df is not None and right_df is not None:
    st.success("파일을 정상적으로 읽었습니다.")
    if parse_meta:
        st.write("자동 탐지 경로", parse_meta)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Left 샘플 수", f"{len(left_df):,}")
    k2.metric("Right 샘플 수", f"{len(right_df):,}")
    k3.metric("Left duration(s)", f"{(left_df['ts'].iloc[-1]-left_df['ts'].iloc[0])/1000:.2f}")
    k4.metric("Right duration(s)", f"{(right_df['ts'].iloc[-1]-right_df['ts'].iloc[0])/1000:.2f}")

    if show_preview_table:
        with st.expander("변환된 CSV 미리보기", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("left.csv preview")
                st.dataframe(left_df.head(20), use_container_width=True)
            with c2:
                st.subheader("right.csv preview")
                st.dataframe(right_df.head(20), use_container_width=True)

    if st.button("분석 실행 및 영상 생성", type="primary", use_container_width=True):
        with st.spinner("CSV 변환, 통계량 계산, 분석 영상 생성 중입니다..."):
            st.session_state.analysis_result = analyze_and_render(left_df, right_df, cfg)
            st.session_state.analysis_done = True

if st.session_state.analysis_done and st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result

    st.markdown("---")
    st.subheader("앱에서 바로 분석 영상 보기")
    st.video(
        result["video_bytes"],
        format="video/mp4",
        autoplay=autoplay,
        muted=autoplay,
        width="stretch",
    )
    st.caption(
        "영상은 앱 안에서 바로 재생됩니다. 다운로드는 선택 사항입니다. 자동재생을 켜면 브라우저 정책상 무음으로 재생됩니다."
    )

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("기록 길이(s)", result["pair_summary"]["recording_duration_s"])
    p2.metric("평균 cadence", result["pair_summary"]["mean_cadence_steps_per_min"])
    p3.metric("평균 stance(%)", result["pair_summary"]["mean_stance_pct"])
    p4.metric("거리 비대칭(m)", result["pair_summary"]["distance_symmetry_abs_diff_m"])
    p5.metric("비디오 길이(s)", round(result["video_duration_s"], 2))

    tab1, tab2, tab3 = st.tabs(["요약 통계", "CSV/표 보기", "다운로드"])
    with tab1:
        st.dataframe(result["summary_df"], use_container_width=True)
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("left.csv preview")
            st.dataframe(pd.read_csv(pd.io.common.BytesIO(result["left_csv_bytes"])).head(30), use_container_width=True)
        with c2:
            st.subheader("right.csv preview")
            st.dataframe(pd.read_csv(pd.io.common.BytesIO(result["right_csv_bytes"])).head(30), use_container_width=True)
    with tab3:
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            st.download_button("left.csv", result["left_csv_bytes"], file_name="left.csv", mime="text/csv", use_container_width=True)
        with d2:
            st.download_button("right.csv", result["right_csv_bytes"], file_name="right.csv", mime="text/csv", use_container_width=True)
        with d3:
            st.download_button("summary.csv", result["summary_csv_bytes"], file_name="summary.csv", mime="text/csv", use_container_width=True)
        with d4:
            st.download_button("analysis.mp4", result["video_bytes"], file_name="smart_insole_analysis.mp4", mime="video/mp4", use_container_width=True)
        with d5:
            st.download_button("전체 ZIP", result["zip_bytes"], file_name="smart_insole_results.zip", mime="application/zip", use_container_width=True)

else:
    st.markdown(
        """
        ### 사용 방법
        1. 통합 JSON 1개 또는 left/right JSON 2개를 업로드합니다.
        2. 앱이 JSON을 CSV로 변환합니다.
        3. 분석 실행 버튼을 누르면 통계량과 3D 분석 영상을 생성합니다.
        4. 앱 안에서 바로 영상을 확인하고, 필요하면 CSV/mp4/ZIP만 내려받습니다.

        ### 현재 분석에 포함된 내용
        - 쿼터니언 기반 3D 인솔 자세 시각화
        - 고정 전진축 기반 전진/정지 추정
        - 압력 히트맵과 총압 추세
        - stance/swing 기반 기본 통계량
        """
    )
