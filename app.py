from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.transform import Rotation as R

from smart_insole_core import (
    PRESSURE_LAYOUT,
    RenderConfig,
    analyze_and_render,
    build_sampler,
    make_box_vertices,
    parse_combined_json,
    parse_single_side_json,
    validate_insole_df,
)

st.set_page_config(page_title="스마트 인솔 분석 대시보드(제자리 3D)", page_icon="👣", layout="wide")

# ------------------------------
# Session state
# ------------------------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "frame_store" not in st.session_state:
    st.session_state.frame_store = None


# ------------------------------
# Plot helpers
# ------------------------------

def _rotation_apply(rot: R, pts: np.ndarray) -> np.ndarray:
    return rot.apply(pts)


def _box_traces(center: np.ndarray, rot: R, color: str, foot_dims: tuple[float, float, float], stance: bool):
    verts_local, edges = make_box_vertices(*foot_dims)
    verts = _rotation_apply(rot, verts_local) + center
    lw = 9 if stance else 6
    traces = []
    for a, b in edges:
        seg = verts[[a, b]]
        traces.append(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(color=color, width=lw),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def _axis_traces(center: np.ndarray, rot: R, axis_len: float = 0.07):
    axes = np.eye(3) * axis_len
    axes_world = _rotation_apply(rot, axes) + center
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    traces = []
    for i in range(3):
        seg = np.vstack([center, axes_world[i]])
        traces.append(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(color=colors[i], width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def _gyro_trace(center: np.ndarray, rot: R, gyro_body: np.ndarray, scale: float = 0.00055, max_len: float = 0.10):
    g = np.asarray(gyro_body, dtype=float) * scale
    g_len = float(np.linalg.norm(g))
    if g_len > max_len:
        g *= (max_len / g_len)

    base = center + np.array([0.0, 0.0, 0.015])
    tip = base + _rotation_apply(rot, g.reshape(1, 3))[0]
    seg = np.vstack([base, tip])
    return go.Scatter3d(
        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
        mode="lines",
        line=dict(color="#8a2be2", width=8),
        showlegend=False,
        hoverinfo="skip",
    )


def _ground_grid_stationary(x_range=(-0.35, 0.35), y_range=(-0.30, 0.30), z=0.0):
    xs = np.linspace(x_range[0], x_range[1], 9)
    ys = np.linspace(y_range[0], y_range[1], 7)
    traces = []
    for x in xs:
        traces.append(go.Scatter3d(
            x=[x, x], y=[y_range[0], y_range[1]], z=[z, z],
            mode="lines",
            line=dict(color="#efefef", width=3),
            showlegend=False,
            hoverinfo="skip",
        ))
    for y in ys:
        traces.append(go.Scatter3d(
            x=[x_range[0], x_range[1]], y=[y, y], z=[z, z],
            mode="lines",
            line=dict(color="#f2f2f2", width=3),
            showlegend=False,
            hoverinfo="skip",
        ))
    # center line
    traces.append(go.Scatter3d(
        x=[x_range[0], x_range[1]], y=[0.0, 0.0], z=[z, z],
        mode="lines",
        line=dict(color="#333333", width=4),
        showlegend=False,
        hoverinfo="skip",
    ))
    return traces


def make_motion_figure(frame_store: dict, idx: int, cfg: RenderConfig) -> go.Figure:
    left_f = frame_store["left_frames"]
    right_f = frame_store["right_frames"]
    total_frames = len(frame_store["t_frames"])
    idx = int(np.clip(idx, 0, total_frames - 1))

    l_center = left_f["pos"][idx]
    r_center = right_f["pos"][idx]

    foot_dims = (cfg.foot_length, cfg.foot_width, cfg.foot_thick)
    traces = []
    traces.extend(_ground_grid_stationary())

    traces.extend(_box_traces(l_center, left_f["rot"][idx], "#1f77b4", foot_dims, bool(left_f["stance"][idx])))
    traces.extend(_box_traces(r_center, right_f["rot"][idx], "#ff7f0e", foot_dims, bool(right_f["stance"][idx])))

    traces.extend(_axis_traces(l_center, left_f["rot"][idx]))
    traces.extend(_axis_traces(r_center, right_f["rot"][idx]))

    traces.append(_gyro_trace(l_center, left_f["rot"][idx], left_f["gyro"][idx], scale=cfg.gyro_arrow_scale, max_len=cfg.gyro_arrow_max))
    traces.append(_gyro_trace(r_center, right_f["rot"][idx], right_f["gyro"][idx], scale=cfg.gyro_arrow_scale, max_len=cfg.gyro_arrow_max))

    traces.append(go.Scatter3d(
        x=[l_center[0]], y=[l_center[1]], z=[l_center[2]],
        mode="markers+text",
        marker=dict(size=5, color="#1f77b4"),
        text=["LEFT"], textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    ))
    traces.append(go.Scatter3d(
        x=[r_center[0]], y=[r_center[1]], z=[r_center[2]],
        mode="markers+text",
        marker=dict(size=5, color="#ff7f0e"),
        text=["RIGHT"], textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
        scene=dict(
            xaxis=dict(title="Forward (fixed)", range=[-0.35, 0.35], backgroundcolor="white", gridcolor="#efefef"),
            yaxis=dict(title="Lateral (m)", range=[-0.30, 0.30], backgroundcolor="white", gridcolor="#efefef"),
            zaxis=dict(title="Vertical (m)", range=[0.0, 0.12], backgroundcolor="white", gridcolor="#efefef"),
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.0, z=0.7),
            camera=dict(
                eye=dict(x=-1.25, y=-1.25, z=0.70),
                center=dict(x=0.02, y=0.0, z=-0.05),
            ),
        ),
        title="제자리 3D: 인솔 자세(쿼터니언) · 자이로 벡터 · 압력 기반 스윙 리프트",
        showlegend=False,
    )
    return fig


def make_pressure_heatmap(sensor_vals: np.ndarray, vmax: float, title: str) -> go.Figure:
    vals = np.asarray(sensor_vals, dtype=float)
    grid = vals[PRESSURE_LAYOUT]
    annotations = []
    idx_grid = PRESSURE_LAYOUT.copy()
    for r in range(3):
        for c in range(3):
            txt = f"ps_{int(idx_grid[r, c])}<br>{grid[r, c]:.2f}"
            annotations.append(
                dict(
                    x=c,
                    y=r,
                    text=txt,
                    showarrow=False,
                    font=dict(size=14, color="white" if grid[r, c] > 0.55 * vmax else "black"),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            colorscale="Turbo",
            zmin=0,
            zmax=float(max(vmax, 1e-8)),
            showscale=True,
            colorbar=dict(title="pressure"),
            hovertemplate="row=%{y}, col=%{x}, value=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=310,
        margin=dict(l=10, r=10, t=42, b=10),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        annotations=annotations,
    )
    return fig


def make_timeseries_figure(t_frames: np.ndarray, idx: int, left_f: dict, right_f: dict, window_sec: float = 8.0) -> go.Figure:
    t_now = float(t_frames[idx])
    t_min = max(0.0, t_now - window_sec)
    mask = (t_frames >= t_min) & (t_frames <= t_now)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_frames[mask], y=left_f["pressure"][mask], mode="lines", name="L totalP"))
    fig.add_trace(go.Scatter(x=t_frames[mask], y=right_f["pressure"][mask], mode="lines", name="R totalP"))
    fig.add_trace(go.Scatter(x=t_frames[mask], y=left_f["gyro_mag"][mask], mode="lines", name="L gyro mag (dps)"))
    fig.add_trace(go.Scatter(x=t_frames[mask], y=right_f["gyro_mag"][mask], mode="lines", name="R gyro mag (dps)"))
    fig.add_trace(go.Scatter(x=t_frames[mask], y=left_f["v_fwd"][mask], mode="lines", name="L v_fwd (m/s)"))
    fig.add_trace(go.Scatter(x=t_frames[mask], y=right_f["v_fwd"][mask], mode="lines", name="R v_fwd (m/s)"))

    # stance markers (0/1)
    fig.add_trace(go.Scatter(
        x=t_frames[mask],
        y=left_f["stance"][mask].astype(int),
        mode="lines",
        name="L stance(0/1)",
        line=dict(dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=t_frames[mask],
        y=right_f["stance"][mask].astype(int),
        mode="lines",
        name="R stance(0/1)",
        line=dict(dash="dot"),
    ))

    fig.add_vline(x=t_now, line_dash="dash", line_color="#444444")
    fig.update_layout(
        title="최근 구간 시계열(압력/자이로/전진지표/stance)",
        height=340,
        margin=dict(l=10, r=10, t=42, b=10),
        xaxis_title="Time (s)",
        legend=dict(orientation="h", y=1.10),
    )
    return fig


def _csv_preview_from_bytes(b: bytes, n: int = 20) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b)).head(n)


def prepare_frame_store(result: dict, cfg: RenderConfig) -> dict:
    left_track = result["left_track"]
    right_track = result["right_track"]
    total_t = float(min(left_track["t"][-1], right_track["t"][-1]))

    t_frames = np.arange(0, total_t, 1.0 / cfg.target_fps)
    if len(t_frames) == 0 or t_frames[-1] < total_t:
        t_frames = np.append(t_frames, total_t)

    left_frames = build_sampler(left_track, t_frames)
    right_frames = build_sampler(right_track, t_frames)

    sensor_vmax = float(
        max(
            np.quantile(left_track["pressure_matrix"], 0.995),
            np.quantile(right_track["pressure_matrix"], 0.995),
            1e-6,
        )
    )
    return {
        "t_frames": t_frames,
        "left_frames": left_frames,
        "right_frames": right_frames,
        "sensor_vmax": sensor_vmax,
    }


# ------------------------------
# UI
# ------------------------------

st.title("👣 스마트 인솔 분석 대시보드")
st.caption("전진(절대 위치) 복원은 제거하고, 인솔의 **자세(쿼터니언)** · **회전(자이로)** · **압력 변화**를 앱에서 바로 렌더링합니다. mp4/ZIP은 필요할 때만 다운로드하세요.")

with st.sidebar:
    st.header("설정")
    input_mode = st.radio(
        "입력 형식",
        ["통합 JSON 1개", "왼쪽/오른쪽 JSON 2개", "이미 변환된 CSV 2개"],
        index=0,
    )

    target_fps = st.slider("표시/영상 FPS", 8, 24, 15, 1)
    width = st.select_slider("영상 생성 기준 너비", options=[1200, 1366, 1480, 1600, 1700], value=1480)
    step_width = st.slider("좌우 발 간격(m)", 0.12, 0.30, 0.20, 0.01)
    lift_max = st.slider("스윙 최대 상승량(m)", 0.01, 0.06, 0.035, 0.001)
    show_csv_preview = st.toggle("변환된 CSV 미리보기", value=False)

    st.info("압력 히트맵은 ps_0~ps_8 값을 3×3 도식 배치로 표현합니다(물리 배치 정보가 없어서).")

cfg = RenderConfig(
    target_fps=int(target_fps),
    width=int(width),
    height=860,
    motion_width=int(int(width) * 0.61),
    step_width=float(step_width),
    lift_max=float(lift_max),
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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Left 샘플 수", f"{len(left_df):,}")
    m2.metric("Right 샘플 수", f"{len(right_df):,}")
    m3.metric("Left duration(s)", f"{(left_df['ts'].iloc[-1]-left_df['ts'].iloc[0])/1000:.2f}")
    m4.metric("Right duration(s)", f"{(right_df['ts'].iloc[-1]-right_df['ts'].iloc[0])/1000:.2f}")

    if show_csv_preview:
        with st.expander("변환된 CSV 미리보기", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(left_df.head(30), use_container_width=True)
            with c2:
                st.dataframe(right_df.head(30), use_container_width=True)

    if st.button("분석 실행", type="primary", use_container_width=True):
        with st.spinner("분석/통계 계산 + 다운로드용 mp4 생성 중..."):
            result = analyze_and_render(left_df, right_df, cfg, mode="stationary")
            st.session_state.analysis_result = result
            st.session_state.frame_store = prepare_frame_store(result, cfg)
            st.session_state.analysis_done = True
            st.session_state.current_frame = 0


if st.session_state.analysis_done and st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    frame_store = st.session_state.frame_store
    t_frames = frame_store["t_frames"]
    total_frames = len(t_frames)

    st.markdown("---")
    st.subheader("앱 안에서 바로 보는 분석 장면")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 6, 1])
    with ctrl1:
        if st.button("◀ 이전", use_container_width=True) and st.session_state.current_frame > 0:
            st.session_state.current_frame -= 1
    with ctrl3:
        if st.button("다음 ▶", use_container_width=True) and st.session_state.current_frame < total_frames - 1:
            st.session_state.current_frame += 1
    with ctrl2:
        st.session_state.current_frame = st.slider(
            "시간 프레임",
            min_value=0,
            max_value=total_frames - 1,
            value=int(st.session_state.current_frame),
            step=1,
            format="%d",
        )

    idx = int(st.session_state.current_frame)
    tf = float(t_frames[idx])
    left_f = frame_store["left_frames"]
    right_f = frame_store["right_frames"]

    # quick status
    moving_thr = 0.08
    l_moving = (not bool(left_f["stance"][idx])) and (float(left_f["v_fwd"][idx]) > moving_thr)
    r_moving = (not bool(right_f["stance"][idx])) and (float(right_f["v_fwd"][idx]) > moving_thr)

    cmet1, cmet2, cmet3, cmet4, cmet5, cmet6 = st.columns(6)
    cmet1.metric("현재 시각(s)", f"{tf:.2f}")
    cmet2.metric("Left stance", "1" if bool(left_f["stance"][idx]) else "0")
    cmet3.metric("Right stance", "1" if bool(right_f["stance"][idx]) else "0")
    cmet4.metric("Left v_fwd(지표)", f"{float(left_f['v_fwd'][idx]):.2f} m/s")
    cmet5.metric("Right v_fwd(지표)", f"{float(right_f['v_fwd'][idx]):.2f} m/s")
    cmet6.metric("이동 상태(추정)", f"L={'MOVE' if l_moving else 'STOP'} | R={'MOVE' if r_moving else 'STOP'}")

    top_left, top_right = st.columns([1.65, 1.0])
    with top_left:
        st.plotly_chart(make_motion_figure(frame_store, idx, cfg), use_container_width=True, config={"displaylogo": False})
    with top_right:
        st.markdown("##### 현재 프레임 요약")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "side": "left",
                        "stance": int(bool(left_f["stance"][idx])),
                        "v_fwd": float(left_f["v_fwd"][idx]),
                        "a_fwd": float(left_f["a_fwd"][idx]),
                        "total_pressure": float(left_f["pressure"][idx]),
                        "gyro_mag_dps": float(left_f["gyro_mag"][idx]),
                    },
                    {
                        "side": "right",
                        "stance": int(bool(right_f["stance"][idx])),
                        "v_fwd": float(right_f["v_fwd"][idx]),
                        "a_fwd": float(right_f["a_fwd"][idx]),
                        "total_pressure": float(right_f["pressure"][idx]),
                        "gyro_mag_dps": float(right_f["gyro_mag"][idx]),
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.info("빨강/초록/파랑 = local x/y/z 축, 보라색 = 자이로 벡터")

    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(
            make_pressure_heatmap(
                left_f["pressure_matrix"][idx],
                frame_store["sensor_vmax"],
                f"LEFT pressure map · t={tf:.2f}s",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with p2:
        st.plotly_chart(
            make_pressure_heatmap(
                right_f["pressure_matrix"][idx],
                frame_store["sensor_vmax"],
                f"RIGHT pressure map · t={tf:.2f}s",
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    st.plotly_chart(
        make_timeseries_figure(t_frames, idx, left_f, right_f, window_sec=8.0),
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.markdown("#### 전체 요약 통계")
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("기록 길이(s)", result["pair_summary"]["recording_duration_s"])
    pm2.metric("평균 cadence", result["pair_summary"]["mean_cadence_steps_per_min"])
    pm3.metric("평균 stance(%)", result["pair_summary"]["mean_stance_pct"])
    pm4.metric("좌우 step 차", result["pair_summary"]["step_count_abs_diff"])
    pm5.metric("비디오 길이(s)", round(float(result["video_duration_s"]), 2))
    st.dataframe(result["summary_df"], use_container_width=True)

    with st.expander("CSV 미리보기", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(_csv_preview_from_bytes(result["left_csv_bytes"], 30), use_container_width=True)
        with c2:
            st.dataframe(_csv_preview_from_bytes(result["right_csv_bytes"], 30), use_container_width=True)

    # ------------------------------
    # Downloads (bottom)
    # ------------------------------
    st.markdown("---")
    st.subheader("다운로드")
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.download_button("left.csv", result["left_csv_bytes"], file_name="left.csv", mime="text/csv", use_container_width=True)
    with d2:
        st.download_button("right.csv", result["right_csv_bytes"], file_name="right.csv", mime="text/csv", use_container_width=True)
    with d3:
        st.download_button("summary.csv", result["summary_csv_bytes"], file_name="summary.csv", mime="text/csv", use_container_width=True)
    with d4:
        st.download_button("analysis.mp4", result["video_bytes"], file_name="smart_insole_analysis_stationary.mp4", mime="video/mp4", use_container_width=True)
    with d5:
        st.download_button("전체 ZIP", result["zip_bytes"], file_name="smart_insole_results.zip", mime="application/zip", use_container_width=True)

else:
    st.markdown(
        """
        ### 앱 동작
        - JSON/CSV 업로드 → CSV 변환
        - **제자리 3D**로: 인솔 자세(쿼터니언), 자이로 벡터, 압력 히트맵/추세를 **프레임 슬라이더**로 탐색
        - 다운로드(mp4/CSV/ZIP)는 맨 아래 제공

        ### 왜 전진을 뺐나?
        - IMU만으로 절대 위치(전진 거리)를 적분하면 드리프트가 누적되어 **시각적으로 더 부자연**해질 수 있습니다.
        - 따라서 신뢰도가 높은 **자세/회전/압력/이벤트(stance/swing)** 중심으로 보여줍니다.

        업로드하면 바로 시작할 수 있어요.
        """
    )
