
from __future__ import annotations

import io
from pathlib import Path

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

st.set_page_config(page_title="스마트 인솔 분석 대시보드", page_icon="👣", layout="wide")

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

def _rotation_apply(rot, pts: np.ndarray) -> np.ndarray:
    return rot.apply(pts)


def _box_traces(center: np.ndarray, rot, color: str, name: str, foot_dims: tuple[float, float, float]):
    verts_local, edges = make_box_vertices(*foot_dims)
    verts = _rotation_apply(rot, verts_local) + center
    traces = []
    for a, b in edges:
        seg = verts[[a, b]]
        traces.append(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(color=color, width=7),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces, verts


def _axis_traces(center: np.ndarray, rot, axis_len: float = 0.07):
    axes = np.eye(3) * axis_len
    axes_world = _rotation_apply(rot, axes) + center
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    names = ["local x", "local y", "local z"]
    traces = []
    for i in range(3):
        seg = np.vstack([center, axes_world[i]])
        traces.append(
            go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line=dict(color=colors[i], width=6),
                name=names[i],
                legendgroup=names[i],
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def _gyro_trace(center: np.ndarray, rot, gyro_body: np.ndarray, color: str = "#8e44ad"):
    g = np.asarray(gyro_body, dtype=float) * 0.00055
    g_len = np.linalg.norm(g)
    gyro_arrow_max = 0.10
    if g_len > gyro_arrow_max:
        g *= gyro_arrow_max / g_len
    base = center + np.array([0.0, 0.0, 0.015])
    tip = base + _rotation_apply(rot, g.reshape(1, 3))[0]
    return go.Scatter3d(
        x=[base[0], tip[0]], y=[base[1], tip[1]], z=[base[2], tip[2]],
        mode="lines",
        line=dict(color=color, width=8),
        name="gyro",
        showlegend=False,
        hoverinfo="skip",
    )


def _trail_trace(trail_xyz: np.ndarray, color: str, name: str):
    if len(trail_xyz) < 2:
        return None
    return go.Scatter3d(
        x=trail_xyz[:, 0], y=trail_xyz[:, 1], z=trail_xyz[:, 2],
        mode="lines",
        line=dict(color=color, width=4),
        name=name,
        showlegend=False,
        hoverinfo="skip",
        opacity=0.45,
    )


def _ground_grid_traces(x_center: float, x_span: tuple[float, float], step_width: float):
    traces = []
    x_min, x_max = x_span
    for gx in np.linspace(x_min, x_max, 16):
        traces.append(go.Scatter3d(
            x=[gx, gx], y=[-0.28, 0.28], z=[0.0, 0.0],
            mode="lines", line=dict(color="#e8e8e8", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    for gy in np.linspace(-0.25, 0.25, 6):
        traces.append(go.Scatter3d(
            x=[x_min, x_max], y=[gy, gy], z=[0.0, 0.0],
            mode="lines", line=dict(color="#ededed", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    guide_specs = [
        (0.0, "#333333", 5, "center line"),
        (+step_width / 2, "#1f77b4", 3, "left path"),
        (-step_width / 2, "#ff7f0e", 3, "right path"),
    ]
    for gy, color, width, name in guide_specs:
        traces.append(go.Scatter3d(
            x=[x_min, x_max], y=[gy, gy], z=[0.0, 0.0],
            mode="lines", line=dict(color=color, width=width),
            name=name, showlegend=False, hoverinfo="skip",
        ))
    return traces


def make_motion_figure(frame_store: dict, idx: int, cfg: RenderConfig) -> go.Figure:
    left_f = frame_store["left_frames"]
    right_f = frame_store["right_frames"]
    total_frames = len(frame_store["t_frames"])
    idx = int(np.clip(idx, 0, total_frames - 1))
    trail_n = frame_store["trail_n"]

    l_center = left_f["pos"][idx]
    r_center = right_f["pos"][idx]
    x_now = max(l_center[0], r_center[0])
    x_min, x_max = x_now - 1.2, x_now + 0.9

    traces = []
    traces.extend(_ground_grid_traces(x_now, (x_min, x_max), cfg.step_width))

    left_trail = left_f["pos"][max(0, idx - trail_n): idx + 1]
    right_trail = right_f["pos"][max(0, idx - trail_n): idx + 1]
    t1 = _trail_trace(left_trail, "#1f77b4", "left trail")
    t2 = _trail_trace(right_trail, "#ff7f0e", "right trail")
    if t1 is not None:
        traces.append(t1)
    if t2 is not None:
        traces.append(t2)

    foot_dims = (cfg.foot_length, cfg.foot_width, cfg.foot_thick)
    left_box, _ = _box_traces(l_center, left_f["rot"][idx], "#1f77b4", "LEFT", foot_dims)
    right_box, _ = _box_traces(r_center, right_f["rot"][idx], "#ff7f0e", "RIGHT", foot_dims)
    traces.extend(left_box)
    traces.extend(right_box)
    traces.extend(_axis_traces(l_center, left_f["rot"][idx]))
    traces.extend(_axis_traces(r_center, right_f["rot"][idx]))
    traces.append(_gyro_trace(l_center, left_f["rot"][idx], left_f["gyro"][idx]))
    traces.append(_gyro_trace(r_center, right_f["rot"][idx], right_f["gyro"][idx]))

    traces.append(go.Scatter3d(
        x=[l_center[0]], y=[l_center[1]], z=[l_center[2]],
        mode="markers+text",
        marker=dict(size=5, color="#1f77b4"),
        text=["LEFT"], textposition="top center",
        name="left", showlegend=False,
    ))
    traces.append(go.Scatter3d(
        x=[r_center[0]], y=[r_center[1]], z=[r_center[2]],
        mode="markers+text",
        marker=dict(size=5, color="#ff7f0e"),
        text=["RIGHT"], textposition="top center",
        name="right", showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
        scene=dict(
            xaxis=dict(title="Forward (m)", range=[x_min, x_max], backgroundcolor="white", gridcolor="#efefef"),
            yaxis=dict(title="Lateral (m)", range=[-0.30, 0.30], backgroundcolor="white", gridcolor="#efefef"),
            zaxis=dict(title="Vertical (m)", range=[0.0, 0.12], backgroundcolor="white", gridcolor="#efefef"),
            aspectmode="manual",
            aspectratio=dict(x=2.6, y=1.0, z=0.7),
            camera=dict(
                eye=dict(x=-1.65, y=-1.45, z=0.72),
                center=dict(x=0.05, y=0.0, z=-0.05),
            ),
        ),
        title="앱 내 실시간 분석 뷰: 3D 인솔 자세 · 궤적 · 자이로",
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
            annotations.append(dict(x=c, y=r, text=txt, showarrow=False, font=dict(size=14, color="white" if grid[r, c] > 0.55 * vmax else "black")))

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale="Turbo",
        zmin=0,
        zmax=float(max(vmax, 1e-8)),
        showscale=True,
        colorbar=dict(title="pressure"),
        hovertemplate="row=%{y}, col=%{x}, value=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=310,
        margin=dict(l=10, r=10, t=42, b=10),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        annotations=annotations,
    )
    return fig


def make_pressure_trend_figure(t_frames: np.ndarray, idx: int, left_total: np.ndarray, right_total: np.ndarray, window_sec: float = 6.0) -> go.Figure:
    t_now = float(t_frames[idx])
    t_min = max(0.0, t_now - window_sec)
    mask = (t_frames >= t_min) & (t_frames <= t_now)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_frames[mask], y=left_total[mask], mode="lines", name="LEFT", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=t_frames[mask], y=right_total[mask], mode="lines", name="RIGHT", line=dict(color="#ff7f0e", width=3)))
    fig.add_vline(x=t_now, line_dash="dash", line_color="#444444")
    fig.update_layout(
        title="최근 구간 Total pressure 추세",
        height=280,
        margin=dict(l=10, r=10, t=42, b=10),
        xaxis_title="Time (s)",
        yaxis_title="Total pressure",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def _csv_preview_from_bytes(b: bytes, n: int = 20) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b)).head(n)


def prepare_frame_store(result: dict, cfg: RenderConfig) -> dict:
    left_track = result["left_track"]
    right_track = result["right_track"]
    total_t = min(left_track["t"][-1], right_track["t"][-1])
    t_frames = np.arange(0, total_t, 1.0 / cfg.target_fps)
    if len(t_frames) == 0 or t_frames[-1] < total_t:
        t_frames = np.append(t_frames, total_t)
    left_frames = build_sampler(left_track, t_frames)
    right_frames = build_sampler(right_track, t_frames)
    sensor_vmax = float(max(np.quantile(left_track["pressure_matrix"], 0.995), np.quantile(right_track["pressure_matrix"], 0.995), 1e-6))
    total_vmax = float(max(np.quantile(left_track["pressure"], 0.995), np.quantile(right_track["pressure"], 0.995), 1e-6))
    return {
        "t_frames": t_frames,
        "left_frames": left_frames,
        "right_frames": right_frames,
        "sensor_vmax": sensor_vmax,
        "total_vmax": total_vmax,
        "trail_n": max(2, int(cfg.trail_sec * cfg.target_fps)),
    }


# ------------------------------
# UI
# ------------------------------

st.title("👣 스마트 인솔 분석 대시보드")
st.caption("JSON/CSV 업로드 후, 영상으로 굽기 전에 분석 장면 자체를 앱 안에서 바로 확인할 수 있도록 구성했습니다.")

with st.sidebar:
    st.header("설정")
    input_mode = st.radio(
        "입력 형식",
        ["통합 JSON 1개", "왼쪽/오른쪽 JSON 2개", "이미 변환된 CSV 2개"],
        index=0,
    )
    target_fps = st.slider("샘플링 FPS(앱/영상 공통)", 8, 24, 15, 1)
    width = st.select_slider("영상/분석 기준 너비", options=[1200, 1366, 1480, 1600, 1700], value=1480)
    step_width = st.slider("좌우 발 간격(m)", 0.12, 0.30, 0.20, 0.01)
    lift_max = st.slider("스윙 최대 상승량(m)", 0.01, 0.06, 0.035, 0.001)
    show_csv_preview = st.toggle("변환된 CSV 미리보기", value=False)
    st.info("앱 안의 압력 패널은 ps_0~ps_8 값을 3×3 도식 배치로 표현합니다.")

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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Left 샘플 수", f"{len(left_df):,}")
    m2.metric("Right 샘플 수", f"{len(right_df):,}")
    m3.metric("Left duration(s)", f"{(left_df['ts'].iloc[-1]-left_df['ts'].iloc[0])/1000:.2f}")
    m4.metric("Right duration(s)", f"{(right_df['ts'].iloc[-1]-right_df['ts'].iloc[0])/1000:.2f}")

    if show_csv_preview:
        with st.expander("변환된 CSV 미리보기", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(left_df.head(20), use_container_width=True)
            with c2:
                st.dataframe(right_df.head(20), use_container_width=True)

    if st.button("분석 실행", type="primary", use_container_width=True):
        with st.spinner("분석, 통계량 계산, 다운로드용 영상 생성 중입니다..."):
            result = analyze_and_render(left_df, right_df, cfg)
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

    cmet1, cmet2, cmet3, cmet4, cmet5, cmet6 = st.columns(6)
    cmet1.metric("현재 시각(s)", f"{tf:.2f}")
    cmet2.metric("Left stance", "1" if bool(left_f["stance"][idx]) else "0")
    cmet3.metric("Right stance", "1" if bool(right_f["stance"][idx]) else "0")
    cmet4.metric("Left 전진거리(m)", f"{left_f['pos'][idx, 0]:.2f}")
    cmet5.metric("Right 전진거리(m)", f"{right_f['pos'][idx, 0]:.2f}")
    cmet6.metric("좌우 거리 차(m)", f"{abs(left_f['pos'][idx,0]-right_f['pos'][idx,0]):.2f}")

    top_left, top_right = st.columns([1.65, 1.0])
    with top_left:
        st.plotly_chart(make_motion_figure(frame_store, idx, cfg), use_container_width=True, config={"displaylogo": False})
    with top_right:
        st.markdown("##### 현재 프레임 요약")
        st.dataframe(pd.DataFrame([
            {"side": "left", "stance": int(bool(left_f['stance'][idx])), "a_fwd": float(left_f['a_fwd'][idx]), "rel_yaw_deg": float(left_f['rel_yaw'][idx]), "total_pressure": float(left_f['pressure'][idx]), "gyro_mag_dps": float(left_f['gyro_mag'][idx])},
            {"side": "right", "stance": int(bool(right_f['stance'][idx])), "a_fwd": float(right_f['a_fwd'][idx]), "rel_yaw_deg": float(right_f['rel_yaw'][idx]), "total_pressure": float(right_f['pressure'][idx]), "gyro_mag_dps": float(right_f['gyro_mag'][idx])},
        ]), use_container_width=True, hide_index=True)
        st.info("빨강/초록/파랑 = local x/y/z 축, 보라색 = 자이로 벡터")

    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(
            make_pressure_heatmap(left_f["pressure_matrix"][idx], frame_store["sensor_vmax"], f"LEFT pressure map · t={tf:.2f}s"),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with p2:
        st.plotly_chart(
            make_pressure_heatmap(right_f["pressure_matrix"][idx], frame_store["sensor_vmax"], f"RIGHT pressure map · t={tf:.2f}s"),
            use_container_width=True,
            config={"displaylogo": False},
        )

    st.plotly_chart(
        make_pressure_trend_figure(t_frames, idx, left_f["pressure"], right_f["pressure"], window_sec=cfg.total_trend_sec),
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.markdown("#### 전체 요약 통계")
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("기록 길이(s)", result["pair_summary"]["recording_duration_s"])
    pm2.metric("평균 cadence", result["pair_summary"]["mean_cadence_steps_per_min"])
    pm3.metric("평균 stance(%)", result["pair_summary"]["mean_stance_pct"])
    pm4.metric("거리 비대칭(m)", result["pair_summary"]["distance_symmetry_abs_diff_m"])
    pm5.metric("비디오 길이(s)", round(result["video_duration_s"], 2))
    st.dataframe(result["summary_df"], use_container_width=True)

    with st.expander("CSV 미리보기", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(_csv_preview_from_bytes(result["left_csv_bytes"], 30), use_container_width=True)
        with c2:
            st.dataframe(_csv_preview_from_bytes(result["right_csv_bytes"], 30), use_container_width=True)

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
        st.download_button("analysis.mp4", result["video_bytes"], file_name="smart_insole_analysis.mp4", mime="video/mp4", use_container_width=True)
    with d5:
        st.download_button("전체 ZIP", result["zip_bytes"], file_name="smart_insole_results.zip", mime="application/zip", use_container_width=True)

else:
    st.markdown(
        """
        ### 앱 구조
        - 업로드된 JSON/CSV를 바로 CSV 형태로 정리
        - 분석 장면을 mp4 재생 대신 **앱 안에서 직접 렌더링**
        - 3D 인솔 자세, 압력 히트맵, total pressure 추세를 **프레임 슬라이더**로 탐색
        - 다운로드 버튼은 맨 아래 유지

        ### 현재 제공 내용
        - 고정 전진축 기반 전진/정지 추정
        - 쿼터니언 기반 3D 인솔 자세
        - 자이로 벡터 시각화
        - 좌/우 압력 히트맵 및 total pressure 추세
        - 요약 통계량, CSV, mp4, ZIP 다운로드
        """
    )
