from __future__ import annotations

import io
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

# ------------------------------
# Data parsing
# ------------------------------

REQUIRED_BASE_COLUMNS = [
    "ts", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "quat_0", "quat_1", "quat_2", "quat_3",
]
PRESSURE_COLS = [f"ps_{i}" for i in range(9)]


def _is_record_like(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    if "semiRawImuPs" in x:
        return True
    keys = set(x.keys())
    return {"ts", "acc", "gyro"}.issubset(keys) or {"ts", "quat"}.issubset(keys)



def _find_candidate_record_lists(obj: Any, path: str = "root") -> List[Tuple[str, list]]:
    out: List[Tuple[str, list]] = []
    if isinstance(obj, list):
        if obj and all(_is_record_like(item) for item in obj[: min(5, len(obj))]):
            out.append((path, obj))
        for i, item in enumerate(obj[:20]):
            out.extend(_find_candidate_record_lists(item, f"{path}[{i}]"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_find_candidate_record_lists(v, f"{path}.{k}"))
    return out



def _normalize_record(d: dict) -> dict | None:
    if "semiRawImuPs" in d:
        semi = d.get("semiRawImuPs") or {}
        acc = semi.get("acc", {}) or {}
        gyro = semi.get("gyro", {}) or {}
        ps = semi.get("ps")
        quat = semi.get("quat")
        if ps is None or quat is None:
            return None
        row = {
            "ts": d.get("ts"),
            "acc_x": acc.get("x"),
            "acc_y": acc.get("y"),
            "acc_z": acc.get("z"),
            "gyro_x": gyro.get("x"),
            "gyro_y": gyro.get("y"),
            "gyro_z": gyro.get("z"),
        }
        for i, v in enumerate(ps[:9]):
            row[f"ps_{i}"] = v
        for i, v in enumerate(quat[:4]):
            row[f"quat_{i}"] = v
        return row

    acc = d.get("acc", {}) or {}
    gyro = d.get("gyro", {}) or {}
    ps = d.get("ps")
    quat = d.get("quat")
    if ps is None or quat is None:
        # maybe already flattened
        row = {k: d.get(k) for k in REQUIRED_BASE_COLUMNS + PRESSURE_COLS}
        if row.get("ts") is None:
            return None
        return row

    row = {
        "ts": d.get("ts"),
        "acc_x": acc.get("x"),
        "acc_y": acc.get("y"),
        "acc_z": acc.get("z"),
        "gyro_x": gyro.get("x"),
        "gyro_y": gyro.get("y"),
        "gyro_z": gyro.get("z"),
    }
    for i, v in enumerate(ps[:9]):
        row[f"ps_{i}"] = v
    for i, v in enumerate(quat[:4]):
        row[f"quat_{i}"] = v
    return row



def records_to_df(records: list, side_name: str) -> pd.DataFrame:
    rows = []
    for d in records:
        row = _normalize_record(d)
        if row is None:
            continue
        row["side"] = side_name
        rows.append(row)
    if not rows:
        raise ValueError(f"{side_name} 데이터에서 유효한 레코드를 찾지 못했습니다.")

    df = pd.DataFrame(rows)
    for col in REQUIRED_BASE_COLUMNS + PRESSURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    keep_cols = ["side", "ts"] + [c for c in df.columns if c not in {"side", "ts"}]
    df = df[keep_cols].copy()
    numeric_cols = [c for c in df.columns if c != "side"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["ts", "quat_0", "quat_1", "quat_2", "quat_3"]).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df



def parse_combined_json(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    obj = json.loads(file_bytes.decode("utf-8"))
    meta: Dict[str, Any] = {"mode": "combined_json"}

    # explicit keys first
    left_records = None
    right_records = None
    if isinstance(obj, dict):
        if isinstance(obj.get("leftInsole"), dict) and isinstance(obj["leftInsole"].get("data"), list):
            left_records = obj["leftInsole"]["data"]
            meta["left_path"] = "root.leftInsole.data"
        elif isinstance(obj.get("left"), list):
            left_records = obj["left"]
            meta["left_path"] = "root.left"

        if isinstance(obj.get("rightInsole"), dict) and isinstance(obj["rightInsole"].get("data"), list):
            right_records = obj["rightInsole"]["data"]
            meta["right_path"] = "root.rightInsole.data"
        elif isinstance(obj.get("right"), list):
            right_records = obj["right"]
            meta["right_path"] = "root.right"

    if left_records is None or right_records is None:
        cands = _find_candidate_record_lists(obj)
        if len(cands) < 2:
            raise ValueError("left/right 인솔 레코드 리스트를 JSON에서 자동으로 찾지 못했습니다.")
        # best effort: prefer names containing left/right
        left_candidates = [x for x in cands if "left" in x[0].lower()]
        right_candidates = [x for x in cands if "right" in x[0].lower()]
        if left_records is None:
            left_records = (left_candidates[0] if left_candidates else cands[0])[1]
            meta["left_path"] = (left_candidates[0] if left_candidates else cands[0])[0]
        if right_records is None:
            fallback = right_candidates[0] if right_candidates else cands[1]
            right_records = fallback[1]
            meta["right_path"] = fallback[0]

    left_df = records_to_df(left_records, "left")
    right_df = records_to_df(right_records, "right")
    return left_df, right_df, meta



def parse_single_side_json(file_bytes: bytes, side_name: str) -> pd.DataFrame:
    obj = json.loads(file_bytes.decode("utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        records = obj["data"]
    elif isinstance(obj, list):
        records = obj
    else:
        cands = _find_candidate_record_lists(obj)
        if not cands:
            raise ValueError(f"{side_name} JSON에서 레코드 리스트를 찾지 못했습니다.")
        records = cands[0][1]
    return records_to_df(records, side_name)



def validate_insole_df(df: pd.DataFrame, side_name: str) -> pd.DataFrame:
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    missing = [c for c in REQUIRED_BASE_COLUMNS + PRESSURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{side_name} CSV에 필수 컬럼이 없습니다: {missing}")
    if "side" not in df.columns:
        df["side"] = side_name
    numeric_cols = [c for c in df.columns if c != "side"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["ts", "quat_0", "quat_1", "quat_2", "quat_3"]).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


# ------------------------------
# Motion estimation and video
# ------------------------------

@dataclass
class RenderConfig:
    target_fps: int = 15
    width: int = 1480
    height: int = 860
    motion_width: int = 900
    foot_length: float = 0.25
    foot_width: float = 0.095
    foot_thick: float = 0.018
    step_width: float = 0.20
    base_z: float = 0.009
    lift_max: float = 0.035
    gyro_arrow_scale: float = 0.00055
    gyro_arrow_max: float = 0.10
    trail_sec: float = 1.2
    pressure_thr_frac: float = 0.22
    gyro_stance_dps: float = 38.0
    min_stance_samples: int = 8
    sg_window: int = 31
    sg_poly: int = 3
    yaw_hp_window: int = 401
    rel_yaw_clip_deg: float = 18.0
    total_trend_sec: float = 6.0


PRESSURE_LAYOUT = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
BG = (255, 255, 255)
BLACK = (20, 20, 20)
DARK = (60, 60, 60)
BLUE = (220, 120, 40)     # BGR for left
ORANGE = (40, 140, 240)   # BGR for right
RED = (60, 60, 220)
GREEN = (60, 170, 60)
ROYAL = (220, 120, 40)
PURPLE = (180, 80, 180)
GRAY = (180, 180, 180)
LIGHT = (232, 232, 232)
PALE = (247, 247, 247)



def moving_savgol(x, window=31, poly=3):
    x = np.asarray(x, dtype=float)
    if len(x) < 7:
        return x.copy()
    win = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if win < 7:
        return x.copy()
    poly = min(poly, win - 2)
    return savgol_filter(x, win, poly, mode="interp")



def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0] = 1.0
    q = q / n
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i - 1], out[i]) < 0:
            out[i] *= -1.0
    return out



def contiguous_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))



def make_box_vertices(length=0.25, width=0.095, thickness=0.018):
    L = length / 2.0
    W = width / 2.0
    T = thickness / 2.0
    verts = np.array([
        [-L, -W, -T], [L, -W, -T], [L, W, -T], [-L, W, -T],
        [-L, -W, T], [L, -W, T], [L, W, T], [-L, W, T],
    ], dtype=float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return verts, edges



def load_insole_df_for_motion(df: pd.DataFrame):
    df = df.copy().sort_values("ts").reset_index(drop=True)
    ts = df["ts"].to_numpy(dtype=float)
    dt = np.diff(ts, prepend=ts[0]) / 1000.0
    med = float(np.median(dt[1:])) if len(dt) > 1 else 0.005
    dt[0] = med
    dt = np.clip(dt, med * 0.5, med * 3.0)
    t = np.cumsum(dt) - dt[0]

    q = normalize_quaternion(df[[f"quat_{i}" for i in range(4)]].to_numpy(dtype=float))
    rot_raw = R.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))
    pressure_matrix = df[PRESSURE_COLS].to_numpy(dtype=float)
    return {
        "t": t,
        "dt": dt,
        "acc": df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float),
        "gyro": df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float),
        "pressure": pressure_matrix.sum(axis=1),
        "pressure_matrix": pressure_matrix,
        "rot_raw": rot_raw,
    }



def detect_stance(pressure, gyro, cfg: RenderConfig):
    p_s = moving_savgol(pressure, 21, 3)
    p10, p90 = np.quantile(p_s, [0.10, 0.90])
    p_thr = p10 + cfg.pressure_thr_frac * (p90 - p10)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    g_s = moving_savgol(gyro_mag, 21, 3)
    stance = (p_s > p_thr) & (g_s < cfg.gyro_stance_dps)
    cleaned = stance.copy()
    for s, e in contiguous_true_segments(stance):
        if e - s < cfg.min_stance_samples:
            cleaned[s:e] = False
    return cleaned



def build_fixed_forward_rotation(rot_raw, cfg: RenderConfig):
    eul = rot_raw.as_euler('ZYX', degrees=False)  # yaw pitch roll
    yaw = np.unwrap(eul[:, 0])
    pitch = eul[:, 1]
    roll = eul[:, 2]
    drift = moving_savgol(yaw, cfg.yaw_hp_window, 3)
    rel_yaw = yaw - drift
    rel_yaw = np.clip(rel_yaw, -np.deg2rad(cfg.rel_yaw_clip_deg), np.deg2rad(cfg.rel_yaw_clip_deg))
    rot_vis = R.from_euler('ZYX', np.column_stack([rel_yaw, pitch, roll]), degrees=False)
    return rot_vis, np.rad2deg(rel_yaw)



def estimate_forward_only_motion(df: pd.DataFrame, y_offset: float, cfg: RenderConfig):
    pkg = load_insole_df_for_motion(df)
    stance = detect_stance(pkg["pressure"], pkg["gyro"], cfg)
    rot_vis, rel_yaw_deg = build_fixed_forward_rotation(pkg["rot_raw"], cfg)
    acc_world = rot_vis.apply(pkg["acc"])
    acc_world = np.column_stack([moving_savgol(acc_world[:, i], cfg.sg_window, cfg.sg_poly) for i in range(3)])
    bias = np.median(acc_world[stance], axis=0) if np.any(stance) else np.median(acc_world, axis=0)
    acc_world = acc_world - bias

    a_fwd = moving_savgol(acc_world[:, 0], 41, 3)
    a_up = moving_savgol(acc_world[:, 2], 41, 3)

    n = len(pkg["t"])
    v = np.zeros(n)
    x = np.zeros(n)
    for i in range(1, n):
        if stance[i]:
            v[i] = 0.0
        else:
            v[i] = max(0.0, v[i - 1] + a_fwd[i] * pkg["dt"][i])
        x[i] = x[i - 1] + v[i] * pkg["dt"][i]

    p_s = moving_savgol(pkg["pressure"], 31, 3)
    p95 = np.quantile(p_s, 0.95) if np.quantile(p_s, 0.95) > 1e-8 else 1.0
    p_norm = np.clip(p_s / p95, 0.0, 1.0)
    swing = 1.0 - moving_savgol(stance.astype(float), 31, 3)
    up_term = np.clip(a_up, 0.0, None)
    up98 = np.quantile(up_term, 0.98)
    if up98 > 1e-8:
        up_term = up_term / up98
    lift = cfg.lift_max * np.clip(0.65 * swing + 0.25 * (1.0 - p_norm) + 0.10 * up_term, 0.0, 1.0)
    lift = moving_savgol(lift, 31, 3)
    z = cfg.base_z + lift
    y = np.full(n, y_offset)
    pos = np.column_stack([x, y, z])

    return {
        "t": pkg["t"],
        "dt": pkg["dt"],
        "pos": pos,
        "rot": rot_vis,
        "stance": stance,
        "a_fwd": a_fwd,
        "rel_yaw_deg": rel_yaw_deg,
        "gyro": pkg["gyro"],
        "gyro_mag": np.linalg.norm(pkg["gyro"], axis=1),
        "pressure": pkg["pressure"],
        "pressure_matrix": pkg["pressure_matrix"],
        "v_fwd": v,
        "x_fwd": x,
        "distance_m": float(x[-1]),
    }





def estimate_stationary_motion(df: pd.DataFrame, y_offset: float, cfg: RenderConfig):
    """제자리(전진 제거) 시각화를 위한 트랙 생성."""
    tr = estimate_forward_only_motion(df, y_offset=y_offset, cfg=cfg)
    tr["pos"] = tr["pos"].copy()
    tr["pos"][:, 0] = 0.0
    tr["distance_m"] = 0.0
    if "x_fwd" in tr:
        tr["x_fwd"] = tr["x_fwd"].copy()
        tr["x_fwd"][:] = 0.0
    return tr


def build_sampler(track, t_frames):
    t = track["t"]
    slerp = Slerp(t, track["rot"])
    rot_frames = slerp(t_frames)
    pos_frames = np.column_stack([np.interp(t_frames, t, track["pos"][:, i]) for i in range(3)])
    stance_frames = np.interp(t_frames, t, track["stance"].astype(float)) > 0.5
    a_fwd_frames = np.interp(t_frames, t, track["a_fwd"])
    v_src = track["v_fwd"] if "v_fwd" in track else np.zeros_like(t)
    v_fwd_frames = np.interp(t_frames, t, v_src)
    rel_yaw_frames = np.interp(t_frames, t, track["rel_yaw_deg"])
    gyro_mag_frames = np.interp(t_frames, t, track["gyro_mag"])
    pressure_frames = np.interp(t_frames, t, track["pressure"])
    gyro_frames = np.column_stack([np.interp(t_frames, t, track["gyro"][:, i]) for i in range(3)])
    ps_frames = np.column_stack([np.interp(t_frames, t, track["pressure_matrix"][:, i]) for i in range(track["pressure_matrix"].shape[1])])
    return {
        "rot": rot_frames,
        "pos": pos_frames,
        "stance": stance_frames,
        "a_fwd": a_fwd_frames,
        "v_fwd": v_fwd_frames,
        "rel_yaw": rel_yaw_frames,
        "gyro_mag": gyro_mag_frames,
        "pressure": pressure_frames,
        "pressure_matrix": ps_frames,
        "gyro": gyro_frames,
    }



def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    zaxis = target - eye
    zaxis = zaxis / (np.linalg.norm(zaxis) + 1e-12)
    xaxis = np.cross(zaxis, up)
    xaxis = xaxis / (np.linalg.norm(xaxis) + 1e-12)
    yaxis = np.cross(xaxis, zaxis)
    Rwc = np.vstack([xaxis, yaxis, zaxis])
    return Rwc, eye



def project_points(points, eye, target, width, height, focal=1120.0, x0=0, y0=0):
    Rwc, eye = look_at(eye, target)
    p = np.asarray(points, dtype=float)
    pc = (Rwc @ (p - eye).T).T
    z = pc[:, 2]
    ok = z > 0.05
    uv = np.full((len(points), 2), np.nan)
    uv[ok, 0] = x0 + width / 2 + focal * (pc[ok, 0] / z[ok])
    uv[ok, 1] = y0 + height / 2 - focal * (pc[ok, 1] / z[ok])
    return uv, ok, pc



def draw_polyline(img, pts, color, thickness=1, alpha=1.0):
    pts = np.asarray(pts, dtype=float)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 2:
        return
    overlay = img.copy()
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts_i], False, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)



def draw_line(img, p1, p2, color, thickness=2):
    if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
        return
    cv2.line(img, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), color, thickness, cv2.LINE_AA)



def draw_text_block(img, lines, origin, line_h=28, color=BLACK, bg=(255, 255, 255), alpha=0.92, scale=0.7):
    x, y = origin
    widths = []
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        widths.append(w)
    box_w = max(widths) + 22
    box_h = line_h * len(lines) + 14
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - 24), (x + box_w, y - 24 + box_h), bg, -1)
    cv2.rectangle(overlay, (x, y - 24), (x + box_w, y - 24 + box_h), LIGHT, 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    yy = y
    for line in lines:
        cv2.putText(img, line, (x + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        yy += line_h



def pressure_color(val, vmax):
    ratio = float(np.clip(val / max(vmax, 1e-8), 0.0, 1.0))
    gray = np.uint8(round(255 * ratio))
    color = cv2.applyColorMap(np.array([[gray]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return tuple(int(c) for c in color.tolist())



def draw_panel_box(img, rect, title, title_color=BLACK):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PALE, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), LIGHT, 1)
    if title:
        cv2.putText(img, title, (x + 14, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, title_color, 2, cv2.LINE_AA)



def draw_pressure_heatmap(img, rect, sensor_vals, vmax, title, accent, total_pressure, stance_bool):
    x, y, w, h = rect
    draw_panel_box(img, rect, title, accent)
    status = "STANCE" if stance_bool else "SWING"
    cv2.putText(img, status, (x + w - 115, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                accent if stance_bool else DARK, 2, cv2.LINE_AA)

    grid_x = x + 20
    grid_y = y + 58
    grid_h = h - 120
    cell_gap = 10
    cell_w = int((w - 40 - 2 * cell_gap) / 3)
    cell_h = int((grid_h - 2 * cell_gap) / 3)

    vals = np.asarray(sensor_vals, dtype=float)
    vals_grid = vals[PRESSURE_LAYOUT]

    for r in range(3):
        for c in range(3):
            idx = PRESSURE_LAYOUT[r, c]
            val = vals_grid[r, c]
            x1 = grid_x + c * (cell_w + cell_gap)
            y1 = grid_y + r * (cell_h + cell_gap)
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            color = pressure_color(val, vmax)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), LIGHT, 2)
            cv2.putText(img, f"ps_{idx}", (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 1, cv2.LINE_AA)
            txt_color = (255, 255, 255) if val > 0.55 * vmax else BLACK
            cv2.putText(img, f"{val:.2f}", (x1 + 8, y1 + cell_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, txt_color, 2, cv2.LINE_AA)

    cv2.putText(img, f"total pressure = {total_pressure:.2f}", (x + 18, y + h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLACK, 1, cv2.LINE_AA)



def draw_pressure_trend(img, rect, t_frames, i, left_total, right_total, vmax, cfg: RenderConfig):
    x, y, w, h = rect
    draw_panel_box(img, rect, "", BLACK)
    cv2.putText(img, "Total pressure trend", (x + 14, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, "LEFT", (x + 18, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.54, BLUE, 2, cv2.LINE_AA)
    cv2.putText(img, "RIGHT", (x + 88, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.54, ORANGE, 2, cv2.LINE_AA)
    cv2.putText(img, f"{cfg.total_trend_sec:.0f}s window", (x + w - 118, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, DARK, 1, cv2.LINE_AA)

    x1, y1 = x + 20, y + 66
    x2, y2 = x + w - 18, y + h - 24
    cv2.rectangle(img, (x1, y1), (x2, y2), (252, 252, 252), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), LIGHT, 1)

    t_now = t_frames[i]
    t_min = max(0.0, t_now - cfg.total_trend_sec)
    t_max = max(t_min + 1e-6, t_now)

    for frac in np.linspace(0.0, 1.0, 5):
        yy = int(round(y2 - frac * (y2 - y1)))
        cv2.line(img, (x1, yy), (x2, yy), LIGHT, 1, cv2.LINE_AA)

    def plot_series(values, color):
        mask = (t_frames >= t_min) & (t_frames <= t_max)
        tt = t_frames[mask]
        vv = values[mask]
        if len(tt) < 2:
            return
        xs = x1 + (tt - t_min) / max(t_max - t_min, 1e-6) * (x2 - x1)
        ys = y2 - np.clip(vv / max(vmax, 1e-8), 0.0, 1.0) * (y2 - y1)
        draw_polyline(img, np.column_stack([xs, ys]), color, thickness=2, alpha=1.0)

    plot_series(left_total, BLUE)
    plot_series(right_total, ORANGE)
    cv2.line(img, (x2, y1), (x2, y2), BLACK, 1, cv2.LINE_AA)
    cv2.putText(img, "0", (x1 + 4, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, DARK, 1, cv2.LINE_AA)
    cv2.putText(img, f"{vmax:.1f}", (x1 + 4, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, DARK, 1, cv2.LINE_AA)



def render_video(left_track, right_track, out_mp4: str | Path, cfg: RenderConfig, mode: str = "forward"):
    width, height, motion_w = cfg.width, cfg.height, cfg.motion_width
    panel_x0 = motion_w + 14
    panel_w = width - panel_x0 - 18

    total_t = min(left_track["t"][-1], right_track["t"][-1])
    t_frames = np.arange(0, total_t, 1.0 / cfg.target_fps)
    if len(t_frames) == 0 or t_frames[-1] < total_t:
        t_frames = np.append(t_frames, total_t)

    left_f = build_sampler(left_track, t_frames)
    right_f = build_sampler(right_track, t_frames)
    verts_local, edges = make_box_vertices(cfg.foot_length, cfg.foot_width, cfg.foot_thick)
    trail_n = max(2, int(cfg.trail_sec * cfg.target_fps))
    left_sensor_q99 = np.quantile(left_track["pressure_matrix"], 0.995)
    right_sensor_q99 = np.quantile(right_track["pressure_matrix"], 0.995)
    sensor_vmax = max(left_sensor_q99, right_sensor_q99, 1e-6)
    total_vmax = max(np.quantile(left_track["pressure"], 0.995), np.quantile(right_track["pressure"], 0.995), 1e-6)

    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*'mp4v'), cfg.target_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter open failed")

    for i, tf in enumerate(t_frames):
        img = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (motion_w, 0), (motion_w, height), LIGHT, 2)

        l_center = left_f["pos"][i]
        r_center = right_f["pos"][i]
        x_now = max(l_center[0], r_center[0])
        target = np.array([x_now + 0.45, 0.0, 0.05])
        eye = np.array([x_now - 1.10, -1.35, 0.72])

        x_min, x_max = x_now - 1.2, x_now + 0.9
        for gx in np.linspace(x_min, x_max, 16):
            pts = np.array([[gx, -0.25, 0.0], [gx, 0.25, 0.0]])
            uv, _, _ = project_points(pts, eye, target, motion_w - 20, height - 20, x0=10, y0=10)
            draw_line(img, uv[0], uv[1], LIGHT, 1)
        for gy in np.linspace(-0.25, 0.25, 6):
            pts = np.array([[x_min, gy, 0.0], [x_max, gy, 0.0]])
            uv, _, _ = project_points(pts, eye, target, motion_w - 20, height - 20, x0=10, y0=10)
            draw_line(img, uv[0], uv[1], LIGHT, 1)
        for gy, col in [(0.0, BLACK), (cfg.step_width / 2, BLUE), (-cfg.step_width / 2, ORANGE)]:
            pts = np.array([[x_min, gy, 0.0], [x_max, gy, 0.0]])
            uv, _, _ = project_points(pts, eye, target, motion_w - 20, height - 20, x0=10, y0=10)
            draw_line(img, uv[0], uv[1], col, 2 if gy == 0.0 else 1)

        def draw_foot(center, rot, trail_world, gyro_body, color, label, stance_bool):
            verts_world = rot.apply(verts_local) + center
            uv, ok, _ = project_points(verts_world, eye, target, motion_w - 20, height - 20, x0=10, y0=10)
            for a, b in edges:
                if ok[a] and ok[b]:
                    draw_line(img, uv[a], uv[b], color, 3 if stance_bool else 2)
            if mode != "stationary":
                trail_uv, _, _ = project_points(trail_world, eye, target, motion_w - 20, height - 20, x0=10, y0=10)
                draw_polyline(img, trail_uv, color, thickness=2, alpha=0.40)

            axis_len = 0.07
            axes_world = rot.apply(np.eye(3) * axis_len) + center
            axes_uv, axes_ok, _ = project_points(np.vstack([center, axes_world]), eye, target, motion_w - 20, height - 20, x0=10, y0=10)
            if axes_ok[0] and axes_ok[1]: draw_line(img, axes_uv[0], axes_uv[1], RED, 2)
            if axes_ok[0] and axes_ok[2]: draw_line(img, axes_uv[0], axes_uv[2], GREEN, 2)
            if axes_ok[0] and axes_ok[3]: draw_line(img, axes_uv[0], axes_uv[3], ROYAL, 2)

            g = np.asarray(gyro_body, dtype=float) * cfg.gyro_arrow_scale
            g_len = np.linalg.norm(g)
            if g_len > cfg.gyro_arrow_max:
                g *= cfg.gyro_arrow_max / g_len
            base = center + np.array([0.0, 0.0, 0.015])
            tip = base + rot.apply(g)
            g_uv, g_ok, _ = project_points(np.vstack([base, tip]), eye, target, motion_w - 20, height - 20, x0=10, y0=10)
            if g_ok[0] and g_ok[1]:
                draw_line(img, g_uv[0], g_uv[1], PURPLE, 3)

            c_uv, c_ok, _ = project_points(np.array([center]), eye, target, motion_w - 20, height - 20, x0=10, y0=10)
            if c_ok[0]:
                pos = tuple(np.round(c_uv[0] + np.array([10, -10])).astype(int))
                cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        draw_foot(l_center, left_f["rot"][i], left_f["pos"][max(0, i - trail_n):i + 1], left_f["gyro"][i], BLUE, 'LEFT', bool(left_f["stance"][i]))
        draw_foot(r_center, right_f["rot"][i], right_f["pos"][max(0, i - trail_n):i + 1], right_f["gyro"][i], ORANGE, 'RIGHT', bool(right_f["stance"][i]))

        cv2.putText(img, 'Smart Insole Motion Analysis', (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLACK, 2, cv2.LINE_AA)
        if mode == 'stationary':
            cv2.putText(img, 'Stationary view | Quaternion orientation | Gyro vector | Pressure heatmap',
                        (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.60, DARK, 1, cv2.LINE_AA)
            info_lines = [
                f't = {tf:5.2f} s',
                f'LEFT  | stance={int(left_f["stance"][i])} | v_fwd={left_f["v_fwd"][i]:5.2f} m/s | a_fwd={left_f["a_fwd"][i]:6.2f} m/s^2 | totalP={left_f["pressure"][i]:5.2f}',
                f'RIGHT | stance={int(right_f["stance"][i])} | v_fwd={right_f["v_fwd"][i]:5.2f} m/s | a_fwd={right_f["a_fwd"][i]:6.2f} m/s^2 | totalP={right_f["pressure"][i]:5.2f}',
            ]
        else:
            cv2.putText(img, 'Fixed forward axis | Quaternion orientation | Acceleration based forward motion | Pressure heatmap',
                        (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.60, DARK, 1, cv2.LINE_AA)
            info_lines = [
                f't = {tf:5.2f} s',
                f'LEFT  | stance={int(left_f["stance"][i])} | a_fwd={left_f["a_fwd"][i]:6.2f} m/s^2 | yaw_rel={left_f["rel_yaw"][i]:5.1f} deg | totalP={left_f["pressure"][i]:5.2f}',
                f'RIGHT | stance={int(right_f["stance"][i])} | a_fwd={right_f["a_fwd"][i]:6.2f} m/s^2 | yaw_rel={right_f["rel_yaw"][i]:5.1f} deg | totalP={right_f["pressure"][i]:5.2f}',
                f'Forward distance  L={l_center[0]:.2f} m, R={r_center[0]:.2f} m',
            ]
        draw_text_block(img, info_lines, (28, 120), line_h=30, scale=0.63)
        draw_text_block(img, ['red = local x', 'green = local y', 'blue = local z', 'purple = gyro'], (670, 118), line_h=26, scale=0.58)

        left_rect = (panel_x0 + 6, 24, panel_w - 12, 230)
        right_rect = (panel_x0 + 6, 268, panel_w - 12, 230)
        trend_rect = (panel_x0 + 6, 512, panel_w - 12, 206)
        note_rect = (panel_x0 + 6, 734, panel_w - 12, 92)
        draw_pressure_heatmap(img, left_rect, left_f["pressure_matrix"][i], sensor_vmax, "LEFT pressure map", BLUE, left_f["pressure"][i], bool(left_f["stance"][i]))
        draw_pressure_heatmap(img, right_rect, right_f["pressure_matrix"][i], sensor_vmax, "RIGHT pressure map", ORANGE, right_f["pressure"][i], bool(right_f["stance"][i]))
        draw_pressure_trend(img, trend_rect, t_frames, i, left_f["pressure"], right_f["pressure"], total_vmax, cfg)
        draw_text_block(img, ['Pressure panel uses schematic 3x3 placement of ps_0~ps_8.', 'Color scale is shared across left/right.'], (note_rect[0], note_rect[1] + 26), line_h=26, scale=0.54)
        writer.write(img)

    writer.release()
    return len(t_frames), float(t_frames[-1]), str(out_mp4)


# ------------------------------
# Summary stats and packaging
# ------------------------------


def _safe_fs_hz(ts_ms: np.ndarray) -> float:
    if len(ts_ms) < 2:
        return float("nan")
    dt_ms = np.diff(ts_ms)
    med = np.median(dt_ms[dt_ms > 0]) if np.any(dt_ms > 0) else np.nan
    return float(1000.0 / med) if med and not np.isnan(med) else float("nan")



def _count_steps_from_stance(t: np.ndarray, stance: np.ndarray) -> Tuple[int, float]:
    segs = contiguous_true_segments(stance)
    if not segs:
        return 0, float("nan")
    starts = np.array([s for s, _ in segs], dtype=int)
    # remove too-close starts (<0.25s)
    kept = [starts[0]]
    for s in starts[1:]:
        if t[s] - t[kept[-1]] >= 0.25:
            kept.append(s)
    step_count = len(kept)
    if len(kept) >= 2:
        intervals = np.diff(t[kept])
        median_step_time = float(np.median(intervals))
    else:
        median_step_time = float("nan")
    return step_count, median_step_time



def summarize_side(df: pd.DataFrame, track: dict, side_name: str) -> Dict[str, Any]:
    ts = df["ts"].to_numpy(dtype=float)
    duration = (ts[-1] - ts[0]) / 1000.0 if len(ts) > 1 else 0.0
    fs = _safe_fs_hz(ts)
    pressure = track["pressure"]
    gyro_mag = track["gyro_mag"]
    stance = track["stance"]
    step_count, median_step_time = _count_steps_from_stance(track["t"], stance)
    cadence = step_count / duration * 60.0 if duration > 0 else float("nan")
    return {
        "side": side_name,
        "samples": int(len(df)),
        "duration_s": round(float(duration), 3),
        "fs_hz": round(float(fs), 3) if not np.isnan(fs) else np.nan,
        "forward_distance_m": round(float(track["distance_m"]), 3),
        "stance_pct": round(float(np.mean(stance) * 100.0), 2),
        "swing_pct": round(float((1.0 - np.mean(stance)) * 100.0), 2),
        "step_count": int(step_count),
        "cadence_steps_per_min": round(float(cadence), 2) if not np.isnan(cadence) else np.nan,
        "median_step_time_s": round(float(median_step_time), 3) if not np.isnan(median_step_time) else np.nan,
        "pressure_mean": round(float(np.mean(pressure)), 4),
        "pressure_max": round(float(np.max(pressure)), 4),
        "gyro_mean_dps": round(float(np.mean(gyro_mag)), 4),
        "gyro_peak_dps": round(float(np.max(gyro_mag)), 4),
        "acc_forward_mean": round(float(np.mean(track["a_fwd"])), 4),
        "acc_forward_peak": round(float(np.max(np.abs(track["a_fwd"]))), 4),
        "rel_yaw_abs_mean_deg": round(float(np.mean(np.abs(track["rel_yaw_deg"]))), 4),
        "rel_yaw_abs_peak_deg": round(float(np.max(np.abs(track["rel_yaw_deg"]))), 4),
    }



def summarize_pair(left_df: pd.DataFrame, right_df: pd.DataFrame, left_track: dict, right_track: dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    left_summary = summarize_side(left_df, left_track, "left")
    right_summary = summarize_side(right_df, right_track, "right")
    pair_summary = {
        "recording_duration_s": round(min(left_summary["duration_s"], right_summary["duration_s"]), 3),
        "distance_symmetry_abs_diff_m": round(abs(left_summary["forward_distance_m"] - right_summary["forward_distance_m"]), 3),
        "step_count_abs_diff": abs(left_summary["step_count"] - right_summary["step_count"]),
        "mean_cadence_steps_per_min": round(np.nanmean([left_summary["cadence_steps_per_min"], right_summary["cadence_steps_per_min"]]), 2),
        "mean_stance_pct": round(np.mean([left_summary["stance_pct"], right_summary["stance_pct"]]), 2),
    }
    summary_df = pd.DataFrame([left_summary, right_summary])
    return summary_df, pair_summary



def analyze_and_render(left_df: pd.DataFrame, right_df: pd.DataFrame, cfg: RenderConfig | None = None, mode: str = "stationary") -> Dict[str, Any]:
    cfg = cfg or RenderConfig()
    left_df = validate_insole_df(left_df, "left")
    right_df = validate_insole_df(right_df, "right")
    if mode not in {"stationary", "forward"}:
        raise ValueError("mode must be 'stationary' or 'forward'")

    if mode == "stationary":
        left_track = estimate_stationary_motion(left_df, y_offset=+cfg.step_width / 2, cfg=cfg)
        right_track = estimate_stationary_motion(right_df, y_offset=-cfg.step_width / 2, cfg=cfg)
    else:
        left_track = estimate_forward_only_motion(left_df, y_offset=+cfg.step_width / 2, cfg=cfg)
        right_track = estimate_forward_only_motion(right_df, y_offset=-cfg.step_width / 2, cfg=cfg)
    summary_df, pair_summary = summarize_pair(left_df, right_df, left_track, right_track)

    tmpdir = tempfile.mkdtemp(prefix="smart_insole_")
    out_mp4 = Path(tmpdir) / "smart_insole_analysis.mp4"
    frames, duration, out_path = render_video(left_track, right_track, out_mp4, cfg, mode=mode)
    with open(out_path, "rb") as f:
        video_bytes = f.read()

    left_csv = left_df.to_csv(index=False).encode("utf-8-sig")
    right_csv = right_df.to_csv(index=False).encode("utf-8-sig")
    summary_csv = summary_df.to_csv(index=False).encode("utf-8-sig")
    summary_json = json.dumps({"pair_summary": pair_summary, "side_summary": summary_df.to_dict(orient="records")}, ensure_ascii=False, indent=2).encode("utf-8")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("left.csv", left_csv)
        zf.writestr("right.csv", right_csv)
        zf.writestr("summary.csv", summary_csv)
        zf.writestr("summary.json", summary_json)
        zf.writestr("smart_insole_analysis.mp4", video_bytes)
    zip_bytes = zip_buffer.getvalue()

    return {
        "video_bytes": video_bytes,
        "left_csv_bytes": left_csv,
        "right_csv_bytes": right_csv,
        "summary_df": summary_df,
        "pair_summary": pair_summary,
        "summary_csv_bytes": summary_csv,
        "summary_json_bytes": summary_json,
        "zip_bytes": zip_bytes,
        "frames": frames,
        "video_duration_s": duration,
        "left_track": left_track,
        "right_track": right_track,
    }
