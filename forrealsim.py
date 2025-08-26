import cv2
import numpy as np
import time
import random
import math
from collections import deque
LISTROAD = []
# ========================== 視覺與造型（黑白） ==========================
WIDTH, HEIGHT = 1200, 800
DOT_RADIUS = 10
BG_COLOR = (0, 0, 0)          # 背景黑
FG_COLOR = (255, 255, 255)      # 白點

# 外框與標註樣式（灰階）
BORDER_THICK = 4
BORDER_COLOR = (200, 200, 200)
CORNER_LEN = 24
HUD_TEXT = (180, 180, 180)
HUD_BG = (20, 20, 20)
LABEL_MARGIN = 10
RIGHT_PANEL_W = 260
# 軌跡顏色
TRAJECTORY_COLORS = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (220, 220, 100)]


# ========================== 模擬玩家位置（P1滑鼠 + P2~P4虛擬滑鼠） ==========================
positions = [(200, 300), (400, 300), (600, 300), (400, 150)]
velocities = [(0.0, 0.0) for _ in range(4)]
next_turn_time = [0.0 for _ in range(4)]
target_dir = [(0.0, 0.0) for _ in range(4)]

# ---- 可調運動參數（手感）----
MAX_SPEED = 900.0       # 最大速度 px/s
ACCEL = 2800.0          # 加速度 px/s^2
JITTER = 450.0          # 抖動加速度 px/s^2
FRICTION = 3.5          # 摩擦 1/s（指數衰減）
TURN_MIN, TURN_MAX = 0.20, 0.80
EDGE_PAD = DOT_RADIUS * 2
DT_CLAMP = 1/60

# ========================== MCU 分時槽（每人 3ms，含保護帶） ==========================
SLOT_MS = 3.0                   # 總時槽長度
ACTIVE_MS = 2.5                 # 有效訊號時間 (IR ON)
GUARD_MS = SLOT_MS - ACTIVE_MS  # 保護帶時間 (IR OFF)

SLOT_TIME = SLOT_MS / 1000.0
ACTIVE_TIME = ACTIVE_MS / 1000.0 # 有效訊號時間（秒）

NUM_PLAYERS = 4


# ========================== 偵測 + 追蹤 + 相位（自動標註） ==========================
class Track:
    def __init__(self, tid, pos, now_t, hist_len=160, traj_len=50):
        self.tid = tid
        self.pos = np.array(pos, dtype=np.float32)
        self.last_seen = now_t
        self.visible_hist = deque(maxlen=hist_len)
        self.pos_hist = deque(maxlen=traj_len)      # 歷史位置佇列，用於繪製軌跡
        self.phase_est = None
        self.label = None

    def update_pos(self, pos, now_t):
        self.pos = 0.7 * self.pos + 0.3 * np.array(pos, dtype=np.float32)
        self.last_seen = now_t
        self.pos_hist.append(tuple(self.pos.astype(int)))

tracks = []
HIST_LEN = 160
ASSOC_DIST = 100.0
PHASE_SMOOTH = 0.25

# ---- P1 錨在相位 0（按 START 那刻）----
ANCHOR_P1 = True
anchor_pos = None

# ---- 穩定綁定（Stable Binding）----
BIND_STABLE = True
BIND_WARMUP = 30
MISSING_TOL_FRAMES = 45
label_to_tid = {i: None for i in range(NUM_PLAYERS)}
tid_to_label = {}
bindings_initialized = False
frame_count = 0

# 高精度時基
start_t = time.perf_counter()

def get_cycle_ms():
    return SLOT_MS * NUM_PLAYERS

# <<< 函式大改：解決 KeyError 崩潰問題
def assign_to_track(detected_pos, now_t, current_player_idx):
    """更穩健的智慧關聯函式"""
    global tracks
    updated_tid = None

    if detected_pos is not None:
        px, py = detected_pos
        
        target_tid = label_to_tid.get(current_player_idx)
        
        if target_tid is not None:
            # 優先更新已綁定的軌跡
            tr = next((t for t in tracks if t.tid == target_tid), None)
            if tr:
                tr.update_pos((px, py), now_t)
                updated_tid = tr.tid
        else:
            # 當前玩家未綁定，從「未綁定」軌跡中找最近的
            unbound_tracks = [tr for tr in tracks if tr.tid not in tid_to_label]
            best_tr = None
            if unbound_tracks:
                best_tr = min(unbound_tracks, key=lambda tr: math.hypot(tr.pos[0] - px, tr.pos[1] - py))
                if math.hypot(best_tr.pos[0] - px, best_tr.pos[1] - py) > ASSOC_DIST:
                    best_tr = None
            
            if best_tr:
                best_tr.update_pos((px, py), now_t)
                updated_tid = best_tr.tid
            elif len(tracks) < NUM_PLAYERS:
                # 無合適的未綁定軌跡，且總數未滿，開一個新的
                new_tid = 0
                if tracks: # 避免列表為空
                    new_tid = max(tr.tid for tr in tracks) + 1
                tr = Track(new_tid, (px, py), now_t)
                tracks.append(tr)
                updated_tid = tr.tid

    # 根據本幀的更新情況，更新所有軌跡的 'visible_hist'
    for tr in tracks:
        tr.visible_hist.append(1 if tr.tid == updated_tid else 0)


def estimate_phase_labels(now_t):
    if not tracks: return
    dt_est = max(DT_CLAMP, 1/60)
    for tr in tracks:
        vis = np.array(tr.visible_hist, dtype=np.float32)
        if vis.sum() < 3: continue
        times = [now_t - i * dt_est for i, v in enumerate(reversed(vis)) if v > 0.5]
        if not times: continue
        
        phases = [(((ti - start_t) * 1000.0 % get_cycle_ms()) / SLOT_MS) for ti in times]
        angs = np.array(phases) * (2 * math.pi / NUM_PLAYERS)
        mean_vec = np.array([np.cos(angs).mean(), np.sin(angs).mean()])
        mean_ang = math.atan2(mean_vec[1], mean_vec[0])
        phase_float = (mean_ang * NUM_PLAYERS / (2 * math.pi)) % NUM_PLAYERS

        if tr.phase_est is None:
            tr.phase_est = phase_float
        else:
            # 處理相位環繞問題
            diff = phase_float - tr.phase_est
            if diff > NUM_PLAYERS / 2: diff -= NUM_PLAYERS
            if diff < -NUM_PLAYERS / 2: diff += NUM_PLAYERS
            tr.phase_est = (tr.phase_est + PHASE_SMOOTH * diff) % NUM_PLAYERS

def rel_phase(a, b, n=NUM_PLAYERS):
    diff = (a - b) % n
    return min(diff, n - diff)

def bind_initial_with_anchor():
    global bindings_initialized, label_to_tid, tid_to_label
    if not BIND_STABLE or not tracks or anchor_pos is None: return
    
    cands = [tr for tr in tracks if tr.phase_est is not None]
    if not cands: return

    anchor_tr = min(cands, key=lambda tr: math.hypot(tr.pos[0] - anchor_pos[0], tr.pos[1] - anchor_pos[1]))
    phi0 = anchor_tr.phase_est

    others = sorted([tr for tr in cands if tr.tid != anchor_tr.tid], 
                    key=lambda tr: rel_phase(tr.phase_est, phi0))

    label_to_tid = {i: None for i in range(NUM_PLAYERS)}
    tid_to_label.clear()
    
    label_to_tid[0] = anchor_tr.tid
    tid_to_label[anchor_tr.tid] = 0
    
    for i, tr in enumerate(others, start=1):
        if i >= NUM_PLAYERS: break
        label_to_tid[i] = tr.tid
        tid_to_label[tr.tid] = i

    bindings_initialized = True

def maintain_bindings(now_t):
    if not BIND_STABLE or not bindings_initialized: return

    for lbl, tid in list(label_to_tid.items()):
        if tid is None: continue
        tr = next((t for t in tracks if t.tid == tid), None)
        if tr is None or (now_t - tr.last_seen > MISSING_TOL_FRAMES * (1/30.0)):
            label_to_tid[lbl] = None
            tid_to_label.pop(tid, None)

    if label_to_tid[0] is None and anchor_pos is not None:
        cands = [tr for tr in tracks if tr.phase_est is not None and tr.tid not in tid_to_label]
        if cands:
            tr_best = min(cands, key=lambda tr: math.hypot(tr.pos[0] - anchor_pos[0], tr.pos[1] - anchor_pos[1]))
            label_to_tid[0] = tr_best.tid
            tid_to_label[tr_best.tid] = 0

    tr0 = next((t for t in tracks if t.tid == label_to_tid.get(0)), None)
    if not tr0 or tr0.phase_est is None: return

    phi0 = tr0.phase_est
    for lbl in range(1, NUM_PLAYERS):
        if label_to_tid[lbl] is not None: continue
        
        target_phase = (phi0 + lbl) % NUM_PLAYERS
        cands = [tr for tr in tracks if tr.phase_est is not None and tr.tid not in tid_to_label]
        if not cands: continue
        
        best_tr = min(cands, key=lambda tr: rel_phase(tr.phase_est, target_phase))
        label_to_tid[lbl] = best_tr.tid
        tid_to_label[best_tr.tid] = lbl

# ========================== 互動：P1 由滑鼠控制 ==========================
game_started = False
just_started = False
def mouse_move(event, x, y, flags, param):
    global game_started, just_started
    if not game_started:
        if event == cv2.EVENT_LBUTTONDOWN:
            game_started = True
            just_started = True
    else:
        if event == cv2.EVENT_MOUSEMOVE:
            positions[0] = (x, y)

# ========================== 工具 & 繪圖 ==========================
def clamp_vec(x, y, max_len):
    mag = math.hypot(x, y)
    if mag > max_len and mag > 0:
        s = max_len / mag
        return x*s, y*s
    return x, y

def draw_outer_frame(img):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w-1, h-1), BORDER_COLOR, BORDER_THICK)
    L = CORNER_LEN
    t = max(2, BORDER_THICK//2)
    cv2.line(img, (BORDER_THICK, BORDER_THICK), (BORDER_THICK+L, BORDER_THICK), BORDER_COLOR, t)
    cv2.line(img, (BORDER_THICK, BORDER_THICK), (BORDER_THICK, BORDER_THICK+L), BORDER_COLOR, t)
    cv2.line(img, (w-1-BORDER_THICK, BORDER_THICK), (w-1-BORDER_THICK-L, BORDER_THICK), BORDER_COLOR, t)
    cv2.line(img, (w-1-BORDER_THICK, BORDER_THICK), (w-1-BORDER_THICK, BORDER_THICK+L), BORDER_COLOR, t)
    cv2.line(img, (BORDER_THICK, h-1-BORDER_THICK), (BORDER_THICK+L, h-1-BORDER_THICK), BORDER_COLOR, t)
    cv2.line(img, (BORDER_THICK, h-1-BORDER_THICK), (BORDER_THICK, h-1-BORDER_THICK-L), BORDER_COLOR, t)
    cv2.line(img, (w-1-BORDER_THICK, h-1-BORDER_THICK), (w-1-BORDER_THICK-L, h-1-BORDER_THICK), BORDER_COLOR, t)
    cv2.line(img, (w-1-BORDER_THICK, h-1-BORDER_THICK), (w-1-BORDER_THICK, h-1-BORDER_THICK-L), BORDER_COLOR, t)

def draw_trajectories(img, tracks):
    for tr in tracks:
        lbl = tid_to_label.get(tr.tid)
        if lbl is not None and len(tr.pos_hist) > 1:
            points = np.array(tr.pos_hist, dtype=np.int32)
            cv2.polylines(img, [points], isClosed=False, color=TRAJECTORY_COLORS[lbl], thickness=2, lineType=cv2.LINE_AA)

def draw_point_label(img, x, y, text, strong=False):
    font_scale = 0.7 if strong else 0.55
    thickness = 2 if strong else 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 4
    bx, by = int(x + LABEL_MARGIN), int(y - th - pad)
    cv2.rectangle(img, (bx, by - pad), (bx + tw + 2*pad, by + th + pad), HUD_BG, -1)
    cv2.putText(img, text, (bx + pad, by + th - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, HUD_TEXT, thickness, cv2.LINE_AA)
    cv2.line(img, (int(x), int(y)), (bx, by + th//2), (170,170,170), 1, cv2.LINE_AA)

def draw_right_panel(img, tracks, slot_ms):
    h, w = img.shape[:2]
    x0 = w - RIGHT_PANEL_W
    cv2.line(img, (x0, 0), (x0, h), (80,80,80), 1)
    title = f"IR Sync | slot={slot_ms:.1f}ms (act:{ACTIVE_MS:.1f}+grd:{GUARD_MS:.1f})"
    cv2.putText(img, title, (x0 + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_TEXT, 1, cv2.LINE_AA)
    
    y_p_start = 50
    bar_h = 22
    bar_w = w - x0 - 24
    for i in range(NUM_PLAYERS):
        py = y_p_start + i * (bar_h + 4)
        cv2.rectangle(img, (x0+12, py), (w-12, py+bar_h), (50,50,50), -1)
        active_w = bar_w * (ACTIVE_MS / SLOT_MS)
        cv2.rectangle(img, (x0+12, py), (int(x0+12+active_w), py+bar_h), (80,80,80), -1)
        cv2.putText(img, f"P{i+1}", (x0+18, py+bar_h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_TEXT, 1)

    total_cycle_ms = slot_ms * NUM_PLAYERS
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    prog_cycle = (elapsed_ms % total_cycle_ms) / total_cycle_ms
    marker_x = x0 + 12 + bar_w * prog_cycle
    cv2.line(img, (int(marker_x), y_p_start-2), (int(marker_x), y_p_start+NUM_PLAYERS*(bar_h+4)+2), (0,255,0), 1)
    
    y = y_p_start + NUM_PLAYERS*(bar_h+4) + 20
    for i in range(NUM_PLAYERS):
        tid = label_to_tid.get(i)
        tr = next((t for t in tracks if t.tid == tid), None)
        phase_str = f"{tr.phase_est:.2f}" if tr and tr.phase_est is not None else "N/A"
        txt = f"P{i+1} -> Track{tid if tid is not None else '-'}: phase={phase_str}"
        col, th = ((230,230,230), 2) if i == 0 else ((200,200,200), 1)
        cv2.putText(img, txt, (x0 + 12, y + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, th, cv2.LINE_AA)

# ========================== 初始化 ==========================
cv2.namedWindow("IR Gun Simulation")
cv2.setMouseCallback("IR Gun Simulation", mouse_move)

prev_t = time.perf_counter()
last_switch_time = prev_t
current_player = 0

# ========================== 主迴圈 ==========================
while True:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t = time.perf_counter()
    dt = min(t - prev_t, DT_CLAMP)
    prev_t = t

    if not game_started:
        cv2.putText(frame, "Click LEFT MOUSE BUTTON to START", (60, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
    else:
        if just_started:
            anchor_pos = positions[0] if ANCHOR_P1 else None
            start_t = time.perf_counter()
            last_switch_time = start_t
            tracks.clear()
            label_to_tid = {i: None for i in range(NUM_PLAYERS)}
            tid_to_label.clear()
            bindings_initialized = False
            frame_count = 0
            just_started = False

        for i in range(1, NUM_PLAYERS):
            px, py = positions[i]
            vx, vy = velocities[i]
            if t >= next_turn_time[i]:
                ang = math.atan2(vy, vx) if vx or vy else random.uniform(0, 2*math.pi)
                ang += random.uniform(-math.pi/2, math.pi/2)
                target_dir[i] = (math.cos(ang), math.sin(ang))
                next_turn_time[i] = t + random.uniform(TURN_MIN, TURN_MAX)
            ax = target_dir[i][0]*ACCEL + random.uniform(-JITTER, JITTER)
            ay = target_dir[i][1]*ACCEL + random.uniform(-JITTER, JITTER)
            vx += ax * dt; vy += ay * dt
            decay = math.exp(-FRICTION * dt); vx *= decay; vy *= decay
            vx, vy = clamp_vec(vx, vy, MAX_SPEED)
            nx, ny = px + vx*dt, py + vy*dt
            if not (EDGE_PAD < nx < WIDTH - EDGE_PAD): vx *= -0.8
            if not (EDGE_PAD < ny < HEIGHT - EDGE_PAD): vy *= -0.8
            positions[i] = (np.clip(nx, EDGE_PAD, WIDTH-EDGE_PAD), np.clip(ny, EDGE_PAD, HEIGHT-EDGE_PAD))
            velocities[i] = (vx, vy)

        while t >= last_switch_time + SLOT_TIME:
            current_player = (current_player + 1) % NUM_PLAYERS
            last_switch_time += SLOT_TIME

        detected = None
        time_in_slot = t - last_switch_time
        if time_in_slot < ACTIVE_TIME:
            cx, cy = positions[current_player]
            cv2.circle(frame, (int(cx), int(cy)), DOT_RADIUS, FG_COLOR, -1)
            detected = (cx, cy)
        
        assign_to_track(detected, t, current_player)
        estimate_phase_labels(t)

        frame_count += 1
        if BIND_STABLE:
            if not bindings_initialized and frame_count >= BIND_WARMUP:
                bind_initial_with_anchor()
            if bindings_initialized:
                maintain_bindings(t)

        draw_outer_frame(frame)
        draw_trajectories(frame, tracks)
        
        for tr in tracks:
            lbl = tid_to_label.get(tr.tid)
            if lbl is not None:
                draw_point_label(frame, tr.pos[0], tr.pos[1], f"P{lbl+1}", strong=(lbl==0))
        
        draw_right_panel(frame, tracks, SLOT_MS)

    cv2.imshow("IR Gun Simulation", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break

cv2.destroyAllWindows()