import cv2
import numpy as np
import time
import random
import math

# 視窗名稱
WIN_NAME = "IR Gun Simulation"

# 畫面大小
width, height = 1200, 600

# 顏色 (BGR)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

# 初始位置
positions = [(200, 300), (400, 300), (600, 300), (400, 150)]

# ---- P2~P4 虛擬滑鼠狀態 ----
velocities = [(0.0, 0.0) for _ in range(4)]
next_turn_time = [0.0 for _ in range(4)]
target_dir = [(0.0, 0.0) for _ in range(4)]

# 可調參數
MAX_SPEED = 900.0      # 玩家虛擬滑鼠的最大速度（像素/秒），限制移動不會過快
ACCEL = 2800.0         # 加速度（像素/秒²），決定轉向或加速的反應速度
JITTER = 450.0         # 抖動加速度（像素/秒²），模擬手抖或微小修正，數值越大晃動越明顯
FRICTION = 3.5         # 摩擦係數（1/秒），數值越大減速越快，移動更鈍
TURN_MIN, TURN_MAX = 0.20, 0.80  # 隨機換方向的時間範圍（秒）
EDGE_PAD = 20          # 與邊界保持的安全距離（像素）
DT_CLAMP = 1 / 60      # 每次迴圈最大允許的時間步長（秒）

# 嚴格時槽（每槽毫秒）
slot_ms = 3.0          # 預設每位 3 ms（P1 -> P2 -> P3 -> P4）
paused = False
start_t = time.perf_counter()
current_player = 0     # 畫面顯示用（由時槽決定）

# 遊戲狀態
game_started = False

def cycle_ms():
    return slot_ms * 4.0

def active_player(now_t):
    """根據啟動時間與 slot_ms，算出此刻該亮哪一位"""
    if paused:
        return current_player
    elapsed_ms = (now_t - start_t) * 1000.0
    slot = int((elapsed_ms % cycle_ms()) // slot_ms)
    return slot  # 0..3 -> P1..P4

def mouse_move(event, x, y, flags, param):
    global game_started
    if not game_started:
        if event == cv2.EVENT_LBUTTONDOWN:
            game_started = True
    else:
        if event == cv2.EVENT_MOUSEMOVE:
            positions[0] = (x, y)  # P1 = 真滑鼠

def clamp_vec(x, y, m):
    mag = math.hypot(x, y)
    if mag > m and mag > 0:
        s = m / mag
        return x * s, y * s
    return x, y

# 初始化
now = time.perf_counter()
for i in range(1, 4):
    ang = random.uniform(0, 2 * math.pi)
    target_dir[i] = (math.cos(ang), math.sin(ang))
    next_turn_time[i] = now + random.uniform(TURN_MIN, TURN_MAX)

# 建立視窗與滑鼠監聽
cv2.namedWindow(WIN_NAME)
cv2.setMouseCallback(WIN_NAME, mouse_move)

# 預先配置 frame，之後每幀清空以避免反覆配置記憶體
frame = np.zeros((height, width, 3), dtype=np.uint8)

# FPS 限制，避免 CPU 滿轉
TARGET_FPS = 120.0
TARGET_DT = 1.0 / TARGET_FPS
last_tick = time.perf_counter()

prev_t = time.perf_counter()

try:
    while True:
        # 若視窗被關閉，退出
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        # FPS 控制
        now_tick = time.perf_counter()
        step = now_tick - last_tick
        if step < TARGET_DT:
            time.sleep(TARGET_DT - step)
            now_tick = time.perf_counter()
            step = now_tick - last_tick
        last_tick = now_tick

        # 清空舊畫面（重用緩衝）
        frame[:] = 0

        # 時間步進
        t = time.perf_counter()
        dt = min(t - prev_t, DT_CLAMP)
        prev_t = t

        # 計算目前時槽玩家
        current_player = active_player(t)

        if not game_started:
            cv2.putText(
                frame,
                "Click LEFT MOUSE BUTTON to START",
                (100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
        else:
            # 更新 P2~P4 的「虛擬滑鼠」移動（時槽無關，持續模擬人手移動）
            for i in range(1, 4):
                px, py = positions[i]
                vx, vy = velocities[i]

                if t >= next_turn_time[i]:
                    base = math.atan2(target_dir[i][1], target_dir[i][0]) if target_dir[i] != (0.0, 0.0) else random.uniform(0, 2 * math.pi)
                    delta = random.uniform(-math.pi / 3, math.pi / 3)
                    ang = base + delta
                    target_dir[i] = (math.cos(ang), math.sin(ang))
                    next_turn_time[i] = t + random.uniform(TURN_MIN, TURN_MAX)

                ax = target_dir[i][0] * ACCEL + random.uniform(-JITTER, JITTER)
                ay = target_dir[i][1] * ACCEL + random.uniform(-JITTER, JITTER)

                if px < EDGE_PAD and vx < 0:
                    ax += (+ACCEL * 1.8)
                if px > width - EDGE_PAD and vx > 0:
                    ax += (-ACCEL * 1.8)
                if py < EDGE_PAD and vy < 0:
                    ay += (+ACCEL * 1.8)
                if py > height - EDGE_PAD and vy > 0:
                    ay += (-ACCEL * 1.8)

                vx += ax * dt
                vy += ay * dt
                decay = math.exp(-FRICTION * dt)
                vx *= decay
                vy *= decay
                vx, vy = clamp_vec(vx, vy, MAX_SPEED)

                nx = px + vx * dt
                ny = py + vy * dt
                bounced = False

                if nx < EDGE_PAD:
                    nx = EDGE_PAD
                    vx = abs(vx)
                    bounced = True
                elif nx > width - EDGE_PAD:
                    nx = width - EDGE_PAD
                    vx = -abs(vx)
                    bounced = True

                if ny < EDGE_PAD:
                    ny = EDGE_PAD
                    vy = abs(vy)
                    bounced = True
                elif ny > height - EDGE_PAD:
                    ny = height - EDGE_PAD
                    vy = -abs(vy)
                    bounced = True

                if bounced:
                    ang = math.atan2(vy, vx)
                    target_dir[i] = (math.cos(ang), math.sin(ang))
                    next_turn_time[i] = min(next_turn_time[i], t + 0.12)

                positions[i] = (int(nx), int(ny))
                velocities[i] = (vx, vy)

            # 只畫出「當前時槽」的玩家
            idx = current_player
            cv2.circle(frame, positions[idx], 10, colors[idx], -1)
            cv2.putText(
                frame,
                f"Player {idx+1}",
                (positions[idx][0] - 50, positions[idx][1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                colors[idx],
                2
            )

        # HUD
        hz_per_cycle = 1000.0 / max(1e-6, cycle_ms())  # 每秒完成幾次 4P 輪巡
        cv2.putText(
            frame,
            f"slot={slot_ms:.0f} ms  cycle={cycle_ms():.0f} ms  per-cycle={hz_per_cycle:.1f} Hz  active=P{current_player+1}  {'PAUSED' if paused else ''}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

        cv2.imshow(WIN_NAME, frame)

        # 正確的 waitKey 用法（必須帶參數）
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('='), ord('+')):
            slot_ms = min(100.0, slot_ms + 1.0)
        elif key == ord('-'):
            slot_ms = max(1.0, slot_ms - 1.0)
        elif key == ord('0'):
            slot_ms = 3.0
        elif key == ord('p'):
            paused = not paused
            if not paused:
                # 解除暫停時重新對齊起點，避免跳槽
                start_t = time.perf_counter()
        elif key == ord('n') and paused:
            # 暫停時單步
            current_player = (current_player + 1) % 4

except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()