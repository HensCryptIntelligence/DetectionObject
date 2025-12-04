# facemesh_ear_headpose_alarm_tuned.py
# Versi tuned: HAPUS IRIS, GANTI DENGAN EAR (EYE ASPECT RATIO) UNTUK DETEKSI MENGANTUK
# Hybrid FaceMesh + Pose untuk deteksi head turn ekstrem via telinga
# Requirements: mediapipe>=0.10, opencv-python, pygame, numpy

import cv2, mediapipe as mp, numpy as np, math, time, pygame

CAM_INDEX = 0
FRAME_WIDTH = 640
DRAW_MESH = False
SMOOTH_YAW_ALPHA = 0.5  

# --- EAR Threshold ---
EAR_THRESHOLD = 0.25  # Jika EAR < 0.25 → mata tertutup
EAR_CONSEC_FRAMES = 10  # Jumlah frame berturut-turut mata tertutup sebelum trigger alarm

ALARM_FILE = "danger_alarm.wav"
ALARM_DELAY = 3.0     
COOLDOWN = 3.0
VOLUME = 0.9

ON_OFF_FRAMES_THRESHOLD = 5
YAW_HEAD_TURN_THRESH = 18.0   
ALARM_MODE = "either"  # or "both"

# ====================================

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark mata untuk EAR (Eye Aspect Ratio)
LEFT_EYE = [362,263,387,373,380,385]  # left eye landmarks
RIGHT_EYE = [33,133,159,145,153,144]  # right eye landmarks
PNP_INDICES = [1,152,33,263,61,291]

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

pygame.mixer.init()
pygame.mixer.music.load(ALARM_FILE)
pygame.mixer.music.set_volume(VOLUME)

cap = cv2.VideoCapture(CAM_INDEX)

# states
ema_yaw = None
prev_time = None
offcenter_frames = 0
offcenter_start = None
last_alarm_time = -9999
fps_smoothed = 30.0
fps_alpha = 0.08

# EAR states
eye_closed_frames = 0
prev_ear = None

def ema(prev, value, alpha):
    return value if prev is None else prev*(1-alpha) + value*alpha

def calc_ear(eye_landmarks, w, h):
    """Hitung Eye Aspect Ratio (EAR) dari 6 titik mata"""
    # Ambil koordinat titik-titik mata
    x = [lm.x * w for lm in eye_landmarks]
    y = [lm.y * h for lm in eye_landmarks]
    
    # Hitung jarak vertikal (A dan B)
    A = math.dist((x[1], y[1]), (x[5], y[5]))  # atas-bawah
    B = math.dist((x[2], y[2]), (x[4], y[4]))  # atas-bawah
    
    # Hitung jarak horizontal (C)
    C = math.dist((x[0], y[0]), (x[3], y[3]))  # kiri-kanan
    
    if C == 0:
        return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear

def solve_head_pose(lm_list,w,h):
    model_points = np.array([(0.0,0.0,0.0),(0.0,-63.6,-12.5),
                             (-43.3,32.7,-26.0),(43.3,32.7,-26.0),
                             (-28.9,-28.9,-24.1),(28.9,-28.9,-24.1)], dtype=np.float64)
    img_pts=[]
    for idx in PNP_INDICES:
        lm = lm_list[idx]; img_pts.append((lm.x*w, lm.y*h))
    image_points = np.array(img_pts, dtype=np.float64)
    focal = w; center=(w/2,h/2)
    camera_matrix = np.array([[focal,0,center[0]],[0,focal,center[1]],[0,0,1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))
    ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    R,_ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        pitch = math.degrees(math.atan2(-R[1,2], R[1,1]))
        yaw = math.degrees(math.atan2(-R[2,0], sy))
        roll = 0.0
    return yaw, pitch, roll

def detect_head_turn_from_pose(pose_landmarks, w, h):
    """Estimasi kasar head turn dari posisi hidung dan telinga"""
    nose = pose_landmarks.landmark[0]
    ear_left = pose_landmarks.landmark[8]   # left ear
    ear_right = pose_landmarks.landmark[7]  # right ear

    nose_x = nose.x * w
    ear_left_x = ear_left.x * w
    ear_right_x = ear_right.x * w

    ear_mid_x = (ear_left_x + ear_right_x) / 2.0
    offset = nose_x - ear_mid_x
    ear_dist = abs(ear_right_x - ear_left_x)

    if ear_dist < 10:
        return False, 0.0

    normalized_offset = offset / ear_dist

    if abs(normalized_offset) > 0.3:
        return True, normalized_offset
    return False, normalized_offset

try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break

        # FPS smoothing
        if prev_time is None: prev_time = t0
        dt = t0 - prev_time if prev_time else 1/30.0
        prev_time = t0
        inst_fps = 1.0/dt if dt>0 else 30.0
        fps_smoothed = ema(fps_smoothed, inst_fps, fps_alpha)

        # Resize & flip
        h0,w0 = frame.shape[:2]
        scale = FRAME_WIDTH/float(w0)
        frame = cv2.resize(frame, (FRAME_WIDTH, int(h0*scale)))
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses FaceMesh
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)

        face_present = False
        center_state = True
        head_yaw = head_pitch = head_roll = None
        head_turn = False
        eye_open = True  # default: mata terbuka
        ear_avg = None

        # --- Mode 1: FaceMesh aktif ---
        if face_results.multi_face_landmarks:
            face_present = True
            lm_list = face_results.multi_face_landmarks[0].landmark

            if DRAW_MESH:
                mp_drawing.draw_landmarks(
                    image=frame, landmark_list=face_results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # EAR: hitung untuk kiri & kanan
            left_eye_lms = [lm_list[i] for i in LEFT_EYE]
            right_eye_lms = [lm_list[i] for i in RIGHT_EYE]

            ear_left = calc_ear(left_eye_lms, w, h)
            ear_right = calc_ear(right_eye_lms, w, h)
            ear_avg = (ear_left + ear_right) / 2.0

            # Tentukan apakah mata tertutup
            if ear_avg < EAR_THRESHOLD:
                eye_open = False
                eye_closed_frames += 1
            else:
                eye_open = True
                eye_closed_frames = 0

            # Head pose
            hp = solve_head_pose(lm_list, w, h)
            if hp is not None:
                yaw, pitch, roll = hp
                ema_yaw = ema(ema_yaw, yaw, SMOOTH_YAW_ALPHA) if ema_yaw is not None else yaw
                head_yaw = ema_yaw; head_pitch = pitch; head_roll = roll

            if head_yaw is not None and abs(head_yaw) > YAW_HEAD_TURN_THRESH:
                head_turn = True

        # --- Mode 2: FaceMesh TIDAK aktif, tapi Pose aktif ---
        elif pose_results.pose_landmarks:
            face_present = True
            head_turn_fallback, offset = detect_head_turn_from_pose(pose_results.pose_landmarks, w, h)
            head_turn = head_turn_fallback
            eye_open = True  # anggap mata terbuka jika wajah tidak terlihat

        # --- Tentukan offcenter flag ---
        # Jika mata tertutup TERLALU LAMA atau kepala menoleh → offcenter
        eye_closed_long = eye_closed_frames >= EAR_CONSEC_FRAMES
        if ALARM_MODE == "either":
            offcenter_flag = eye_closed_long or head_turn
        else:  # "both"
            offcenter_flag = eye_closed_long and head_turn

        center_state = not offcenter_flag

        # --- Overlay visual ---
        if face_results.multi_face_landmarks or pose_results.pose_landmarks:
            if ear_avg is not None:
                cv2.putText(frame, f"EAR:{ear_avg:.3f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                cv2.putText(frame, "EAR: N/A", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if head_yaw is not None:
                cv2.putText(frame, f"Yaw:{head_yaw:.1f} Pitch:{head_pitch:.1f} Roll:{head_roll:.1f}",
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            status = "HEAD TURN (Pose)" if (not face_results.multi_face_landmarks and head_turn) else \
                     ("HEAD TURN" if head_turn else ("EYES CLOSED" if not eye_open else "EYES OPEN"))
            cv2.putText(frame, f"Status:{status}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        # --- Alarm logic ---
        now = time.time()
        if face_present and not center_state:
            offcenter_frames += 1
        else:
            offcenter_frames = 0
            offcenter_start = None

        frames_needed = 15
        if offcenter_frames >= ON_OFF_FRAMES_THRESHOLD and offcenter_start is None:
            offcenter_start = now

        elapsed = (now - offcenter_start) if offcenter_start else 0.0
        alarm_playing = pygame.mixer.music.get_busy()
        time_since_last = now - last_alarm_time

        if (offcenter_frames >= frames_needed and elapsed >= ALARM_DELAY and
            not alarm_playing and (time_since_last > COOLDOWN)):
            pygame.mixer.music.play()
            last_alarm_time = now
            print("Alarm triggered at", time.strftime("%H:%M:%S"),
                  f"(frames={offcenter_frames}, frames_needed={frames_needed}, elapsed={elapsed:.2f}s)")

        # --- FPS display ---
        cur = time.time()
        inst_fps = 1.0 / (cur - prev_time) if prev_time else fps_smoothed
        prev_time = cur
        cv2.putText(frame, f"FPS:{fps_smoothed:.1f}", (w-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Tuned FaceMesh+EAR+HeadPose+Alarm", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): break
        if key == ord('m'): DRAW_MESH = not DRAW_MESH
        if key == ord('s'): 
            if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pose.close()
    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except: pass