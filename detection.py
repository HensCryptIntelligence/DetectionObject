# facemesh_gaze_headpose_alarm_tuned.py
# Versi tuned: smoothing yaw benar, frame-based trigger konsisten (berdasarkan FPS),
# hapus immediate short-frame trigger, outlier rejection sederhana.
# Requirements: mediapipe opencv-python pygame numpy

import cv2, mediapipe as mp, numpy as np, math, time, pygame

CAM_INDEX = 0
FRAME_WIDTH = 640
DRAW_MESH = False
SMOOTH_ALPHA = 0.5
SMOOTH_YAW_ALPHA = 0.5  

MAX_GAZE_DELTA = 0.15 

LEFT_THRESH = 0.35
RIGHT_THRESH = 0.65
UP_THRESH = 0.35
DOWN_THRESH = 0.65


ALARM_FILE = "danger_alarm.wav"
ALARM_DELAY = 1.0     
COOLDOWN = 1.0
VOLUME = 0.9

ON_OFF_FRAMES_THRESHOLD = 5


YAW_HEAD_TURN_THRESH = 18.0   
ALARM_MODE = "either" # or "both"

# ====================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LEFT_IRIS = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]
LEFT_EYE = [362,263,387,373,380,385]
RIGHT_EYE = [33,133,159,145,153,144]
PNP_INDICES = [1,152,33,263,61,291]

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)

pygame.mixer.init()
pygame.mixer.music.load(ALARM_FILE)
pygame.mixer.music.set_volume(VOLUME)

cap = cv2.VideoCapture(CAM_INDEX)

# states
ema_l = ema_r = ema_v_l = ema_v_r = None
ema_yaw = None
prev_gaze = None
prev_time = None
offcenter_frames = 0
offcenter_start = None
last_alarm_time = -9999
fps_smoothed = 30.0
fps_alpha = 0.08

def ema(prev, value, alpha):
    return value if prev is None else prev*(1-alpha) + value*alpha

def centroid_px(landmarks,w,h):
    xs=[p.x for p in landmarks]; ys=[p.y for p in landmarks]
    return int(sum(xs)/len(xs)*w), int(sum(ys)/len(ys)*h)

def calc_relative(iris_x, eye_xs):
    mn=min(eye_xs); mx=max(eye_xs); width=mx-mn
    if width<=1e-3: return 0.5
    return (iris_x - mn)/width

def calc_relative_y(iris_y, eye_ys):
    mn=min(eye_ys); mx=max(eye_ys); height=mx-mn
    if height<=1e-3: return 0.5
    return (iris_y - mn)/height

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
        # processing
        h0,w0 = frame.shape[:2]; scale = FRAME_WIDTH/float(w0)
        frame = cv2.resize(frame,(FRAME_WIDTH,int(h0*scale))); h,w = frame.shape[:2]
        frame = cv2.flip(frame,1); rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_present=False; center_state=True; gaze_x=gaze_y=None; head_yaw=head_pitch=head_roll=None

        if results.multi_face_landmarks:
            face_present=True
            lm_list = results.multi_face_landmarks[0].landmark
            if DRAW_MESH:
                mp_drawing.draw_landmarks(image=frame, landmark_list=results.multi_face_landmarks[0],
                                          connections=mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            # iris centers
            lx,ly = centroid_px([lm_list[i] for i in LEFT_IRIS], w, h)
            rx,ry = centroid_px([lm_list[i] for i in RIGHT_IRIS], w, h)
            # eye boxes
            left_eye_xs = [lm_list[i].x*w for i in LEFT_EYE]; left_eye_ys = [lm_list[i].y*h for i in LEFT_EYE]
            right_eye_xs = [lm_list[i].x*w for i in RIGHT_EYE]; right_eye_ys = [lm_list[i].y*h for i in RIGHT_EYE]
            # draw
            cv2.circle(frame,(lx,ly),2,(0,255,0),-1); cv2.circle(frame,(rx,ry),2,(0,255,0),-1)

            # relative gaze
            l_rel_x=calc_relative(lx,left_eye_xs); r_rel_x=calc_relative(rx,right_eye_xs)
            l_rel_y=calc_relative_y(ly,left_eye_ys); r_rel_y=calc_relative_y(ry,right_eye_ys)
            cand_x = (l_rel_x + r_rel_x)/2.0; cand_y = (l_rel_y + r_rel_y)/2.0

            # outlier rejection: jika perubahan besar dibanding prev_gaze, ignore update once
            reject = False
            if prev_gaze is not None:
                if abs(cand_x - prev_gaze[0]) > MAX_GAZE_DELTA or abs(cand_y - prev_gaze[1]) > MAX_GAZE_DELTA:
                    reject = True

            if not reject:
                ema_l = ema(ema_l, l_rel_x, SMOOTH_ALPHA); ema_r = ema(ema_r, r_rel_x, SMOOTH_ALPHA)
                ema_v_l = ema(ema_v_l, l_rel_y, SMOOTH_ALPHA); ema_v_r = ema(ema_v_r, r_rel_y, SMOOTH_ALPHA)
                gaze_x = (ema_l + ema_r)/2.0; gaze_y = (ema_v_l + ema_v_r)/2.0
                prev_gaze = (gaze_x, gaze_y)
            else:
                gaze_x, gaze_y = prev_gaze if prev_gaze is not None else (0.5,0.5)

            # head pose
            hp = solve_head_pose(lm_list, w, h)
            if hp is not None:
                yaw,pitch,roll = hp
                ema_yaw = ema(ema_yaw, yaw, SMOOTH_YAW_ALPHA) if 'ema_yaw' in globals() and ema_yaw is not None else yaw
                # ensure global ema_yaw exists
                globals()['ema_yaw'] = ema_yaw
                head_yaw = ema_yaw; head_pitch = pitch; head_roll = roll

            # gaze-> status
            horiz = vert = "CENTER"
            if gaze_x is not None:
                if gaze_x < LEFT_THRESH: horiz = "LEFT"
                elif gaze_x > RIGHT_THRESH: horiz = "RIGHT"
                if gaze_y < UP_THRESH: vert = "UP"
                elif gaze_y > DOWN_THRESH: vert = "DOWN"

            # head-turn boolean using smoothed head_yaw
            head_turn = False
            if head_yaw is not None and abs(head_yaw) > YAW_HEAD_TURN_THRESH:
                head_turn = True

            # decide offcenter flag based on ALARM_MODE
            gaze_center = (horiz=="CENTER" or vert=="CENTER")
            if ALARM_MODE == "either":
                offcenter_flag = (not gaze_center) or head_turn
            else: # "both"
                offcenter_flag = (not gaze_center) and head_turn

            center_state = not offcenter_flag

            # overlays
            cv2.putText(frame, f"GazeX:{gaze_x:.2f} GazeY:{gaze_y:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            if head_yaw is not None:
                cv2.putText(frame, f"Yaw:{head_yaw:.1f} Pitch:{head_pitch:.1f} Roll:{head_roll:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            status = ("HEAD TURN" if head_turn else f"{horiz}/{vert}")
            cv2.putText(frame, f"Status:{status}", (10,h-20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)

        # ---- Frame-based trigger using FPS-derived frames_needed ----
        now = time.time()
        # update offcenter_frames
        if face_present and not center_state:
            offcenter_frames += 1
        else:
            offcenter_frames = 0
            offcenter_start = None

        # compute frames needed from ALARM_DELAY and fps_smoothed
        frames_needed = 15

        # only start timer after ON_OFF_FRAMES_THRESHOLD consecutive off frames (debounce)
        if offcenter_frames >= ON_OFF_FRAMES_THRESHOLD and offcenter_start is None:
            offcenter_start = now

        elapsed = (now - offcenter_start) if offcenter_start else 0.0
        alarm_playing = pygame.mixer.music.get_busy()
        time_since_last = now - last_alarm_time

        # trigger only if offcenter_frames >= frames_needed AND elapsed >= ALARM_DELAY
        if offcenter_frames >= frames_needed and elapsed >= ALARM_DELAY and not alarm_playing and (time_since_last > COOLDOWN):
            pygame.mixer.music.play()
            last_alarm_time = now
            print("Alarm triggered at", time.strftime("%H:%M:%S"), f"(frames={offcenter_frames}, frames_needed={frames_needed}, elapsed={elapsed:.2f}s)")

        # display fps
        cur = time.time()
        inst_fps = 1.0 / (cur - prev_time) if prev_time else fps_smoothed
        prev_time = cur
        cv2.putText(frame, f"FPS:{fps_smoothed:.1f}", (w-120,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

        cv2.imshow("Tuned FaceMesh+Gaze+HeadPose+Alarm", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): break
        if key == ord('m'): DRAW_MESH = not DRAW_MESH
        if key == ord('s'): 
            if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()

finally:
    cap.release(); cv2.destroyAllWindows(); face_mesh.close()
    try: pygame.mixer.music.stop(); pygame.mixer.quit()
    except: pass
