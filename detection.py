# python 3.10.9+
# ==================== BAGIAN IMPORT LIBRARY ====================
import cv2, mediapipe as mp, numpy as np, math, time, pygame, sys, os
# cv2         : Library OpenCV untuk pemrosesan gambar dan video dari kamera
# mediapipe   : Library dari Google untuk deteksi wajah, pose tubuh, dan landmark
# numpy       : Library untuk operasi matematika dan array multidimensi
# math        : Library untuk fungsi matematika dasar (sqrt, atan2, dll)
# time        : Library untuk mengelola waktu dan delay
# pygame      : Library untuk memutar suara alarm
# sys         : Library untuk interaksi dengan sistem Python
# os          : Library untuk operasi sistem file (mengecek file ada atau tidak)


# ==================== BAGIAN KONFIGURASI KAMERA ====================
CAM_INDEX = 0           # Indeks kamera yang digunakan (0 = kamera utama/default)
FRAME_WIDTH = 640       # Lebar frame video dalam piksel (untuk menghemat pemrosesan)
DRAW_MESH = False       # Apakah menampilkan mesh wajah di layar (True/False)


# ==================== BAGIAN KONFIGURASI DETEKSI ====================
EAR_THRESHOLD = 0.25         # Ambang batas Eye Aspect Ratio (EAR) - di bawah nilai ini mata dianggap tertutup
DROWSY_TIME_SECS = 0.5       # Durasi mata tertutup (dalam detik) sebelum dianggap mengantuk
MIN_OFFCENTER_SECONDS = 1.0  # Durasi minimum kondisi tidak fokus sebelum alarm berbunyi

SMOOTH_YAW_ALPHA = 0.80      # Koefisien smoothing untuk menghaluskan pergerakan sudut yaw kepala
                             # Nilai mendekati 1 = lebih responsif tapi lebih goyang
                             # Nilai mendekati 0 = lebih halus tapi lebih lambat merespons

YAW_HEAD_TURN_THRESH = 30.0  # Ambang batas sudut yaw (derajat) - di atas nilai ini kepala dianggap menoleh
COOLDOWN = 1.0               # Jeda waktu (detik) antar alarm agar tidak bunyi terus-menerus
VOLUME = 0.9                 # Volume suara alarm (0.0 sampai 1.0)

ALARM_MODE = "either"        # Mode pemicu alarm:
                             # "either" = alarm bunyi jika mata mengantuk ATAU kepala menoleh
                             # "both"   = alarm bunyi jika mata mengantuk DAN kepala menoleh

ALARM_FILE = "danger_alarm.wav"  # Nama file suara alarm yang akan diputar


# ==================== BAGIAN INISIALISASI MEDIAPIPE ====================
mp_face_mesh = mp.solutions.face_mesh    # Modul untuk mendeteksi 468 titik landmark wajah
mp_pose = mp.solutions.pose              # Modul untuk mendeteksi pose tubuh (33 titik)
mp_drawing = mp.solutions.drawing_utils  # Utilitas untuk menggambar landmark di layar
mp_drawing_styles = mp.solutions.drawing_styles  # Gaya tampilan bawaan untuk menggambar mesh


# ==================== BAGIAN INDEKS LANDMARK MATA ====================
# Indeks titik landmark yang membentuk kontur mata kiri dan kanan
# Urutan titik: sudut luar, atas-luar, atas-dalam, sudut dalam, bawah-dalam, bawah-luar
LEFT_EYE  = [33,160,158,133,153,144]   # 6 titik yang membentuk mata kiri
RIGHT_EYE = [362,387,385,263,373,380]  # 6 titik yang membentuk mata kanan

# Indeks landmark untuk perhitungan pose kepala menggunakan algoritma PnP
# Urutan: hidung, dagu, sudut mata kiri, sudut mata kanan, sudut mulut kiri, sudut mulut kanan
PNP_INDICES = [1,152,33,263,61,291]


# ==================== BAGIAN PEMBUATAN OBJEK DETEKTOR ====================
# Membuat objek Face Mesh untuk mendeteksi landmark wajah
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,              # Maksimal 1 wajah yang dideteksi (untuk efisiensi)
    refine_landmarks=True,        # Aktifkan deteksi landmark mata dan bibir yang lebih detail
    min_detection_confidence=0.5, # Kepercayaan minimum untuk deteksi awal (50%)
    min_tracking_confidence=0.5   # Kepercayaan minimum untuk pelacakan frame ke frame (50%)
)

# Membuat objek Pose untuk mendeteksi pose tubuh
pose = mp_pose.Pose()  # Menggunakan pengaturan default


# ==================== BAGIAN INISIALISASI AUDIO ====================
pygame.mixer.init()  # Menginisialisasi sistem audio pygame

# Mengecek apakah file alarm ada, jika ada maka dimuat
if os.path.exists(ALARM_FILE):
    pygame.mixer.music.load(ALARM_FILE)      # Memuat file audio ke memori
    pygame.mixer.music.set_volume(VOLUME)    # Mengatur volume sesuai konfigurasi


# ==================== BAGIAN INISIALISASI KAMERA ====================
cap = cv2.VideoCapture(CAM_INDEX)  # Membuka koneksi ke kamera dengan indeks yang ditentukan


# ==================== BAGIAN VARIABEL STATE/KONDISI ====================
ema_yaw = None              # Nilai yaw yang sudah dihaluskan (Exponential Moving Average)
                            # None = belum ada nilai sebelumnya

last_alarm_time = -9999     # Waktu terakhir alarm berbunyi (diset sangat lampau agar bisa bunyi di awal)
fps_smoothed = 30.0         # Nilai FPS yang sudah dihaluskan (dimulai dari 30)
fps_alpha = 0.08            # Koefisien smoothing untuk perhitungan FPS

eye_close_start = None      # Waktu mulai mata tertutup (None = mata terbuka)
offcenter_start = None      # Waktu mulai kondisi tidak fokus (None = sedang fokus)
prev_frame_time = time.time()  # Waktu frame sebelumnya untuk menghitung FPS


# ==================== BAGIAN FUNGSI-FUNGSI PEMBANTU ====================

def ema(prev, value, alpha):
    """
    Fungsi Exponential Moving Average (EMA) untuk menghaluskan nilai.
    
    Parameter:
    - prev  : Nilai EMA sebelumnya (bisa None jika belum ada)
    - value : Nilai baru yang akan diproses
    - alpha : Koefisien smoothing (0-1), semakin besar semakin responsif
    
    Return:
    - Nilai EMA yang sudah dihitung
    
    Rumus: EMA_baru = EMA_lama × (1 - alpha) + nilai_baru × alpha
    """
    return value if prev is None else prev*(1-alpha) + value*alpha


def calc_ear(eye_landmarks, w, h):
    """
    Menghitung Eye Aspect Ratio (EAR) untuk mendeteksi apakah mata terbuka atau tertutup.
    
    Parameter:
    - eye_landmarks : List 6 titik landmark mata
    - w, h          : Lebar dan tinggi frame dalam piksel
    
    Return:
    - Nilai EAR (float) - semakin kecil nilainya, semakin tertutup matanya
    
    Cara kerja:
    - EAR = (jarak vertikal A + jarak vertikal B) / (2 × jarak horizontal C)
    - Mata terbuka: EAR sekitar 0.3-0.4
    - Mata tertutup: EAR sekitar 0.1-0.2
    """
    # Mengkonversi koordinat relatif (0-1) ke koordinat piksel
    x = [lm.x * w for lm in eye_landmarks]  # Koordinat x setiap titik
    y = [lm.y * h for lm in eye_landmarks]  # Koordinat y setiap titik
    
    # Menghitung jarak vertikal antara kelopak atas dan bawah
    A = math.dist((x[1], y[1]), (x[5], y[5]))  # Jarak titik 1 ke titik 5 (vertikal luar)
    B = math.dist((x[2], y[2]), (x[4], y[4]))  # Jarak titik 2 ke titik 4 (vertikal dalam)
    
    # Menghitung jarak horizontal mata (sudut ke sudut)
    C = math.dist((x[0], y[0]), (x[3], y[3]))  # Jarak titik 0 ke titik 3 (horizontal)
    
    # Menghitung EAR dengan pembagian aman (menghindari pembagian dengan nol)
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def solve_head_pose(lm_list, w, h):
    """
    Menghitung orientasi kepala (yaw, pitch, roll) menggunakan algoritma PnP.
    
    Parameter:
    - lm_list : Daftar semua landmark wajah
    - w, h    : Lebar dan tinggi frame dalam piksel
    
    Return:
    - Tuple (yaw, pitch, roll) dalam derajat, atau None jika gagal
    
    Penjelasan sudut:
    - Yaw   : Rotasi kepala ke kiri/kanan (seperti menggelengkan kepala)
    - Pitch : Rotasi kepala ke atas/bawah (seperti mengangguk)
    - Roll  : Rotasi kepala miring ke samping (seperti memiringkan kepala ke bahu)
    """
    
    # Model 3D wajah generik dalam koordinat milimeter
    # Titik-titik ini mewakili posisi standar fitur wajah dalam ruang 3D
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Ujung hidung (titik referensi pusat)
        (0.0, -63.6, -12.5),      # Dagu
        (-43.3, 32.7, -26.0),     # Sudut mata kiri luar
        (43.3, 32.7, -26.0),      # Sudut mata kanan luar
        (-28.9, -28.9, -24.1),    # Sudut mulut kiri
        (28.9, -28.9, -24.1)      # Sudut mulut kanan
    ], dtype=np.float64)

    # Mengambil koordinat 2D dari landmark yang terdeteksi di gambar
    # Mengkonversi dari koordinat relatif (0-1) ke koordinat piksel
    img_pts = [(lm_list[idx].x*w, lm_list[idx].y*h) for idx in PNP_INDICES]
    image_points = np.array(img_pts, dtype=np.float64)

    # Membuat matriks kamera (camera matrix) untuk proyeksi 3D ke 2D
    # Asumsi: focal length = lebar frame, pusat gambar di tengah
    focal = w  # Focal length diasumsikan sama dengan lebar frame
    camera_matrix = np.array([
        [focal, 0, w/2],   # Baris 1: focal_x, skew, center_x
        [0, focal, h/2],   # Baris 2: skew, focal_y, center_y
        [0, 0, 1]          # Baris 3: untuk homogeneous coordinates
    ], dtype=np.float64)

    # Menyelesaikan masalah Perspective-n-Point (PnP)
    # Mencari rotasi dan translasi yang memetakan titik 3D ke titik 2D
    ok, rvec, tvec = cv2.solvePnP(
        model_points,           # Titik-titik 3D model
        image_points,           # Titik-titik 2D di gambar
        camera_matrix,          # Matriks kamera
        np.zeros((4,1)),        # Koefisien distorsi (diasumsikan tidak ada)
        flags=cv2.SOLVEPNP_ITERATIVE  # Metode iteratif untuk akurasi lebih baik
    )
    
    # Jika gagal menyelesaikan PnP, kembalikan None
    if not ok:
        return None

    # Mengkonversi rotation vector (rvec) ke rotation matrix (R)
    # Rotation matrix adalah matriks 3x3 yang merepresentasikan rotasi
    R, _ = cv2.Rodrigues(rvec)
    
    # Menghitung nilai sy untuk mengecek singularitas gimbal lock
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)

    # Mengekstrak sudut Euler (yaw, pitch, roll) dari rotation matrix
    if sy > 1e-6:  # Tidak ada gimbal lock
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))   # Rotasi sumbu X
        yaw   = math.degrees(math.atan2(-R[2,0], sy))      # Rotasi sumbu Y
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))   # Rotasi sumbu Z
    else:  # Ada gimbal lock, gunakan perhitungan alternatif
        pitch = math.degrees(math.atan2(-R[1,2], R[1,1]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = 0  # Roll tidak bisa dihitung saat gimbal lock

    return yaw, pitch, roll


def detect_head_turn_from_pose(pose_landmarks, w, h):
    """
    Mendeteksi apakah kepala menoleh berdasarkan pose tubuh.
    Digunakan sebagai fallback ketika wajah tidak terdeteksi dengan baik.
    
    Parameter:
    - pose_landmarks : Landmark pose tubuh dari MediaPipe
    - w, h           : Lebar dan tinggi frame dalam piksel
    
    Return:
    - Tuple (is_turning, normalized_offset)
      - is_turning       : Boolean, True jika kepala menoleh
      - normalized_offset: Float, seberapa jauh kepala dari tengah (-1 sampai 1)
    """
    # Mengambil posisi hidung dan kedua telinga dari landmark pose
    nose = pose_landmarks.landmark[0]       # Hidung
    ear_left = pose_landmarks.landmark[8]   # Telinga kiri
    ear_right = pose_landmarks.landmark[7]  # Telinga kanan

    # Mengkonversi koordinat relatif ke piksel
    nx = nose.x * w     # Posisi x hidung
    lx = ear_left.x * w   # Posisi x telinga kiri
    rx = ear_right.x * w  # Posisi x telinga kanan

    # Menghitung jarak antara kedua telinga
    ear_dist = abs(rx - lx)
    
    # Jika jarak telinga terlalu kecil (wajah terlalu miring atau tidak terdeteksi dengan baik)
    # Kembalikan False untuk menghindari pembagian dengan angka kecil
    if ear_dist < w * 0.05:  # Kurang dari 5% lebar frame
        return False, 0.0

    # Menghitung titik tengah antara kedua telinga
    mid = (lx + rx) / 2
    
    # Menghitung seberapa jauh hidung dari titik tengah, dinormalisasi
    # Nilai positif = menoleh ke kanan, negatif = menoleh ke kiri
    normalized = (nx - mid) / ear_dist
    
    # Kepala dianggap menoleh jika offset lebih dari 30%
    return abs(normalized) > 0.3, normalized


# ==================== BAGIAN LOOP UTAMA PROGRAM ====================
try:
    while True:  # Loop tak terbatas sampai pengguna menekan tombol keluar
        
        # ---------- MEMBACA FRAME DARI KAMERA ----------
        ret, frame = cap.read()  # ret = status berhasil/tidak, frame = gambar
        if not ret:  # Jika gagal membaca frame
            break    # Keluar dari loop
        
        # ---------- MENGHITUNG FPS (Frames Per Second) ----------
        now_t = time.time()           # Waktu saat ini
        dt = now_t - prev_frame_time  # Selisih waktu dengan frame sebelumnya (delta time)
        prev_frame_time = now_t       # Simpan waktu saat ini untuk frame berikutnya
        
        # Menghitung FPS yang dihaluskan menggunakan EMA
        # FPS = 1 / delta_time (berapa frame per detik)
        fps_smoothed = ema(fps_smoothed, 1/dt if dt > 0 else 30, fps_alpha)

        # ---------- PREPROCESSING FRAME ----------
        h0, w0 = frame.shape[:2]  # Mendapatkan tinggi dan lebar asli frame
        scale = FRAME_WIDTH / w0  # Menghitung faktor skala
        
        # Mengubah ukuran frame untuk mempercepat pemrosesan
        frame = cv2.resize(frame, (FRAME_WIDTH, int(h0 * scale)))
        
        # Membalik frame secara horizontal (mirror) agar seperti cermin
        frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]  # Mendapatkan dimensi frame yang sudah diubah ukurannya

        # ---------- KONVERSI WARNA DAN DETEKSI ----------
        # MediaPipe membutuhkan format RGB, sedangkan OpenCV menggunakan BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Menjalankan deteksi wajah (face mesh) pada frame
        face_results = face_mesh.process(rgb)
        
        # Menjalankan deteksi pose tubuh pada frame
        pose_results = pose.process(rgb)

        # ---------- INISIALISASI VARIABEL DETEKSI ----------
        face_present = False      # Apakah wajah terdeteksi
        head_turn = False         # Apakah kepala menoleh
        ear_avg = None            # Nilai rata-rata EAR kedua mata
        eye_closed_long = False   # Apakah mata tertutup cukup lama
        head_yaw = None           # Nilai sudut yaw kepala

        # ---------- PEMROSESAN JIKA WAJAH TERDETEKSI ----------
        if face_results.multi_face_landmarks:
            face_present = True  # Tandai bahwa wajah terdeteksi
            
            # Mengambil landmark wajah pertama (indeks 0)
            lm_list = face_results.multi_face_landmarks[0].landmark
            
            # Jika DRAW_MESH aktif, gambar mesh wajah di layar
            if DRAW_MESH:
                # Menggambar tessellation (jaring segitiga) wajah
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Menggambar kontur wajah (mata, alis, bibir, dll)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            # Mengambil landmark untuk mata kiri dan kanan
            left_eye = [lm_list[i] for i in LEFT_EYE]    # 6 titik mata kiri
            right_eye = [lm_list[i] for i in RIGHT_EYE]  # 6 titik mata kanan
            
            # Menghitung rata-rata EAR dari kedua mata
            ear_avg = (calc_ear(left_eye, w, h) + calc_ear(right_eye, w, h)) / 2.0

            # ---------- LOGIKA DETEKSI MATA TERTUTUP ----------
            now = time.time()  # Waktu saat ini
            
            if ear_avg < EAR_THRESHOLD:  # Jika EAR di bawah ambang batas (mata tertutup)
                if eye_close_start is None:  # Jika baru mulai tertutup
                    eye_close_start = now    # Catat waktu mulai tertutup
                
                # Cek apakah mata sudah tertutup cukup lama
                eye_closed_long = (now - eye_close_start) >= DROWSY_TIME_SECS
            else:  # Mata terbuka
                eye_close_start = None  # Reset timer

            # ---------- MENGHITUNG POSE KEPALA ----------
            hp = solve_head_pose(lm_list, w, h)  # Menghitung orientasi kepala
            
            if hp:  # Jika berhasil menghitung
                yaw, pitch, roll = hp  # Ekstrak sudut-sudut
                
                # Menghaluskan nilai yaw dengan EMA
                ema_yaw = ema(ema_yaw, yaw, SMOOTH_YAW_ALPHA)
                head_yaw = ema_yaw
                
                # Cek apakah kepala menoleh melewati ambang batas
                if abs(head_yaw) > YAW_HEAD_TURN_THRESH:
                    head_turn = True

        # ---------- FALLBACK: DETEKSI DARI POSE TUBUH ----------
        # Jika wajah tidak terdeteksi tapi pose tubuh terdeteksi
        elif pose_results.pose_landmarks:
            face_present = True        # Tetap anggap ada wajah
            eye_close_start = None     # Reset timer mata (tidak bisa deteksi mata dari pose)
            
            # Deteksi kepala menoleh dari posisi hidung relatif terhadap telinga
            head_turn, _ = detect_head_turn_from_pose(pose_results.pose_landmarks, w, h)

        # ---------- TIDAK ADA DETEKSI SAMA SEKALI ----------
        else:
            eye_close_start = None  # Reset timer mata
            head_turn = False       # Tidak ada indikasi menoleh

        # ---------- LOGIKA PEMICU ALARM ----------
        # Menentukan kondisi tidak fokus berdasarkan mode alarm
        if ALARM_MODE == "either":
            # Mode "either": alarm berbunyi jika mata tertutup ATAU kepala menoleh
            offcenter_flag = eye_closed_long or head_turn
        else:
            # Mode lainnya: alarm berbunyi jika mata tertutup DAN kepala menoleh
            offcenter_flag = eye_closed_long and head_turn

        # ---------- MENGHITUNG DURASI TIDAK FOKUS ----------
        now = time.time()
        
        if face_present and offcenter_flag:  # Jika wajah ada dan tidak fokus
            if offcenter_start is None:      # Jika baru mulai tidak fokus
                offcenter_start = now        # Catat waktu mulai
            elapsed = now - offcenter_start  # Hitung durasi tidak fokus
        else:  # Jika fokus atau tidak ada wajah
            offcenter_start = None  # Reset timer
            elapsed = 0             # Tidak ada durasi tidak fokus

        # ---------- LOGIKA PEMUTARAN ALARM ----------
        # Cek apakah alarm sedang diputar
        alarm_playing = pygame.mixer.music.get_busy() if os.path.exists(ALARM_FILE) else False
        
        # Hitung waktu sejak alarm terakhir berbunyi
        time_since_last = now - last_alarm_time

        # Kondisi untuk membunyikan alarm:
        # 1. Durasi tidak fokus >= batas minimum
        # 2. Alarm tidak sedang diputar
        # 3. Sudah melewati waktu cooldown
        if elapsed >= MIN_OFFCENTER_SECONDS and not alarm_playing and time_since_last > COOLDOWN:
            if os.path.exists(ALARM_FILE):
                pygame.mixer.music.play()  # Putar alarm
            last_alarm_time = now          # Catat waktu alarm berbunyi
            print(f"Alarm triggered (elapsed={elapsed:.2f}s)")  # Cetak log ke konsol

        # ==================== BAGIAN MENAMPILKAN INFORMASI DI LAYAR ====================
        
        # Menampilkan nilai EAR di pojok kiri atas
        cv2.putText(frame, f"EAR : {ear_avg:.3f}" if ear_avg else "EAR : N/A",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Parameter: frame, teks, posisi (x,y), font, ukuran, warna BGR, ketebalan

        # Menampilkan nilai Yaw kepala
        if head_yaw is not None:
            cv2.putText(frame, f"Yaw : {head_yaw:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Menampilkan durasi mata tertutup
        if eye_close_start:
            cv2.putText(frame, f"CloseSec : {(now - eye_close_start):.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        else:
            cv2.putText(frame, "CloseSec : 0.00", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        # Menampilkan durasi tidak fokus
        cv2.putText(frame, f"OffElapsed : {elapsed:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Menentukan dan menampilkan status pengguna
        if offcenter_flag:
            if eye_closed_long and head_turn:
                status = "EYES + HEAD"    # Mata tertutup dan kepala menoleh
            elif eye_closed_long:
                status = "EYES CLOSED"    # Hanya mata tertutup
            elif head_turn:
                status = "HEAD TURN"      # Hanya kepala menoleh
        else:
            status = "FOCUSED"            # Pengguna fokus

        # Menampilkan status di pojok kiri bawah
        cv2.putText(frame, f"Status : {status}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Menampilkan FPS di pojok kanan atas
        cv2.putText(frame, f"FPS : {fps_smoothed:.1f}", (w - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Menampilkan frame ke jendela
        cv2.imshow("FOCUSED DETECTIONS MODEL", frame)

        # ---------- MENANGANI INPUT KEYBOARD ----------
        key = cv2.waitKey(1) & 0xFF  # Tunggu 1ms dan ambil input keyboard
        
        if key in (27, ord('q')):    # Jika tekan ESC (27) atau 'q'
            break                     # Keluar dari loop
            
        if key == ord('m'):          # Jika tekan 'm'
            DRAW_MESH = not DRAW_MESH  # Toggle tampilan mesh wajah
            
        if key == ord('s') and os.path.exists(ALARM_FILE):  # Jika tekan 's'
            pygame.mixer.music.stop()  # Hentikan alarm yang sedang berbunyi

# ==================== BAGIAN PEMBERSIHAN (CLEANUP) ====================
finally:
    # Blok finally SELALU dijalankan, bahkan jika ada error atau program dihentikan
    
    cap.release()              # Melepas koneksi ke kamera
    cv2.destroyAllWindows()    # Menutup semua jendela OpenCV
    face_mesh.close()          # Menutup objek face mesh
    pose.close()               # Menutup objek pose
    
    # Membersihkan pygame dengan penanganan error
    try:
        if os.path.exists(ALARM_FILE):
            pygame.mixer.music.stop()  # Hentikan musik jika sedang diputar
        pygame.mixer.quit()            # Tutup sistem audio pygame
    except:
        pass  # Abaikan error yang mungkin terjadi saat cleanup
