import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os

# --- Initialisation MediaPipe Pose et Hands ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# --- Variables de jeu ---
score = 0
circle_radius = 40
circle1_pos = (200, 200)
circle2_pos = (200, 200)

# --- Obstacle ---
obstacle_active = False
obstacle_type = None
obstacle_progress = 0
obstacle_max = 255
obstacle_zone = None
obstacle_start_time = 0
obstacle_duration = 1
obstacle_cooldown = 1
last_obstacle_time = 0

# --- Charger la tête du joueur ---
head_img_mem = cv2.imread("image/head.png", cv2.IMREAD_UNCHANGED)
head_img = head_img_mem.copy()

# --- Capture webcam ---
cap = cv2.VideoCapture(0)

# --- Ajoute 3 secondes d'attente avant le premier obstacle ---
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # --- MediaPipe Pose ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    right_hand_pos = None
    left_hand_pos = None
    head_pos = None

    if results.pose_landmarks:
        # Utilise le centre de la main (moyenne poignet + index)
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        left_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        x_r = int((right_wrist.x + right_index.x) / 2 * w)
        y_r = int((right_wrist.y + right_index.y) / 2 * h)
        x_l = int((left_wrist.x + left_index.x) / 2 * w)
        y_l = int((left_wrist.y + left_index.y) / 2 * h)
        right_hand_pos = (x_r, y_r)
        left_hand_pos = (x_l, y_l)
        x_n = int(nose.x * w)
        y_n = int(nose.y * h)
        head_pos = (x_n, y_n)
    else:
        head_pos = None

    # --- Obstacles ---
    now = time.time()
    # Ajoute ce test pour attendre 3 secondes avant le premier obstacle
    if now - start_time < 3:
        # Affiche un compte à rebours bien centré
        countdown = 3 - int(now - start_time)
        text = f"Debut dans {countdown}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 4
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
        cv2.imshow("Boxing  Simulator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if not obstacle_active and (now - last_obstacle_time > obstacle_cooldown):
        obstacle_active = True
        obstacle_type = random.choice(['vertical', 'horizontal'])
        if obstacle_type == 'vertical':
            x = random.randint(int(w * 0.4), int(w * 0.6))
            w2 = w
            if x < w / 2:
                w2 = x
                x = 0
            obstacle_zone = (x, 0, w2, h)
        else:
            y = random.randint(int(h * 0.3), int(h * 0.5))
            obstacle_zone = (0, 0, w, y)
        obstacle_progress = 0
        obstacle_start_time = now

    if obstacle_active:
        elapsed = now - obstacle_start_time
        obstacle_progress = min(int((elapsed / obstacle_duration) * obstacle_max), obstacle_max)
        overlay = frame.copy()
        cv2.rectangle(overlay, (obstacle_zone[0], obstacle_zone[1]), (obstacle_zone[2], obstacle_zone[3]), (0, 0, obstacle_progress), -1)
        alpha = 0.4 + 0.6 * (obstacle_progress / obstacle_max)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Si la tête touche la zone quand c'est rouge max, perdu
        if head_pos and obstacle_progress >= obstacle_max:
            x1, y1, x2, y2 = obstacle_zone
            head_x, head_y = head_pos
            if x1 < head_x < x2 and y1 < head_y < y2:
                cv2.putText(frame, "PERDU !", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                cv2.imshow("Boxing  Simulator", frame)
                cv2.waitKey(2000)
                break
        # Retire l'obstacle après passage
        if elapsed > obstacle_duration + 1:
            obstacle_active = False
            last_obstacle_time = now

    # --- Affichage des cibles ---
    cv2.circle(frame, circle1_pos, circle_radius, (0, 0, 255), 5)
    cv2.circle(frame, circle2_pos, circle_radius, (0, 0, 255), 5)

    # --- Vérifie si main touche le cercle ---
    for hand_pos in [right_hand_pos, left_hand_pos]:
        if hand_pos:
            for idx, circle_pos in enumerate([circle1_pos, circle2_pos]):
                dx = hand_pos[0] - circle_pos[0]
                dy = hand_pos[1] - circle_pos[1]
                dist = np.hypot(dx, dy)
                if dist < circle_radius:
                    score += 1
                    print("Hit!", score)
                    if idx == 0:
                        circle1_pos = (random.randint(100, w - 100), random.randint(100, h - 100))
                    else:
                        circle2_pos = (random.randint(100, w - 100), random.randint(100, h - 100))

    # --- Affichage score ---
    cv2.putText(frame, f"Score: {score}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # --- Affiche une tête sur la tête du joueur ---
    if head_pos and results.pose_landmarks:
        head_img = head_img_mem.copy()

        # Utilise la distance et l'angle entre les oreilles
        left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        x_le, y_le = int(left_ear.x * w), int(left_ear.y * h)
        x_re, y_re = int(right_ear.x * w), int(right_ear.y * h)
        head_width = int(np.hypot(x_le - x_re, y_le - y_re) * 3)
        if head_width < 20:
            head_width = 80
        head_height = int(head_width * 1.2)

        # Calcul de l'angle (en degrés)
        angle_rad = np.arctan2(y_le - y_re, x_le - x_re)
        angle_deg = np.degrees(angle_rad)

        head_img = cv2.resize(head_img, (head_width, head_height), interpolation=cv2.INTER_AREA)

        # Rotation de la tête
        center = (head_width // 2, head_height // 2)
        M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        head_img_rot = cv2.warpAffine(
            head_img, M, (head_width, head_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )

        hx, hy = head_pos[0] - head_width // 2, head_pos[1] - head_height // 2
        hh, hw = head_img_rot.shape[:2]
        x1, x2 = max(0, hx), min(w, hx + hw)
        y1, y2 = max(0, hy), min(h, hy + hh)
        img_x1, img_x2 = x1 - hx, hw - (hx + hw - x2)
        img_y1, img_y2 = y1 - hy, hh - (hy + hh - y2)
        if head_img_rot.shape[2] == 4:
            alpha = head_img_rot[img_y1:img_y2, img_x1:img_x2, 3] / 255.0
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha * head_img_rot[img_y1:img_y2, img_x1:img_x2, c] +
                    (1 - alpha) * frame[y1:y2, x1:x2, c]
                ).astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = head_img_rot[img_y1:img_y2, img_x1:img_x2, :3]

    cv2.imshow("Boxing  Simulator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
