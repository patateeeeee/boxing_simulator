import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os

class BoxingSimulator:
    def __init__(self):
        # --- Initialisation MediaPipe Pose ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # --- Variables de jeu ---
        self.score = 0
        self.circle_radius = 40
        self.circle1_pos = (200, 200)
        self.circle2_pos = (200, 200)

        # --- Obstacle ---
        self.obstacle_active = False
        self.obstacle_type = None
        self.obstacle_progress = 0
        self.obstacle_max = 255
        self.obstacle_zone = None
        self.obstacle_start_time = 0
        self.obstacle_duration = 1
        self.obstacle_cooldown = 1
        self.last_obstacle_time = 0

    def detect_pose(self, frame):
        """Détecte la pose et retourne les positions des mains et de la tête"""
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        right_hand_pos = None
        left_hand_pos = None
        head_pos = None

        if results.pose_landmarks:
            # Utilise le centre de la main (moyenne poignet + index)
            right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_index = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX]
            left_index = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_INDEX]
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

            x_r = int((right_wrist.x + right_index.x) / 2 * w)
            y_r = int((right_wrist.y + right_index.y) / 2 * h)
            x_l = int((left_wrist.x + left_index.x) / 2 * w)
            y_l = int((left_wrist.y + left_index.y) / 2 * h)
            right_hand_pos = (x_r, y_r)
            left_hand_pos = (x_l, y_l)
            x_n = int(nose.x * w)
            y_n = int(nose.y * h)
            head_pos = (x_n, y_n)

        return right_hand_pos, left_hand_pos, head_pos

    def update_obstacles(self, frame, head_pos, now, start_time):
        """Met à jour les obstacles et vérifie les collisions"""
        h, w, _ = frame.shape
        
        # Compte à rebours au début
        if now - start_time < 3:
            countdown = 3 - int(now - start_time)
            text = f"Start in {countdown}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 4
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
            return frame, False  # Game not started yet

        # Génération d'obstacles
        if not self.obstacle_active and (now - self.last_obstacle_time > self.obstacle_cooldown):
            self.obstacle_active = True
            self.obstacle_type = random.choice(['vertical', 'horizontal'])
            if self.obstacle_type == 'vertical':
                x = random.randint(int(w * 0.4), int(w * 0.6))
                w2 = w
                if x < w / 2:
                    w2 = x
                    x = 0
                self.obstacle_zone = (x, 0, w2, h)
            else:
                y = random.randint(int(h * 0.3), int(h * 0.5))
                self.obstacle_zone = (0, 0, w, y)
            self.obstacle_progress = 0
            self.obstacle_start_time = now

        # Affichage des obstacles
        if self.obstacle_active:
            elapsed = now - self.obstacle_start_time
            self.obstacle_progress = min(int((elapsed / self.obstacle_duration) * self.obstacle_max), self.obstacle_max)
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.obstacle_zone[0], self.obstacle_zone[1]), 
                         (self.obstacle_zone[2], self.obstacle_zone[3]), (0, 0, self.obstacle_progress), -1)
            alpha = 0.4 + 0.6 * (self.obstacle_progress / self.obstacle_max)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Collision avec la tête
            if head_pos and self.obstacle_progress >= self.obstacle_max:
                x1, y1, x2, y2 = self.obstacle_zone
                head_x, head_y = head_pos
                if x1 < head_x < x2 and y1 < head_y < y2:
                    cv2.putText(frame, "LOST !", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    return frame, True  # Game over
            
            # Retirer l'obstacle après passage
            if elapsed > self.obstacle_duration + 1:
                self.obstacle_active = False
                self.last_obstacle_time = now

        return frame, False  # Continue game

    def update_targets(self, frame, right_hand_pos, left_hand_pos):
        """Met à jour les cibles et vérifie les hits"""
        h, w, _ = frame.shape
        
        # Affichage des cibles
        cv2.circle(frame, self.circle1_pos, self.circle_radius, (0, 0, 255), 5)
        cv2.circle(frame, self.circle2_pos, self.circle_radius, (0, 0, 255), 5)

        # Vérification des hits
        for hand_pos in [right_hand_pos, left_hand_pos]:
            if hand_pos:
                for idx, circle_pos in enumerate([self.circle1_pos, self.circle2_pos]):
                    dx = hand_pos[0] - circle_pos[0]
                    dy = hand_pos[1] - circle_pos[1]
                    dist = np.hypot(dx, dy)
                    if dist < self.circle_radius:
                        self.score += 1
                        print("Hit!", self.score)
                        if idx == 0:
                            self.circle1_pos = (random.randint(100, w - 100), random.randint(100, h - 100))
                        else:
                            self.circle2_pos = (random.randint(100, w - 100), random.randint(100, h - 100))

        return frame

    def draw_score(self, frame):
        """Affiche le score sur la frame"""
        cv2.putText(frame, f"Score: {self.score}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        return frame

    def process_frame(self, frame, start_time):
        """Traite une frame complète du jeu"""
        now = time.time()
        
        # Détection de pose
        right_hand_pos, left_hand_pos, head_pos = self.detect_pose(frame)
        
        # Gestion des obstacles
        frame, game_over = self.update_obstacles(frame, head_pos, now, start_time)
        if game_over:
            return frame, True
            
        # Gestion des cibles (seulement si le jeu a commencé)
        if now - start_time >= 3:
            frame = self.update_targets(frame, right_hand_pos, left_hand_pos)
            frame = self.draw_score(frame)
        
        return frame, False