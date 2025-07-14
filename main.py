import cv2
import time
from faceswap import FaceSwapper
from boxing import BoxingSimulator

# --- Initialiser le BoxingSimulator ---
boxing = BoxingSimulator()

# --- Initialiser le FaceSwapper ---
face_swapper = FaceSwapper("image/head.png")
frame_count = 0

# --- Capture webcam ---
cap = cv2.VideoCapture(0)

# --- Temps de démarrage ---
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # --- Face Swapping ---
    if face_swapper.reference_image is None:
        print("Erreur: Impossible de charger l'image de référence")
    
    # Appliquer le face swap
    frame = face_swapper.process_frame(frame, frame_count)
    frame_count += 1

    # --- Boxing Simulator ---
    frame, game_over = boxing.process_frame(frame, start_time)
    
    if game_over:
        cv2.imshow("Boxing Simulator", frame)
        cv2.waitKey(2000)
        break

    cv2.imshow("Boxing Simulator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
