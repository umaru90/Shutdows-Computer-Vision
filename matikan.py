import cv2
import mediapipe as mp
import os
import numpy as np
import pyautogui

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk mendeteksi gestur jari tengah
def is_middle_finger_gesture(hand_landmarks):
    # Ambil landmark untuk jari-jari tertentu
    landmarks = hand_landmarks.landmark
    
    # Titik-titik untuk jari tengah
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Titik-titik untuk jari lainnya (untuk memastikan jari lain terlipat)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    
    # Cek apakah jari tengah terangkat (tip y lebih kecil dari pip y)
    middle_finger_extended = middle_tip.y < middle_pip.y
    
    # Cek apakah jari lain terlipat (tip y lebih besar dari pip y)
    other_fingers_folded = (
        (index_tip.y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y) and
        (ring_tip.y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y) and
        (pinky_tip.y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y)
    )
    
    # Cek apakah ibu jari terlipat atau tidak mengganggu
    thumb_folded = thumb_tip.x > landmarks[mp_hands.HandLandmark.THUMB_IP].x
    
    return middle_finger_extended and other_fingers_folded and thumb_folded

# Fungsi untuk shutdown Windows
def shutdown_pc():
    os.system("shutdown /s /t 1")

# Main program
cap = cv2.VideoCapture(0)
gesture_detected_frames = 0
required_frames = 10  # Harus terdeteksi selama 10 frame berturut-turut

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    # Flip image horizontal dan konversi warna
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Proses deteksi tangan
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Cek gestur
            if is_middle_finger_gesture(hand_landmarks):
                gesture_detected_frames += 1
                cv2.putText(image, "GESTURE DETECTED!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Jika terdeteksi cukup lama, shutdown PC
                if gesture_detected_frames >= required_frames:
                    cv2.putText(image, "SHUTTING DOWN...", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Shutdown Gesture', image)
                    cv2.waitKey(1000)
                    shutdown_pc()
            else:
                gesture_detected_frames = 0
    else:
        gesture_detected_frames = 0
    
    # Tampilkan image
    cv2.imshow('Shutdown Gesture', image)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()