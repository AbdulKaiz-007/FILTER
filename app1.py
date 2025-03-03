import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import cvzone
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller
import numpy as np
import time

class EyeHandControlApp:
    def __init__(self):
        # Eye Control Setup
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Hand Keyboard Setup
        self.keys = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]
        ]
        self.detector = HandDetector(detectionCon=0.8)
        self.keyboard = Controller()
        self.buttonList = [
            Button((100 * j + 50, 100 * i + 50), key) 
            for i, row in enumerate(self.keys) 
            for j, key in enumerate(row)
        ]
        self.finalText = ""

    def draw_buttons(self, img, buttonList):
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, button.text, (x + 20, y + 65),
                        cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        return img

    def run_eye_control(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape

        if output.multi_face_landmarks:
            landmarks = output.multi_face_landmarks[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                
                if id == 1:
                    screen_x = self.screen_w / frame_w * x
                    screen_y = self.screen_h / frame_h * y
                    pyautogui.moveTo(screen_x, screen_y)

            # Blink detection
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))

            if (left[0].y - left[1].y) < 0.004:
                pyautogui.click()
                time.sleep(0.2)

        cv2.putText(frame, "Eye Control", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def run_hand_keyboard(self, frame):
        hands, frame = self.detector.findHands(frame)
        lmList = hands[0]['lmList'] if hands else []
        
        frame = self.draw_buttons(frame, self.buttonList)

        if lmList:
            for button in self.buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), 
                                  (175, 0, 175), cv2.FILLED)
                    cv2.putText(frame, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    l, _, _ = self.detector.findDistance(lmList[8][:2], lmList[12][:2])

                    # When clicked
                    if l < 30:
                        self.keyboard.press(button.text)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        self.finalText += button.text
                        time.sleep(0.15)

        # Display typed text
        cv2.rectangle(frame, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
        cv2.putText(frame, self.finalText, (60, 430),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        cv2.putText(frame, "Hand Keyboard", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

class Button:
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.size = size
        self.text = text

def main():
    st.title("Eye and Hand Control Interactive App")
    
    app = EyeHandControlApp()

    # Two column layout for side-by-side video feeds
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Eye Control")
        eye_frame = st.empty()
    
    with col2:
        st.header("Hand Keyboard")
        keyboard_frame = st.empty()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)

        # Process and display eye control
        eye_result = app.run_eye_control(frame.copy())
        eye_result_rgb = cv2.cvtColor(eye_result, cv2.COLOR_BGR2RGB)
        eye_frame.image(eye_result_rgb, channels="RGB")

        # Process and display hand keyboard
        keyboard_result = app.run_hand_keyboard(frame.copy())
        keyboard_result_rgb = cv2.cvtColor(keyboard_result, cv2.COLOR_BGR2RGB)
        keyboard_frame.image(keyboard_result_rgb, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()