import cv2
import cvzone  # cvzone version: 1.4.1
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
final_text = ""

keyboard = Controller()

def draw_all(img, button_list):
    # for button in button_list:
    #     x, y = button.position
    #     width, height = button.size
    #     cv2.rectangle(img, button.position, (x + width, y + height), (255, 0, 255), cv2.FILLED)
    #     cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    
    # return img
    img_new = np.zeros_like(img, np.uint8)
    for button in button_list:
        x, y = button.position
        cvzone.cornerRect(img_new, (button.position[0], button.position[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(img_new, button.position, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
        cv2.putText(img_new, button.text, (x + 40, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = img_new.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, img_new, 1 - alpha, 0)[mask]
    return out


class Button():
    def __init__(self, position, text, size=[85, 85]):
        self.position = position
        self.text = text
        self.size = size


button_list = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        button_list.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmark_list, bounding_box_info = detector.findPosition(img)
    img = draw_all(img, button_list)

    if landmark_list:
        for button in button_list:
            x, y = button.position
            width, height = button.size

            if x < landmark_list[8][0] < x + width and y < landmark_list[8][1] < y + height:
                cv2.rectangle(img, button.position, (x + width, y + height), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                length, _, _ = detector.findDistance(8, 12, img, draw=False)
                # print(length)
            
                if length < 30:
                    keyboard.press(button.text)
                    cv2.rectangle(img, button.position, (x + width, y + height), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    final_text += button.text
                    sleep(0.2)
    
    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, final_text, (60, 440), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
