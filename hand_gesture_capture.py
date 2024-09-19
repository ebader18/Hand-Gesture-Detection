import cv2
import mediapipe as mp
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True, type=str,help="Camera resolution. Ex: 1280 720")
args = vars(ap.parse_args())

hres = int(args["res"].split()[0])
vres = int(args["res"].split()[1])

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, hres)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vres)

while True:
    t0 = time.time()
    success, img = cap.read()
    h, w, c = img.shape

    results = hands.process(img)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    t1 = time.time()
    cv2.putText(img, str(int(1.0/(t1-t0))) + ' fps',(0, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 2)
    cv2.putText(img, 'Resolution: ' + str(w) + 'x' + str(h),(0, 60), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
    cv2.imshow("Image", img)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break

cv2.destroyAllWindows()
