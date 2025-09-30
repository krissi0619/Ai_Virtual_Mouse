"""
run_hand_tracking.py
Main script for testing HandTrackingModule.
Keeps the webcam window 'always on top' when supported by OpenCV.
"""

import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy  # for mouse control (optional; ensure installed on your machine)

def webcam_mode():
    wCam, hCam = 1280, 720
    frameR = 100
    smoothening = 7
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    # Use CAP_DSHOW on Windows for better camera support (remove second arg on non-Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Try to set desired resolution (camera may fallback to nearest supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

    detector = htm.handDetector(maxHands=1)

    # screen size for autopy (if autopy available)
    try:
        wScr, hScr = autopy.screen.size()
    except Exception:
        wScr, hScr = wCam, hCam

    # Create window and attempt to make it always on top
    window_name = "Webcam Hand Tracking (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # try to set topmost window property (works on many OpenCV builds)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        # Some OpenCV builds do not support WND_PROP_TOPMOST; ignore if not available
        pass

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Failed to capture frame from camera.")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            fingers = detector.fingersUp()

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)

            # Move Mode
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                try:
                    autopy.mouse.move(wScr - clocX, clocY)
                except Exception:
                    # autopy can fail if not available or in headless environment
                    pass
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Click Mode
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)
                    try:
                        autopy.mouse.click()
                    except Exception:
                        pass

        # FPS (protect against division by zero)
        cTime = time.time()
        dt = cTime - pTime if (cTime - pTime) > 1e-6 else 1e-6
        fps = 1.0 / dt
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        # Keep window on top each loop if supported (some builds require repeated setting)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def image_mode(image_path):
    detector = htm.handDetector(maxHands=2)
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        return
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if lmList:
        print("Hand landmarks found:", len(lmList))
    cv2.imshow("Image Hand Tracking", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    choice = input("Choose mode: (1) Webcam  (2) Image => ").strip()
    if choice == "1":
        webcam_mode()
    elif choice == "2":
        path = input("Enter image path: ").strip()
        image_mode(path)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
