import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # Initializing MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.max_hands, 
            min_detection_confidence=self.detection_con, 
            min_tracking_confidence=self.track_con
        )

        self.mpDraw = mp.solutions.drawing_utils  

    def findHands(self, img, draw=True):
        # Convert to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the RGB image
        self.results = self.hands.process(imgRGB)

        # Draw landmarks if detected
        if self.results.multi_hand_landmarks:
            for handLMK in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMK, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):
        lnList = []

        # Check if landmarks are detected
        if self.results.multi_hand_landmarks:
            myHands = self.results.multi_hand_landmarks[handNum]

            for id, ln in enumerate(myHands.landmark):
                # Convert normalized coordinates to pixel values
                h, w, c = img.shape
                cx, cy = int(ln.x * w), int(ln.y * h)
                lnList.append([id, cx, cy])
                if draw:
                    # Draw circles on index finger and thumb landmarks
                    cv2.circle(img, (cx, cy), 8, (234, 255, 0), cv2.FILLED)
        return lnList

def main():
    cam = cv2.VideoCapture(0)
    detector = handDetector()
    ptime = 0  # Initialize previous time for FPS calculation

    while True:
        # Read frame from webcam
        success, img = cam.read()
        if not success:
            print("Failed to capture image")
            break

        img = detector.findHands(img)
        lnList = detector.findPosition(img)

        # Print position of landmark with ID 4 (thumb tip) if detected
        if len(lnList) != 0:
            print("Thumb tip position:", lnList[4])

        # FPS calculation
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 3)

        # Display the frame
        cv2.imshow("Image", img)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
