import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import queue
from constant import *

class PersonDetection:
    def __init__(self, model_path = MODEL_PATH):
        self.model = YOLO(model_path)

        # Initialize Video Capture
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Initialize text to speech engine
        self.engine = pyttsx3.init()
        self.speech_queue = queue.Queue()
        self.last_alert = None
        self.FOCAL_LENGTH = None

        self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()

        
    def speech_worker(self):
        """Thread function to process speech queue"""

        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()
            self.speech_queue.task_done()


    def run(self):
        """main loop for detecting person and distance estimation"""

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            small_frame = cv2.resize(frame, (FRAME_RESIZE, FRAME_RESIZE))
            results = self.model.predict(small_frame, conf=CONF)
            annotated_frame = results[0].plot()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    height_pixels = y2 - y1


                    if height_pixels > 0:
                        if self.FOCAL_LENGTH is None:
                            self.FOCAL_LENGTH = (height_pixels * REFERENCE_DISTANCE) / KNOWN_HEIGHT

                        # Calculate distance
                        distance = (KNOWN_HEIGHT * self.FOCAL_LENGTH) / height_pixels
                        distance_meter = round(distance, 2)

                        # Draw Bounding box and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            annotated_frame, f"Person: {distance_meter}m",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )

                        # Alert message logic
                        if distance_meter < 1.0:
                            alert_text = "Person! Person! Person!"
                        else:
                            alert_text = f"Person detected at {distance_meter} meters."

                        if alert_text != self.last_alert and self.speech_queue.qsize() == 0:
                            self.speech_queue.put(alert_text)
                            self.last_alert = alert_text


            cv2.imshow("YOLO Detection & Distance Estimation", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.speech_queue.put(None)
        self.speech_thread.join(timeout=3)
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    detector = PersonDetection()
    try:
        detector.run()
    except Exception as e:
        print(e)        

