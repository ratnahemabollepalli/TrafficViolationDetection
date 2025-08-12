import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import pytesseract  # Correct package name
import re
import csv
from datetime import datetime, timedelta

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class AccurateLicensePlateDetector:
    def __init__(self, video_path, plate_model_path, vehicle_model_path, speed_limit=60, callback=None, should_stop=None):
        self.video_path = video_path
        self.plate_model = YOLO(plate_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.callback = callback  # Callback function for live updates
        self.should_stop = should_stop  # Function to check if processing should stop

        # Initialize trackers
        self.frame_count = 0
        self.plate_text = ""
        self.plate_conf = 0
        self.plate_update_freq = 5

        # Speed calculation
        self.track_history = defaultdict(list)
        self.pixel_to_km = 0.0001  # Calibrate this value based on your camera setup
        self.fps = max(self.cap.get(cv2.CAP_PROP_FPS), 1)
        self.speed_unit = "km/h"
        self.speed_limit = speed_limit

        # OCR config
        self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '

        # CSV setup
        self.csv_file = "speed_violations.csv"
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["License Plate", "Speed", "Date", "Time"])

        # Store last log times for each plate
        self.logged_plates = {}
        self.log_cooldown = timedelta(minutes=5)  # Prevent duplicate logs for same plate

        print(f"Initialized detector | FPS: {self.fps} | Speed limit: {self.speed_limit} {self.speed_unit}")

    def preprocess_plate(self, plate_roi):
        """Enhance license plate image for better OCR results"""
        if plate_roi.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Apply morphological operations to clean up image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    def extract_plate_text(self, plate_roi):
        """Perform OCR on license plate image"""
        processed = self.preprocess_plate(plate_roi)
        if processed is None:
            return "NO-PLATE", 0.0

        # Use Tesseract OCR
        text = pytesseract.image_to_string(processed, config=self.ocr_config)

        # Clean up OCR results
        text = re.sub(r'[^A-Z0-9-]', '', text.upper()).strip()

        # Confidence scoring
        if len(text) >= 6:  # Minimum characters for a valid plate
            return text, 0.9
        elif len(text) > 0:
            return text, 0.5
        else:
            return "UNREADABLE", 0.1

    def calculate_speed(self, displacement_pixels):
        """Convert pixel movement to km/h"""
        return displacement_pixels * self.pixel_to_km * self.fps * 3600  # km/h

    def log_speed_violation(self, plate_text, speed):
        """Record violation to CSV and trigger callback"""
        now = datetime.now()

        # Skip if this plate was recently logged
        if plate_text in self.logged_plates:
            if now - self.logged_plates[plate_text] < self.log_cooldown:
                return

        # Prepare violation data
        violation = {
            "License Plate": plate_text,
            "Speed": round(speed, 2),
            "Date": now.date().strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S")
        }

        # Write to CSV
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                violation["License Plate"],
                violation["Speed"],
                violation["Date"],
                violation["Time"]
            ])

        # Update last logged time
        self.logged_plates[plate_text] = now

        # Trigger callback for live updates
        if self.callback:
            self.callback(violation)

    def process_frame(self, frame):
        """Process a single video frame for violations"""
        self.frame_count += 1

        # Check if processing should stop
        if self.should_stop and self.should_stop():
            return None

        # Detect vehicles
        vehicle_results = self.vehicle_model.track(frame, persist=True, verbose=False)

        # Detect license plates
        plate_results = self.plate_model(frame, verbose=False)

        # Process vehicle tracking
        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Update tracking history
                self.track_history[track_id].append(box)
                if len(self.track_history[track_id]) > 10:
                    self.track_history[track_id].pop(0)

                # Calculate speed if we have enough frames
                if len(self.track_history[track_id]) >= 2:
                    prev_box = self.track_history[track_id][-2]

                    # Calculate center point movement
                    dx = (box[0] + box[2]) / 2 - (prev_box[0] + prev_box[2]) / 2
                    dy = (box[1] + box[3]) / 2 - (prev_box[1] + prev_box[3]) / 2
                    speed = self.calculate_speed(np.sqrt(dx ** 2 + dy ** 2))

                    # Check for speed violation
                    if speed > self.speed_limit:
                        # Process all detected plates in this frame
                        for plate in plate_results[0].boxes:
                            px1, py1, px2, py2 = map(int, plate.xyxy[0].cpu().numpy())
                            plate_roi = frame[py1:py2, px1:px2]
                            plate_text, _ = self.extract_plate_text(plate_roi)
                            self.log_speed_violation(plate_text, speed)

        return frame

    def run(self):
        """Main processing loop"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Check if processing should stop
            if self.should_stop and self.should_stop():
                break

            processed_frame = self.process_frame(frame)
            if processed_frame is None:  # Processing was stopped
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
