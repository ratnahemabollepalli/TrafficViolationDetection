from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from main import AccurateLicensePlateDetector
import threading
import time
import cv2

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = "video_input"
CSV_FILE = "speed_violations.csv"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global processing state
processing_state = {
    'is_processing': False,
    'progress': 0,
    'violations': [],
    'video_path': None,
    'should_stop': False
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video(video_path, speed_limit):
    global processing_state
    processing_state['is_processing'] = True
    processing_state['progress'] = 0
    processing_state['violations'] = []
    processing_state['video_path'] = video_path
    processing_state['should_stop'] = False

    try:
        # Get total frames for progress calculation
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        detector = AccurateLicensePlateDetector(
            video_path=video_path,
            plate_model_path="best.pt",
            vehicle_model_path="yolov8n.pt",
            speed_limit=speed_limit,
            callback=lambda v: update_processing_status(v, total_frames),
            should_stop=lambda: processing_state['should_stop']
        )
        detector.run()
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        processing_state['is_processing'] = False
        processing_state['progress'] = 100


def update_processing_status(violation, total_frames):
    global processing_state
    if processing_state['should_stop']:
        return

    processing_state['violations'].append(violation)

    # Update progress based on frame count
    if total_frames > 0:
        processing_state['progress'] = min(100, int((len(processing_state['violations']) / total_frames) * 100 * 5))


@app.route("/")
def home():
    return render_template("getstarted.html")


@app.route("/getstarted")
def getstarted():
    return redirect(url_for("upload_video"))


@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['video']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            if not os.path.exists(video_path):
                flash('File upload failed')
                return redirect(request.url)

            try:
                speed_limit = int(request.form.get('speed_limit', 60))
            except:
                speed_limit = 60

            thread = threading.Thread(target=process_video, args=(video_path, speed_limit))
            thread.start()

            return redirect(url_for("results"))

    return render_template("upload.html")


@app.route("/results")
def results():
    return render_template("results.html")


@app.route("/api/status")
def get_status():
    return jsonify(processing_state)


@app.route("/api/stop")
def stop_processing():
    processing_state['should_stop'] = True
    return jsonify({'success': True})


@app.route("/download")
def download_csv():
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True)
    flash('No CSV found. Process a video first.')
    return redirect(url_for("upload_video"))


if __name__ == "__main__":
    app.run(debug=True)