import csv
import os
import queue
import sys
from datetime import date
from pathlib import Path

import cv2
import supervision as sv
import torch
from dotenv import load_dotenv
from inference import InferencePipeline
from trackers import ByteTrackTracker

# torch's generic device-index resolution expects every accelerator backend to
# expose current_device(), but torch.mps doesn't on this version - MPS only
# ever has one device, so index 0 is always correct.
if not hasattr(torch.mps, "current_device"):
    torch.mps.current_device = lambda: 0

load_dotenv()

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "amsterdam-boat-spotting-v2/1")
RTSP_URL = os.environ.get("RTSP_URL")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
LINE_COORDS = os.environ.get("LINE_COORDS")  # "x1,y1,x2,y2", optional

if not API_KEY:
    sys.exit("ROBOFLOW_API_KEY is not set. Copy .env.example to .env and fill it in.")
if not RTSP_URL:
    sys.exit("RTSP_URL is not set. Copy .env.example to .env and fill it in.")

DAILY_COUNTS_FILE = Path(__file__).parent / "daily_counts.csv"

tracker = ByteTrackTracker()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_zone_annotator = sv.LineZoneAnnotator(text_scale=1, thickness=2)

# Built lazily on the first frame, since we need the frame size to place
# a default line (unless LINE_COORDS was given explicitly).
line_zone = None
frame_dims = None  # (width, height), cached so the line can be rebuilt on day rollover
current_day = None

# on_prediction runs on a background thread (see use_main_thread=False below)
# so the main thread stays free to keep calling cv2.waitKey. Without this,
# a stalled RTSP frame (camera hiccup) stops cv2 from pumping macOS's event
# loop and the window shows the spinning "beach ball" until frames resume.
frame_queue: "queue.Queue" = queue.Queue(maxsize=1)


def build_line_zone(frame_width, frame_height):
    if LINE_COORDS:
        x1, y1, x2, y2 = (int(v) for v in LINE_COORDS.split(","))
    else:
        x1, y1, x2, y2 = frame_width // 2, 0, frame_width // 2, frame_height
    print(f"Counting line: ({x1}, {y1}) -> ({x2}, {y2}). "
          f"Set LINE_COORDS in .env to reposition it.")
    return sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))


def log_daily_total(day, line_zone):
    file_exists = DAILY_COUNTS_FILE.exists()
    with open(DAILY_COUNTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "total_boats"])
        writer.writerow([day.isoformat(), line_zone.in_count + line_zone.out_count])


def on_prediction(result, video_frame):
    global line_zone, frame_dims, current_day

    frame = video_frame.image
    today = date.today()

    if line_zone is None:
        frame_dims = (frame.shape[1], frame.shape[0])
        line_zone = build_line_zone(*frame_dims)
        current_day = today
    elif today != current_day:
        log_daily_total(current_day, line_zone)
        print(f"New day ({today.isoformat()}) - resetting boat count to zero. "
              f"Previous day's total saved to {DAILY_COUNTS_FILE.name}.")
        line_zone = build_line_zone(*frame_dims)
        current_day = today

    detections = sv.Detections.from_inference(result)
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = tracker.update(detections, frame)

    # LineZone keys crossings by tracker_id, so a boat that lingers near
    # the line for many frames only increments the count once.
    line_zone.trigger(detections)

    labels = [
        f"boat {tracker_id} {confidence:.2f}"
        for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)
    ]
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    line_zone_annotator.annotate(frame=annotated, line_counter=line_zone)

    total = line_zone.in_count + line_zone.out_count
    cv2.putText(annotated, f"Total boats counted: {total}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Drop any unshown frame so the display loop always has the latest one.
    try:
        frame_queue.get_nowait()
    except queue.Empty:
        pass
    frame_queue.put_nowait(annotated)


pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    video_reference=RTSP_URL,
    on_prediction=on_prediction,
    api_key=API_KEY,
)

print("Starting pipeline. Press 'q' in the video window to stop.")
pipeline.start(use_main_thread=False)

while True:
    try:
        annotated = frame_queue.get(timeout=0.05)
        cv2.imshow("Boat Counter", annotated)
    except queue.Empty:
        pass
    # Called every ~50ms regardless of new frames, so macOS keeps seeing
    # the window as responsive even during a stream stall.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        pipeline.terminate()
        break

pipeline.join()
cv2.destroyAllWindows()
