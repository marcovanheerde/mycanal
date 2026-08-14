import csv
import json
import os
import queue
import subprocess
import sys
import time
from datetime import date, datetime, timezone
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
PUBLISH_ENABLED = os.environ.get("PUBLISH_ENABLED", "false").lower() == "true"
PUBLISH_INTERVAL_SECONDS = float(os.environ.get("PUBLISH_INTERVAL_SECONDS", "300"))

if not API_KEY:
    sys.exit("ROBOFLOW_API_KEY is not set. Copy .env.example to .env and fill it in.")
if not RTSP_URL:
    sys.exit("RTSP_URL is not set. Copy .env.example to .env and fill it in.")

PROJECT_DIR = Path(__file__).parent
DAILY_COUNTS_FILE = PROJECT_DIR / "daily_counts.csv"
DATA_DIR = PROJECT_DIR / "data"
STATUS_FILE = DATA_DIR / "status.json"
LATEST_IMAGE_FILE = DATA_DIR / "latest.jpg"
DATA_DIR.mkdir(exist_ok=True)

last_publish_time = 0.0

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
    # Default triggering_anchors checks all 4 box corners and skips any frame
    # where corners fall on both sides at once ("ambiguous"). A boat's box is
    # often wide relative to the line, so it can straddle both sides for many
    # consecutive frames and never resolve - silently preventing any count.
    # A single center point can't straddle, so it crosses cleanly.
    return sv.LineZone(
        start=sv.Point(x1, y1), end=sv.Point(x2, y2),
        triggering_anchors=[sv.Position.CENTER],
    )


def log_daily_total(day, line_zone):
    file_exists = DAILY_COUNTS_FILE.exists()
    with open(DAILY_COUNTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "total_boats"])
        writer.writerow([day.isoformat(), line_zone.in_count + line_zone.out_count])


def read_history():
    if not DAILY_COUNTS_FILE.exists():
        return []
    with open(DAILY_COUNTS_FILE, newline="") as f:
        return [
            {"date": row["date"], "count": int(row["total_boats"])}
            for row in csv.DictReader(f)
        ]


def publish_status(annotated_frame, today, total):
    # Downscale for publishing - the local display window keeps full res,
    # but every published frame gets committed to git history, so keeping
    # these small matters for repo size over weeks of continuous running.
    h, w = annotated_frame.shape[:2]
    if w > 960:
        scale = 960 / w
        annotated_frame = cv2.resize(annotated_frame, (960, int(h * scale)))
    cv2.imwrite(str(LATEST_IMAGE_FILE), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "today": {"date": today.isoformat(), "count": total},
        "history": read_history(),
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

    try:
        subprocess.run(
            ["git", "add", str(STATUS_FILE), str(LATEST_IMAGE_FILE), str(DAILY_COUNTS_FILE)],
            cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=PROJECT_DIR, capture_output=True,
        )
        if diff.returncode == 0:
            return  # nothing changed since the last publish
        subprocess.run(
            ["git", "commit", "-m", f"Publish: {total} boats on {today.isoformat()}"],
            cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "push"], cwd=PROJECT_DIR, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Publish failed (will retry next interval): {e.stderr.strip()}")


def on_prediction(result, video_frame):
    global line_zone, frame_dims, current_day, last_publish_time

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
        if PUBLISH_ENABLED:
            publish_status(frame, today, 0)
            last_publish_time = time.time()

    detections = sv.Detections.from_inference(result)
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = tracker.update(detections, frame)

    # LineZone keys crossings by tracker_id, so a boat that lingers near
    # the line for many frames only increments the count once.
    crossed_in, crossed_out = line_zone.trigger(detections)
    just_counted = bool(crossed_in.any() or crossed_out.any())

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

    # Publish immediately when a boat is actually counted, so the published
    # snapshot shows the boat itself rather than an empty canal - the interval
    # publish below is just a freshness fallback for stretches with no boats.
    if PUBLISH_ENABLED and (just_counted or time.time() - last_publish_time >= PUBLISH_INTERVAL_SECONDS):
        publish_status(annotated, today, total)
        last_publish_time = time.time()


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
