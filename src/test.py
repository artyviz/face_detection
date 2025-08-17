import os
import time
import queue
import threading
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from pathlib import Path

IMAGE_DIR = Path("C:/Users/Farhan/Desktop/face detection/images")
CSV_FILE = "attendance.csv"
FRAME_QUEUE_MAX = 6
DOWNSCALE = 0.5
DETECT_EVERY_N = 2
SIM_THRESHOLD = 0.35
USE_GPU_CTX = 0

if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)
pd.DataFrame(columns=["Name", "Time"]).to_csv(CSV_FILE, index=False)

print("[INFO] Initializing FaceAnalysis (this may download models the first time)...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=USE_GPU_CTX)
print("[INFO] FaceAnalysis ready. GPU ctx:", USE_GPU_CTX)

known_embeddings = []
classnames = []
for fname in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] could not read {path}, skipping")
        continue
    faces = app.get(img)
    if len(faces) == 0:
        print(f"[WARN] no face found in {fname}, skipping")
        continue
    emb = faces[0].normed_embedding
    known_embeddings.append(emb.astype(np.float32))
    classnames.append(os.path.splitext(fname)[0])
if len(known_embeddings) == 0:
    raise RuntimeError("No valid gallery faces found. Populate image_folder with face images.")
known_matrix = np.stack(known_embeddings)
print(f"[INFO] Loaded gallery: {len(classnames)} faces -> {classnames}")

marked_names = set()

def mark_attendance(name):
    if name in marked_names:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a") as f:
        f.write(f"{name},{now}\n")
    marked_names.add(name)
    print(f"[ATTEND] {name} @ {now}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open laptop camera")

trackers = []
frame_count = 0
FPS_TIMER = time.time()
frame_seen = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] empty frame grabbed")
        time.sleep(0.1)
        continue

    frame_count += 1
    frame_seen += 1

    small = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    updated_boxes = []
    alive_trackers = []
    for tdict in trackers:
        ok, box = tdict["tracker"].update(frame)
        if ok and tdict["ttl"] > 0:
            x, y, w, h = [int(v) for v in box]
            updated_boxes.append((x, y, w, h, tdict.get("name")))
            tdict["ttl"] -= 1
            alive_trackers.append(tdict)
    trackers = alive_trackers

    if frame_count % DETECT_EVERY_N == 0 or len(trackers) == 0:
        faces = app.get(frame)
        if faces:
            trackers = []
            for f in faces:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                try:
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                emb = f.normed_embedding.astype(np.float32)
                sims = known_matrix @ emb
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                name = None
                if best_sim >= SIM_THRESHOLD:
                    name = classnames[best_idx].upper()
                    mark_attendance(name)
                trackers.append({"tracker": tracker, "name": name, "ttl": DETECT_EVERY_N * 3})

    for (x, y, w, h, name) in updated_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = name if name else "Unknown"
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for tdict in trackers:
        ok, box = tdict["tracker"].update(frame)
        if ok:
            x, y, w, h = [int(v) for v in box]
            name = tdict.get("name") or "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(frame, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if frame_seen % 30 == 0:
        now = time.time()
        fps = 30.0 / max(1e-6, (now - FPS_TIMER))
        FPS_TIMER = now

    cv2.imshow("Attendance (InsightFace GPU)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] finished. Attendance CSV:", os.path.abspath(CSV_FILE))