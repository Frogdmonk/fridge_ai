import os
import re
import uuid
import threading
from collections import defaultdict

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq


load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DETECTIONS_FOLDER = os.path.join("static", "detections")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)


print("Loading YOLO model...")
yolo_model = YOLO("yolov8m.pt")

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("fridge-recipes")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


FOOD_CLASSES = {
    "apple", "orange", "banana", "broccoli", "carrot",
    "sandwich", "pizza", "cake", "donut", "hot dog",
    "bottle", "wine glass", "cup", "bowl", "fork",
    "knife", "spoon", "refrigerator"
}

HIGH_RISK_CLASSES = {"donut", "cake", "sandwich", "hot dog", "bowl"}

BASE_CONFIDENCE_THRESHOLD = 0.45
HIGH_RISK_CONFIDENCE_THRESHOLD = 0.65

MIN_FRAME_HITS = 3

MOTION_DIFF_SIZE = (64, 64)
MOTION_THRESHOLD = 8.0
FALLBACK_INTERVAL = 15
BATCH_SIZE = 8
INFERENCE_SIZE = 480

BOX_COLORS_BGR = [
    (0, 255, 0), (0, 140, 255), (255, 200, 0), (200, 0, 255),
    (0, 255, 255), (255, 0, 150), (150, 255, 0), (80, 80, 255)
]
BOX_COLORS_HEX = [
    "#4ade80", "#ff8c00", "#00c8ff", "#ff00c8",
    "#ffff00", "#9600ff", "#00ff96", "#ff5050"
]


# In-memory job store. Persists ingredient list + retrieved recipes so
# /regenerate can re-run retrieval+generation without reprocessing video.
PROGRESS = {}


def init_job(job_id):
    PROGRESS[job_id] = {
        "steps": [
            {"key": "upload", "label": "Video uploaded", "status": "done"},
            {"key": "detect", "label": "Detecting ingredients (smart sampling)", "status": "pending"},
            {"key": "embed", "label": "Embedding ingredient query", "status": "pending"},
            {"key": "retrieve", "label": "Retrieving + ranking recipes (Pinecone)", "status": "pending"},
            {"key": "generate", "label": "Generating recipe (Groq)", "status": "pending"},
        ],
        "logs": [],
        "finished": False,
        "error": None,
        "result": None,
        "all_ingredient_names": [],
    }


def update_step(job_id, key, status):
    for step in PROGRESS[job_id]["steps"]:
        if step["key"] == key:
            step["status"] = status
    add_log(job_id, key + " -> " + status)


def add_log(job_id, message):
    PROGRESS[job_id]["logs"].append(message)


def required_confidence(name):
    if name in HIGH_RISK_CLASSES:
        return HIGH_RISK_CONFIDENCE_THRESHOLD
    return BASE_CONFIDENCE_THRESHOLD


def small_gray(frame):
    small = cv2.resize(frame, MOTION_DIFF_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def select_keyframes(job_id, video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    add_log(job_id, "Total frames in video: " + str(total_frames))

    keyframes = []
    prev_gray = None
    frame_count = 0
    last_selected = -FALLBACK_INTERVAL

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = small_gray(frame)
        should_process = False

        if prev_gray is None:
            should_process = True
        else:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(np.mean(diff))
            if score >= MOTION_THRESHOLD:
                should_process = True
            elif frame_count - last_selected >= FALLBACK_INTERVAL:
                should_process = True

        if should_process:
            keyframes.append((frame_count, frame.copy()))
            last_selected = frame_count

        prev_gray = gray
        frame_count += 1

    cap.release()
    add_log(job_id, "Selected " + str(len(keyframes)) + " keyframes out of " + str(total_frames) + " total")
    return keyframes, total_frames


def run_batched_yolo(job_id, keyframes, total_frames):
    hits = defaultdict(list)
    frame_lookup = {}
    frame_detections = defaultdict(list)

    for batch_start in range(0, len(keyframes), BATCH_SIZE):
        batch = keyframes[batch_start:batch_start + BATCH_SIZE]
        batch_indices = [k[0] for k in batch]
        batch_frames = [k[1] for k in batch]

        results = yolo_model(batch_frames, imgsz=INFERENCE_SIZE, verbose=False)

        for frame_idx, frame, result in zip(batch_indices, batch_frames, results):
            frame_had_detection = False
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                name = yolo_model.names[class_id]

                if name not in FOOD_CLASSES:
                    continue
                if confidence < required_confidence(name):
                    continue

                hits[name].append({"confidence": confidence, "frame_idx": frame_idx, "box": box.xyxy[0]})
                frame_detections[frame_idx].append({"name": name, "confidence": confidence, "box": box.xyxy[0]})
                frame_had_detection = True

            if frame_had_detection:
                frame_lookup[frame_idx] = frame

        add_log(job_id, "Processed keyframe batch " + str(batch_start + len(batch)) + " / " + str(len(keyframes)))

    return hits, frame_lookup, frame_detections


def crop_thumbnail(frame, box, padding_ratio=0.25):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    box_w, box_h = x2 - x1, y2 - y1
    pad_x, pad_y = int(box_w * padding_ratio), int(box_h * padding_ratio)

    cx1, cy1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
    cx2, cy2 = min(x2 + pad_x, w), min(y2 + pad_y, h)

    crop = frame[cy1:cy2, cx1:cx2].copy()
    local_x1, local_y1 = x1 - cx1, y1 - cy1
    local_x2, local_y2 = local_x1 + box_w, local_y1 + box_h

    cv2.rectangle(crop, (local_x1, local_y1), (local_x2, local_y2), (0, 255, 0), 2)
    return crop


def build_multi_object_overview(job_id, confirmed_names, frame_lookup, frame_detections, color_map):
    best_frame_idx, best_count, best_names_in_frame = None, -1, []

    for frame_idx, detections in frame_detections.items():
        names_here = set(d["name"] for d in detections if d["name"] in confirmed_names)
        if len(names_here) > best_count:
            best_count, best_frame_idx, best_names_in_frame = len(names_here), frame_idx, names_here

    if best_frame_idx is None or best_frame_idx not in frame_lookup:
        return None, 0

    frame = frame_lookup[best_frame_idx].copy()

    for d in frame_detections[best_frame_idx]:
        if d["name"] not in confirmed_names:
            continue
        x1, y1, x2, y2 = map(int, d["box"])
        color = color_map.get(d["name"], (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = d["name"] + " " + str(round(d["confidence"], 2))
        text_y = max(y1 - 10, 20)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, text_y - text_h - 6), (x1 + text_w + 6, text_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    overview_filename = job_id + "_overview.jpg"
    overview_path = os.path.join(DETECTIONS_FOLDER, overview_filename)
    cv2.imwrite(overview_path, frame)

    add_log(job_id, "Multi-object overview frame: " + str(best_count) + " ingredients together")
    return "/" + overview_path.replace(os.sep, "/"), best_count


def build_detections(job_id, hits, frame_lookup):
    detected_items = {}
    for name, entries in hits.items():
        if len(entries) < MIN_FRAME_HITS:
            add_log(job_id, "Rejected " + name + " (seen in only " + str(len(entries)) + " keyframe(s))")
            continue

        best = max(entries, key=lambda e: e["confidence"])
        frame = frame_lookup.get(best["frame_idx"])
        if frame is None:
            continue

        thumbnail = crop_thumbnail(frame, best["box"])
        image_filename = job_id + "_" + name.replace(" ", "_") + ".jpg"
        image_path = os.path.join(DETECTIONS_FOLDER, image_filename)
        cv2.imwrite(image_path, thumbnail)

        detected_items[name] = {
            "confidence": best["confidence"],
            "image": "/" + image_path.replace(os.sep, "/"),
            "hit_count": len(entries)
        }
        add_log(job_id, "Confirmed " + name + " (" + str(round(best["confidence"], 2)) + ", " + str(len(entries)) + " frames)")

    return detected_items


def detect_ingredients(job_id, video_path):
    keyframes, total_frames = select_keyframes(job_id, video_path)
    hits, frame_lookup, frame_detections = run_batched_yolo(job_id, keyframes, total_frames)

    detected_items = build_detections(job_id, hits, frame_lookup)
    confirmed_names = sorted(detected_items.keys())

    color_map = {}
    color_map_hex = {}
    for i, name in enumerate(confirmed_names):
        color_map[name] = BOX_COLORS_BGR[i % len(BOX_COLORS_BGR)]
        color_map_hex[name] = BOX_COLORS_HEX[i % len(BOX_COLORS_HEX)]

    for name in detected_items:
        detected_items[name]["color"] = color_map_hex[name]

    overview_image, overview_count = build_multi_object_overview(
        job_id, set(confirmed_names), frame_lookup, frame_detections, color_map
    )

    return detected_items, overview_image, overview_count


def embed_query(ingredient_names):
    query = "A recipe using these ingredients: " + ", ".join(ingredient_names)
    return embedder.encode(query).tolist()


def keyword_overlap_score(ingredient_names, recipe_text):
    text_lower = recipe_text.lower()
    hits = sum(1 for name in ingredient_names if name.lower() in text_lower)
    return hits / max(len(ingredient_names), 1)


def retrieve_recipes(job_id, query_vector, ingredient_names, top_k=8, final_k=3):
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    matches = results["matches"]

    ranked = []
    for m in matches:
        overlap = keyword_overlap_score(ingredient_names, m["metadata"]["recipe_text"])
        combined_score = (0.4 * m["score"]) + (0.6 * overlap)
        ranked.append((combined_score, overlap, m))

    ranked.sort(key=lambda x: x[0], reverse=True)

    for combined_score, overlap, m in ranked:
        add_log(job_id, "Candidate: " + m["metadata"]["title"] + " (vector=" + str(round(m["score"], 2)) + ", overlap=" + str(round(overlap, 2)) + ")")

    return [m for _, _, m in ranked[:final_k]]


def build_prompt(ingredient_names, retrieved_recipes):
    context = "\n\n".join([m["metadata"]["recipe_text"] for m in retrieved_recipes])
    ingredients_str = ", ".join(ingredient_names) if ingredient_names else "none confidently detected"

    prompt = (
        "You are an AI cooking assistant.\n\n"
        "Detected food ingredients from video (computer vision, confirmed across multiple frames):\n"
        + ingredients_str + "\n\n"
        "Retrieved reference recipes (ranked by both semantic similarity and ingredient overlap):\n"
        + context + "\n\n"
        "Strict rules:\n"
        "1. The final recipe's MAIN ingredients must be drawn ONLY from the detected list above. "
        "Do not introduce other main ingredients (proteins, vegetables, grains) that were not detected.\n"
        "2. You may add basic pantry staples only: salt, pepper, oil, butter, water, sugar.\n"
        "3. Use the retrieved recipes only for technique and flavor inspiration, not for their ingredient lists.\n"
        "4. If the detected ingredients genuinely can't form a coherent dish, say so honestly and suggest "
        "the simplest reasonable preparation using only what's available.\n"
        "5. Format as: Ingredients list (marking detected vs pantry), then numbered cooking steps.\n"
        "6. On the VERY LAST line, output exactly one line in this format (no extra text after it):\n"
        "MISSING_INGREDIENTS: item1, item2  (or MISSING_INGREDIENTS: none if the dish is complete as-is)\n"
        "List 1-3 ingredients that would meaningfully improve or complete the dish if the person had them.\n"
    )
    return prompt


def generate_recipe(ingredient_names, retrieved_recipes):
    prompt = build_prompt(ingredient_names, retrieved_recipes)

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.choices[0].message.content

    missing = []
    match = re.search(r"MISSING_INGREDIENTS:\s*(.*)", raw_text, re.IGNORECASE)
    if match:
        missing_str = match.group(1).strip()
        raw_text = raw_text[:match.start()].rstrip()
        if missing_str.lower() != "none":
            missing = [m.strip() for m in missing_str.split(",") if m.strip()]

    return raw_text, missing, prompt


def build_result_payload(detections, overview_image, overview_count, recipes, final_recipe, missing, prompt):
    return {
        "detections": [
            {
                "name": name, "confidence": data["confidence"], "image": data["image"],
                "hit_count": data["hit_count"], "color": data.get("color", "#4ade80")
            }
            for name, data in detections.items()
        ],
        "overview_image": overview_image,
        "overview_count": overview_count,
        "recipes": [
            {"title": m["metadata"]["title"], "score": m["score"], "text": m["metadata"]["recipe_text"]}
            for m in recipes
        ],
        "final_recipe": final_recipe,
        "missing_ingredients": missing,
        "prompt": prompt
    }


def finish_job(job_id, detections, overview_image, overview_count, recipes, final_recipe, missing, prompt):
    PROGRESS[job_id]["result"] = build_result_payload(
        detections, overview_image, overview_count, recipes, final_recipe, missing, prompt
    )
    PROGRESS[job_id]["detections_raw"] = detections
    PROGRESS[job_id]["overview_image"] = overview_image
    PROGRESS[job_id]["overview_count"] = overview_count
    PROGRESS[job_id]["all_ingredient_names"] = list(detections.keys())
    PROGRESS[job_id]["finished"] = True


def process_video(job_id, video_path):
    try:
        update_step(job_id, "detect", "active")
        detections, overview_image, overview_count = detect_ingredients(job_id, video_path)
        update_step(job_id, "detect", "done")

        ingredient_names = list(detections.keys())

        if not ingredient_names:
            update_step(job_id, "embed", "done")
            update_step(job_id, "retrieve", "done")
            update_step(job_id, "generate", "done")
            finish_job(
                job_id, {}, None, 0, [],
                "No food-related objects were confidently detected. Try a clearer video with items closer to the camera.",
                [], ""
            )
            return

        update_step(job_id, "embed", "active")
        query_vector = embed_query(ingredient_names)
        update_step(job_id, "embed", "done")

        update_step(job_id, "retrieve", "active")
        retrieved = retrieve_recipes(job_id, query_vector, ingredient_names)
        update_step(job_id, "retrieve", "done")

        update_step(job_id, "generate", "active")
        final_recipe, missing, prompt = generate_recipe(ingredient_names, retrieved)
        update_step(job_id, "generate", "done")

        finish_job(job_id, detections, overview_image, overview_count, retrieved, final_recipe, missing, prompt)

    except Exception as e:
        PROGRESS[job_id]["error"] = str(e)
        PROGRESS[job_id]["finished"] = True

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files or request.files["video"].filename == "":
        return render_template("index.html", error="Please choose a video file.")

    file = request.files["video"]
    job_id = uuid.uuid4().hex

    filename = job_id + "_" + file.filename
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    init_job(job_id)

    thread = threading.Thread(target=process_video, args=(job_id, video_path))
    thread.start()

    return render_template("processing.html", job_id=job_id)


@app.route("/progress/<job_id>")
def progress(job_id):
    job = PROGRESS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/regenerate/<job_id>", methods=["POST"])
def regenerate(job_id):
    job = PROGRESS.get(job_id)
    if job is None or not job.get("finished") or "detections_raw" not in job:
        return jsonify({"error": "Job not ready for regeneration"}), 400

    payload = request.get_json(force=True) or {}
    excluded = set(payload.get("excluded", []))

    all_detections = job["detections_raw"]
    active_names = [n for n in all_detections.keys() if n not in excluded]

    if not active_names:
        return jsonify({"error": "At least one ingredient must remain selected"}), 400

    query_vector = embed_query(active_names)
    retrieved = retrieve_recipes(job_id, query_vector, active_names)
    final_recipe, missing, prompt = generate_recipe(active_names, retrieved)

    active_detections = {n: all_detections[n] for n in active_names}

    result = build_result_payload(
        active_detections, job.get("overview_image"), job.get("overview_count", 0),
        retrieved, final_recipe, missing, prompt
    )

    job["result"] = result
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
