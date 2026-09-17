import cv2
import csv
import json
import re
import time
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

from ultralytics import YOLO
import easyocr


# ============================================================
# LICENSE VISION AI
# VIDEO ALPR ENGINE
#
# YOLO + EASYOCR + TEMPORAL OCR CONSENSUS
# ============================================================

ROOT = Path(__file__).resolve().parent

VIDEO_PATH = ROOT / "test_video.mp4"

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "models"
    / "license_plate_detector"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = ROOT / "video_results"

OUTPUT_VIDEO = OUTPUT_DIR / "annotated_video_consensus.mp4"
OUTPUT_CSV = OUTPUT_DIR / "detections_consensus.csv"
OUTPUT_JSON = OUTPUT_DIR / "video_summary_consensus.json"


# ============================================================
# SETTINGS
# ============================================================

FRAME_SKIP = 5
PROCESS_WIDTH = 1280

YOLO_CONF = 0.35
OCR_CONF = 0.20

# How similar two OCR readings must be
SIMILARITY_THRESHOLD = 0.65

# Number of observations retained for a tracked plate
HISTORY_SIZE = 30


# ============================================================
# OCR CLEANING
# ============================================================

def clean_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# CHARACTER NORMALIZATION
# ============================================================

def normalize_ocr(text):

    text = clean_text(text)

    if not text:
        return ""

    # Common OCR confusions.
    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
    }

    return "".join(
        replacements.get(char, char)
        for char in text
    )


# ============================================================
# STRING SIMILARITY
# ============================================================

def similarity(a, b):

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# PLATE FORMAT SCORE
# ============================================================

def plate_format_score(text):

    if not text:
        return 0

    score = 0

    length = len(text)

    # Indian plates commonly fall around this range.
    if 8 <= length <= 10:
        score += 40

    elif 6 <= length <= 12:
        score += 20

    # Letters
    if re.search(r"[A-Z]", text):
        score += 20

    # Numbers
    if re.search(r"[0-9]", text):
        score += 20

    # State-code-like beginning.
    if re.match(r"^[A-Z]{2}", text):
        score += 10

    # At least two digits.
    if re.search(r"[0-9]{2,}", text):
        score += 10

    return min(score, 100)


# ============================================================
# RESIZE
# ============================================================

def resize_frame(frame):

    height, width = frame.shape[:2]

    if width <= PROCESS_WIDTH:

        return frame, 1.0

    scale = PROCESS_WIDTH / width

    new_width = PROCESS_WIDTH
    new_height = int(
        height * scale
    )

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# OCR
# ============================================================

def run_ocr(reader, crop):

    if crop is None:
        return "", 0.0

    if crop.size == 0:
        return "", 0.0

    try:

        results = reader.readtext(
            crop,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        if not results:
            return "", 0.0

        best_text = ""
        best_conf = 0.0

        for result in results:

            if len(result) < 3:
                continue

            text = clean_text(
                result[1]
            )

            confidence = float(
                result[2]
            )

            if not text:
                continue

            if confidence > best_conf:

                best_text = text
                best_conf = confidence

        return best_text, best_conf

    except Exception:

        return "", 0.0


# ============================================================
# FIND EXISTING PLATE CLUSTER
# ============================================================

def find_matching_plate(
    text,
    clusters
):

    normalized = normalize_ocr(
        text
    )

    best_match = None
    best_similarity = 0

    for cluster_id, cluster in clusters.items():

        representative = cluster[
            "representative"
        ]

        score = similarity(
            normalized,
            normalize_ocr(
                representative
            )
        )

        if score > best_similarity:

            best_similarity = score
            best_match = cluster_id

    if best_similarity >= SIMILARITY_THRESHOLD:

        return best_match

    return None


# ============================================================
# UPDATE OCR CLUSTER
# ============================================================

def update_cluster(
    cluster,
    text,
    confidence,
    frame_number
):

    cluster["observations"].append(
        {
            "text": text,
            "confidence": confidence,
            "frame": frame_number
        }
    )

    if len(
        cluster["observations"]
    ) > HISTORY_SIZE:

        cluster["observations"] = (
            cluster["observations"][-HISTORY_SIZE:]
        )

    # --------------------------------------------------------
    # SELECT BEST REPRESENTATION
    # --------------------------------------------------------

    candidates = cluster[
        "observations"
    ]

    best = max(
        candidates,
        key=lambda item:
        item["confidence"]
    )

    cluster[
        "best_confidence"
    ] = best["confidence"]

    cluster[
        "last_frame"
    ] = frame_number

    # --------------------------------------------------------
    # VOTING
    # --------------------------------------------------------

    votes = defaultdict(float)

    for observation in candidates:

        votes[
            observation["text"]
        ] += observation[
            "confidence"
        ]

    winner = max(
        votes,
        key=votes.get
    )

    cluster[
        "representative"
    ] = winner

    # --------------------------------------------------------
    # FORMAT SCORE
    # --------------------------------------------------------

    cluster[
        "format_score"
    ] = plate_format_score(
        winner
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    cluster[
        "final_confidence"
    ] = (
        cluster["best_confidence"]
        * 0.70
        +
        cluster["format_score"]
        / 100
        * 0.30
    )


# ============================================================
# DRAW
# ============================================================

def draw_plate(
    frame,
    x1,
    y1,
    x2,
    y2,
    text,
    confidence
):

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    label = (
        f"{text} | "
        f"{confidence * 100:.1f}%"
        if text
        else "PLATE"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.75
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(
        label,
        font,
        scale,
        thickness
    )

    label_y = max(
        y1 - 10,
        th + 10
    )

    cv2.rectangle(
        frame,
        (
            x1,
            label_y - th - 10
        ),
        (
            x1 + tw + 10,
            label_y + baseline
        ),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 5,
            label_y - 5
        ),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "LICENSE VISION AI — TEMPORAL OCR ENGINE"
    )
    print(
        "YOLO + EASYOCR + OCR CONSENSUS"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():

        print(
            f"\nERROR: Video not found:\n{VIDEO_PATH}"
        )

        return

    if not MODEL_PATH.exists():

        print(
            f"\nERROR: Model not found:\n{MODEL_PATH}"
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        total_frames / fps
        if fps
        else 0
    )

    print()
    print("VIDEO")
    print("-" * 78)
    print(
        f"Resolution       : {width} x {height}"
    )
    print(
        f"FPS              : {fps:.2f}"
    )
    print(
        f"Total frames     : {total_frames}"
    )
    print(
        f"Duration         : {duration:.2f} sec"
    )

    print()
    print("OPTIMIZATION")
    print("-" * 78)
    print(
        f"Frame sampling   : 1/{FRAME_SKIP}"
    )
    print(
        f"AI width         : {PROCESS_WIDTH}px"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("INITIALIZING YOLO")
    print("=" * 78)

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("INITIALIZING EASYOCR")
    print("=" * 78)

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )

    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # PLATE CLUSTERS
    # --------------------------------------------------------

    clusters = {}

    next_cluster_id = 1

    detection_records = []

    frame_number = 0
    processed_frames = 0
    total_detections = 0
    ocr_success = 0

    start_time = time.time()

    # ========================================================
    # PROCESS
    # ========================================================

    print()
    print("=" * 78)
    print("PROCESSING")
    print("=" * 78)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ----------------------------------------------------
        # SAMPLE FRAMES
        # ----------------------------------------------------

        if frame_number % FRAME_SKIP != 0:

            writer.write(frame)

            continue

        processed_frames += 1

        display_frame = frame.copy()

        processing_frame, scale = resize_frame(
            frame
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(
            source=processing_frame,
            conf=YOLO_CONF,
            device="cpu",
            verbose=False
        )

        if results:

            boxes = results[0].boxes

        else:

            boxes = None

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        if boxes is not None:

            for box in boxes:

                coords = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                yolo_conf = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )

                rx1, ry1, rx2, ry2 = map(
                    int,
                    coords
                )

                # Back to original resolution.
                x1 = int(
                    rx1 / scale
                )

                y1 = int(
                    ry1 / scale
                )

                x2 = int(
                    rx2 / scale
                )

                y2 = int(
                    ry2 / scale
                )

                x1 = max(
                    0,
                    min(x1, width - 1)
                )

                y1 = max(
                    0,
                    min(y1, height - 1)
                )

                x2 = max(
                    x1 + 1,
                    min(x2, width)
                )

                y2 = max(
                    y1 + 1,
                    min(y2, height)
                )

                crop = frame[
                    y1:y2,
                    x1:x2
                ]

                total_detections += 1

                # ------------------------------------------------
                # OCR
                # ------------------------------------------------

                text, ocr_conf = run_ocr(
                    reader,
                    crop
                )

                cluster = None

                if text and ocr_conf >= OCR_CONF:

                    ocr_success += 1

                    cluster_id = find_matching_plate(
                        text,
                        clusters
                    )

                    # ------------------------------------------------
                    # EXISTING CLUSTER
                    # ------------------------------------------------

                    if cluster_id is not None:

                        cluster = clusters[
                            cluster_id
                        ]

                        update_cluster(
                            cluster,
                            text,
                            ocr_conf,
                            frame_number
                        )

                    # ------------------------------------------------
                    # NEW CLUSTER
                    # ------------------------------------------------

                    else:

                        cluster_id = (
                            f"PLATE_{next_cluster_id:04d}"
                        )

                        next_cluster_id += 1

                        clusters[
                            cluster_id
                        ] = {

                            "id": cluster_id,

                            "representative": text,

                            "observations": [
                                {
                                    "text": text,
                                    "confidence": ocr_conf,
                                    "frame": frame_number
                                }
                            ],

                            "first_frame": frame_number,

                            "last_frame": frame_number,

                            "detections": 1,

                            "best_confidence": ocr_conf,

                            "format_score":
                                plate_format_score(text),

                            "final_confidence":
                                ocr_conf
                        }

                        cluster = clusters[
                            cluster_id
                        ]

                # ------------------------------------------------
                # DRAW BEST RESULT
                # ------------------------------------------------

                if cluster:

                    display_text = cluster[
                        "representative"
                    ]

                    display_confidence = cluster[
                        "final_confidence"
                    ]

                else:

                    display_text = ""

                    display_confidence = yolo_conf

                draw_plate(
                    display_frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    display_text,
                    display_confidence
                )

                detection_records.append(
                    {
                        "frame":
                            frame_number,

                        "timestamp":
                            round(
                                frame_number / fps,
                                3
                            ),

                        "plate":
                            display_text,

                        "yolo_confidence":
                            round(
                                yolo_conf,
                                4
                            ),

                        "ocr_confidence":
                            round(
                                ocr_conf,
                                4
                            ),

                        "final_confidence":
                            round(
                                display_confidence,
                                4
                            ),

                        "bbox":
                            [
                                x1,
                                y1,
                                x2,
                                y2
                            ]
                    }
                )

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        ai_fps = (
            processed_frames / elapsed
            if elapsed > 0
            else 0
        )

        progress = (
            frame_number
            / total_frames
            * 100
            if total_frames
            else 0
        )

        cv2.rectangle(
            display_frame,
            (20, 20),
            (640, 145),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            display_frame,
            "LICENSE VISION AI",
            (40, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"FRAME {frame_number}/{total_frames}",
            (40, 87),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"UNIQUE PLATES {len(clusters)}",
            (40, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"AI FPS {ai_fps:.2f}",
            (400, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        writer.write(
            display_frame
        )

        if processed_frames % 5 == 0:

            print(
                f"\rProcessing: "
                f"{progress:6.2f}% | "
                f"Frame {frame_number}/{total_frames} | "
                f"AI FPS: {ai_fps:.2f} | "
                f"Clusters: {len(clusters)}",
                end="",
                flush=True
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    elapsed = (
        time.time()
        - start_time
    )

    # ========================================================
    # FINAL PLATE REGISTRY
    # ========================================================

    registry = []

    for cluster in clusters.values():

        first_timestamp = (
            cluster["first_frame"] / fps
            if fps
            else 0
        )

        last_timestamp = (
            cluster["last_frame"] / fps
            if fps
            else 0
        )

        registry.append(
            {
                "id":
                    cluster["id"],

                "plate":
                    cluster["representative"],

                "detections":
                    len(cluster["observations"]),

                "first_frame":
                    cluster["first_frame"],

                "last_frame":
                    cluster["last_frame"],

                "first_timestamp":
                    round(
                        first_timestamp,
                        3
                    ),

                "last_timestamp":
                    round(
                        last_timestamp,
                        3
                    ),

                "best_ocr_confidence":
                    round(
                        cluster[
                            "best_confidence"
                        ],
                        4
                    ),

                "format_score":
                    cluster[
                        "format_score"
                    ],

                "final_confidence":
                    round(
                        cluster[
                            "final_confidence"
                        ],
                        4
                    )
            }
        )

    registry.sort(
        key=lambda x:
        x["final_confidence"],
        reverse=True
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fields = [
            "frame",
            "timestamp",
            "plate",
            "yolo_confidence",
            "ocr_confidence",
            "final_confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ]

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        csv_writer.writeheader()

        for row in detection_records:

            x1, y1, x2, y2 = (
                row["bbox"]
            )

            csv_writer.writerow(
                {
                    "frame":
                        row["frame"],

                    "timestamp":
                        row["timestamp"],

                    "plate":
                        row["plate"],

                    "yolo_confidence":
                        row[
                            "yolo_confidence"
                        ],

                    "ocr_confidence":
                        row[
                            "ocr_confidence"
                        ],

                    "final_confidence":
                        row[
                            "final_confidence"
                        ],

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            )

    # ========================================================
    # JSON
    # ========================================================

    summary = {

        "video": {

            "filename":
                VIDEO_PATH.name,

            "width":
                width,

            "height":
                height,

            "fps":
                fps,

            "total_frames":
                total_frames,

            "duration_seconds":
                duration
        },

        "processing": {

            "frame_skip":
                FRAME_SKIP,

            "processing_width":
                PROCESS_WIDTH,

            "processed_frames":
                processed_frames,

            "processing_time_seconds":
                round(
                    elapsed,
                    2
                ),

            "ai_fps":
                round(
                    processed_frames / elapsed
                    if elapsed > 0
                    else 0,
                    2
                )
        },

        "detection": {

            "total_detections":
                total_detections,

            "ocr_success":
                ocr_success,

            "unique_plate_clusters":
                len(clusters)
        },

        "plate_registry":
            registry
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print()
    print("=" * 78)
    print("TEMPORAL OCR PROCESSING COMPLETE")
    print("=" * 78)

    print(
        f"\nFrames processed : {processed_frames}"
    )

    print(
        f"Detections      : {total_detections}"
    )

    print(
        f"OCR successes   : {ocr_success}"
    )

    print(
        f"Plate clusters  : {len(clusters)}"
    )

    print(
        f"Processing time : {elapsed:.2f} sec"
    )

    print()
    print("FINAL PLATE REGISTRY")
    print("-" * 78)

    for plate in registry:

        print(
            f"{plate['plate']:15s} | "
            f"Confidence: "
            f"{plate['final_confidence'] * 100:6.2f}% | "
            f"Observations: "
            f"{plate['detections']:3d}"
        )

    print()
    print("OUTPUT")
    print("-" * 78)

    print(
        f"Video : {OUTPUT_VIDEO}"
    )

    print(
        f"CSV   : {OUTPUT_CSV}"
    )

    print(
        f"JSON  : {OUTPUT_JSON}"
    )

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()