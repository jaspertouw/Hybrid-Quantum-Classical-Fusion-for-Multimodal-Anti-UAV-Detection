import json
import os
import cv2

DATASET_ROOT = "C:/Users/shara/Documents/MasterIS/thesis/Anti-UAV-RGBT"
SEQ = "train/20190925_101846_1_1"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

MODALITIES = [
    {
        "name": "ir",
        "video": os.path.join(DATASET_ROOT, SEQ, "infrared.mp4"),
        "label": os.path.join(DATASET_ROOT, SEQ, "infrared.json"),
        "output": os.path.join(OUTPUT_DIR, "vis_gt_ir.png"),
    },
    {
        "name": "rgb",
        "video": os.path.join(DATASET_ROOT, SEQ, "visible.mp4"),
        "label": os.path.join(DATASET_ROOT, SEQ, "visible.json"),
        "output": os.path.join(OUTPUT_DIR, "vis_gt_rgb.png"),
    },
]

BOX_COLOR = (0, 220, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 140, 0)


def find_good_frame(data, start=100):
    for i in range(start, len(data["exist"])):
        if data["exist"][i] == 1:
            rect = data["gt_rect"][i]
            if isinstance(rect, list) and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                return i
    # fallback to any present frame
    for i in range(len(data["exist"])):
        if data["exist"][i] == 1:
            rect = data["gt_rect"][i]
            if isinstance(rect, list) and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
                return i
    return None


def draw_bbox(frame, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
    label = "UAV (GT)"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y - 6, th + 4)
    cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty), TEXT_BG_COLOR, -1)
    cv2.putText(frame, label, (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)


for mod in MODALITIES:
    with open(mod["label"]) as f:
        data = json.load(f)

    frame_idx = find_good_frame(data)
    if frame_idx is None:
        print(f"[{mod['name']}] No valid annotated frame found, skipping.")
        continue

    cap = cv2.VideoCapture(mod["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[{mod['name']}] Could not read frame {frame_idx} from video.")
        continue

    x, y, w, h = [int(v) for v in data["gt_rect"][frame_idx]]
    draw_bbox(frame, x, y, w, h)

    os.makedirs(os.path.dirname(mod["output"]), exist_ok=True)
    cv2.imwrite(mod["output"], frame)
    print(f"[{mod['name']}] Saved frame {frame_idx} -> {mod['output']}  bbox=({x},{y},{w},{h})")
