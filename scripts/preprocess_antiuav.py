"""
One-time preprocessing: adapts the Anti-UAV RGBT test split to the format
expected by EDTC's antiuavdataset.py.

What it does per sequence:
  1. Extracts frames from infrared.mp4  -> ir_frames/000001.jpg, ...
     and copies infrared.json           -> IR_label.json
  2. Extracts frames from visible.mp4   -> rgb_frames/000001.jpg, ...
     and copies visible.json            -> visible_label.json
  3. Writes test/list.txt with one sequence name per line

Note: antiuavdataset.py has been updated to read from ir_frames/ instead of
the sequence root directory.

Run locally on your laptop before syncing to Snellius:
  python scripts/preprocess_antiuav.py

Then sync to Snellius:
  rsync -av --progress \\
    "C:/Users/shara/Documents/MasterIS/thesis/Anti-UAV-RGBT/test/" \\
    ssuttorp@snellius.surf.nl:/home/ssuttorp/Anti-UAV-RGBT/test/
"""
import os
import shutil
import cv2

DATASET_TEST = 'C:/Users/shara/Documents/MasterIS/thesis/Anti-UAV-RGBT/test'

sequences = sorted([
    d for d in os.listdir(DATASET_TEST)
    if os.path.isdir(os.path.join(DATASET_TEST, d))
])

print(f'Found {len(sequences)} sequences in test split.\n')

for i, seq in enumerate(sequences, 1):
    seq_dir = os.path.join(DATASET_TEST, seq)

    # ------------------------------------------------------------------
    # 1. IR frames: extract into seq_dir/ir_frames/
    # ------------------------------------------------------------------
    ir_video     = os.path.join(seq_dir, 'infrared.mp4')
    ir_out_dir   = os.path.join(seq_dir, 'ir_frames')
    ir_label_src = os.path.join(seq_dir, 'infrared.json')
    ir_label_dst = os.path.join(seq_dir, 'IR_label.json')

    first_ir_frame = os.path.join(ir_out_dir, '000001.jpg')
    loose_ir_frame = os.path.join(seq_dir, '000001.jpg')
    if os.path.exists(first_ir_frame):
        print(f'[{i:03d}/{len(sequences)}] {seq} — IR: frames already in ir_frames/, skipping')
    elif os.path.exists(loose_ir_frame):
        # Frames were extracted by the old script into the sequence root — move them
        os.makedirs(ir_out_dir, exist_ok=True)
        loose_frames = sorted(f for f in os.listdir(seq_dir) if f.endswith('.jpg'))
        for fname in loose_frames:
            shutil.move(os.path.join(seq_dir, fname), os.path.join(ir_out_dir, fname))
        print(f'[{i:03d}/{len(sequences)}] {seq} — IR: moved {len(loose_frames)} loose frames -> ir_frames/')
    elif not os.path.exists(ir_video):
        print(f'[{i:03d}/{len(sequences)}] WARNING: {seq} — infrared.mp4 not found, skipping IR')
    else:
        os.makedirs(ir_out_dir, exist_ok=True)
        cap = cv2.VideoCapture(ir_video)
        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(ir_out_dir, f'{frame_idx:06d}.jpg'), frame)
            frame_idx += 1
        cap.release()
        print(f'[{i:03d}/{len(sequences)}] {seq} — IR: extracted {frame_idx - 1} frames')

    if os.path.exists(ir_label_src) and not os.path.exists(ir_label_dst):
        shutil.copy2(ir_label_src, ir_label_dst)

    # ------------------------------------------------------------------
    # 2. RGB frames: extract into seq_dir/rgb_frames/
    # ------------------------------------------------------------------
    rgb_video     = os.path.join(seq_dir, 'visible.mp4')
    rgb_out_dir   = os.path.join(seq_dir, 'rgb_frames')
    rgb_label_src = os.path.join(seq_dir, 'visible.json')
    rgb_label_dst = os.path.join(seq_dir, 'visible_label.json')

    first_rgb_frame = os.path.join(rgb_out_dir, '000001.jpg')
    if not os.path.exists(first_rgb_frame):
        if not os.path.exists(rgb_video):
            print(f'[{i:03d}/{len(sequences)}] WARNING: {seq} — visible.mp4 not found, skipping RGB')
        else:
            os.makedirs(rgb_out_dir, exist_ok=True)
            cap = cv2.VideoCapture(rgb_video)
            frame_idx = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(rgb_out_dir, f'{frame_idx:06d}.jpg'), frame)
                frame_idx += 1
            cap.release()
            print(f'[{i:03d}/{len(sequences)}] {seq} — RGB: extracted {frame_idx - 1} frames')
    else:
        print(f'[{i:03d}/{len(sequences)}] {seq} — RGB: frames already exist, skipping')

    if os.path.exists(rgb_label_src) and not os.path.exists(rgb_label_dst):
        shutil.copy2(rgb_label_src, rgb_label_dst)

# ------------------------------------------------------------------
# 3. Write list.txt
# ------------------------------------------------------------------
list_path = os.path.join(DATASET_TEST, 'list.txt')
with open(list_path, 'w') as f:
    f.write('\n'.join(sequences))

print(f'\nDone. Written {list_path} with {len(sequences)} sequences.')
print('Next step: rsync the test/ folder to Snellius.')
