import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import json


class PairedSequence(Sequence):
    """Sequence subclass that carries both IR and RGB frame paths with their
    respective ground-truth annotations (which live in different coordinate
    spaces due to the unaligned nature of the Anti-UAV RGB-T dataset)."""

    def __init__(self, name, ir_frames, rgb_frames, ir_gt_rect, rgb_gt_rect, dataset, **kwargs):
        super().__init__(name, ir_frames, dataset, ir_gt_rect, **kwargs)
        self.ir_frames = self.frames
        self.ir_ground_truth_rect = self.ground_truth_rect
        self.rgb_frames = rgb_frames
        self.rgb_ground_truth_rect = rgb_gt_rect


class AntiUAVRGBTDataset(BaseDataset):
    """Anti-UAV RGB-Thermal paired dataset.

    Loads both IR (ir_frames/) and RGB (rgb_frames/) frame paths for every
    sequence, along with their respective ground-truth annotations.
    IR annotations are read from IR_label.json (IR coordinate space).
    RGB annotations are read from visible_label.json (RGB coordinate space).
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.antiuav_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_dir = os.path.join(self.base_path, sequence_name)

        # IR ground truth (bounding boxes in IR coordinate space)
        ir_anno_path = os.path.join(seq_dir, 'IR_label.json')
        with open(ir_anno_path, 'r') as f:
            ir_label = json.load(f)
        ir_gt = ir_label['gt_rect']
        for i in range(len(ir_gt)):
            if ir_gt[i] in ([], [0]):
                ir_gt[i] = [0, 0, 0, 0]
        ir_gt_rect = np.array(ir_gt).reshape(-1, 4)

        # RGB ground truth (bounding boxes in RGB coordinate space)
        rgb_anno_path = os.path.join(seq_dir, 'visible_label.json')
        with open(rgb_anno_path, 'r') as f:
            rgb_label = json.load(f)
        rgb_gt = rgb_label['gt_rect']
        for i in range(len(rgb_gt)):
            if rgb_gt[i] in ([], [0]):
                rgb_gt[i] = [0, 0, 0, 0]
        rgb_gt_rect = np.array(rgb_gt).reshape(-1, 4)

        # IR frame paths
        ir_frames_dir = os.path.join(seq_dir, 'ir_frames')
        ir_frame_list = sorted(
            [f for f in os.listdir(ir_frames_dir) if f.endswith('.jpg')],
            key=lambda f: int(f[:-4])
        )
        ir_frames = [os.path.join(ir_frames_dir, f) for f in ir_frame_list]

        # RGB frame paths
        rgb_frames_dir = os.path.join(seq_dir, 'rgb_frames')
        rgb_frame_list = sorted(
            [f for f in os.listdir(rgb_frames_dir) if f.endswith('.jpg')],
            key=lambda f: int(f[:-4])
        )
        rgb_frames = [os.path.join(rgb_frames_dir, f) for f in rgb_frame_list]

        return PairedSequence(
            sequence_name, ir_frames, rgb_frames,
            ir_gt_rect, rgb_gt_rect, 'antiuavrgbt'
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open(os.path.join(self.base_path, 'list.txt')) as f:
            return f.read().splitlines()
