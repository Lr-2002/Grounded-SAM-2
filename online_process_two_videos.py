from video_processor import VideoProcessor
from tqdm import tqdm
import numpy as np
import re
import os
import shutil


class MaskGenerator:
    def __init__(self):
        self.vp = VideoProcessor(
            re_split=False,
            save_video=False,
            save_bbox=False,
            save_mask=False,
            save_bbox_vis=False,
        )

    def infer_two_video(self, gt_video, ge_video, bbox_list=None):
        # video path should only contain videos
        frame_data_1, bbox_list_1 = self.vp.get_detections_with_bbox(
            gt_video, source_video_frame_dir=gt_video, input_boxes=bbox_list
        )
        bbox_list = bbox_list if bbox_list else bbox_list_1
        frame_data_2, bbox_list_2 = self.vp.get_detections_with_bbox(
            ge_video, source_video_frame_dir=ge_video, input_boxes=bbox_list
        )
        frame_data_1, frame_data_2 = (
            self.convert_detect_masks(frame_data_1),
            self.convert_detect_masks(frame_data_2),
        )
        return frame_data_1, frame_data_2

    def convert_detect_masks(self, frame_data):
        return np.array([x.mask.astype(np.float16) for x in frame_data])

    def infer(self, video_path):
        # indeed , the video path is video frames
        frame_data_list = self.vp.get_detections(
            video_path, source_video_frame_dir=video_path
        )


if __name__ == "__main__":
    fi_ge = "/home/lr-2002/code/IRASim/fi_ge"
    fi_gt = "/home/lr-2002/code/IRASim/fi_gt"
    MG = MaskGenerator()
    for video_id in tqdm(os.listdir(fi_gt)):
        gt_video = os.path.join(fi_gt, video_id)
        ge_video = os.path.join(fi_ge, video_id)
        MG.infer_two_video(gt_video=gt_video, ge_video=ge_video)
