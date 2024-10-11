import os
import re
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

"""
Hyperparam for Ground and Tracking
"""
def calculate_iou(bbx1, bbx2):
    """计算两个边界框的IOU（Intersection over Union）。"""
    x1 = max(bbx1[0], bbx2[0])
    y1 = max(bbx1[1], bbx2[1])
    x2 = min(bbx1[2], bbx2[2])
    y2 = min(bbx1[3], bbx2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbx1[2] - bbx1[0]) * (bbx1[3] - bbx1[1])
    area2 = (bbx2[2] - bbx2[0]) * (bbx2[3] - bbx2[1])
    
    # 计算相交面积对两个边界框面积的比例
    ratio1 = intersection / area1 if area1 > 0 else 0
    ratio2 = intersection / area2 if area2 > 0 else 0
    
    return ratio1, ratio2

def filter_bboxes(bbx, labels, confidences, iou_threshold=0.8):
    """过滤边界框，删除满足条件的边界框。"""
    # 根据confidence从高到低排序
    indices = np.argsort(confidences)[::-1]
    bbx = bbx[indices]
    labels = np.array(labels)[indices]
    confidences = np.array(confidences)[indices]

    to_delete = set()
    for i in range(len(bbx)):
        if i in to_delete:
            continue
        
        current_bbx = bbx[i]
        current_area = (current_bbx[2] - current_bbx[0]) * (current_bbx[3] - current_bbx[1])
        
        # 找到与当前bbx的IOU超过阈值的所有bbx
        contained_bbx = []
        for j in range(len(bbx)):
            if i != j and j not in to_delete:
                ratio1, ratio2 = calculate_iou(current_bbx, bbx[j])
                if ratio1 > iou_threshold or ratio2 > iou_threshold:
                    contained_bbx.append(bbx[j])

        # 如果包含两个以上的bbx
        if len(contained_bbx) >= 2:
            # 计算平均面积
            avg_area = np.mean([(bb[2] - bb[0]) * (bb[3] - bb[1]) for bb in contained_bbx])
            large_bbx_found = any((bb[2] - bb[0]) * (bb[3] - bb[1]) > 2 * avg_area for bb in contained_bbx)

            if large_bbx_found:
                to_delete.add(i)  # 标记当前bbx删除

    # 过滤掉要删除的边界框
    filtered_bbx = np.array([bbx[i] for i in range(len(bbx)) if i not in to_delete])
    filtered_labels = [labels[i] for i in range(len(labels)) if i not in to_delete]
    filtered_confidences = [confidences[i] for i in range(len(confidences)) if i not in to_delete]

    return filtered_bbx, filtered_labels, filtered_confidences

def filte_big_box(boxes, labels, confidences,total_area) : 
    assert boxes.shape[1] == 4 
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) 
    mask =( areas  / total_area )  < 0.5
    filtered_boxes = boxes[mask]
    filtered_labels = [label for i, label in enumerate(labels) if mask[i]]
    filtered_confid = [conf for i, conf in enumerate(confidences) if mask[i]]
    filtered_boxes, filtered_labels, filtered_confid = filter_bboxes(filtered_boxes, filtered_labels, filtered_confid)

    return filtered_boxes, filtered_labels, filtered_confid


def save_mask(masks, path="mask_object"):
        # 遍历每个物体的掩码并保存为 PNG 文件
        for i, mask in enumerate(masks):
            # 将掩码转换为图片格式 (0, 255)
            img = Image.fromarray((mask * 255).astype(np.uint8))
            
            # 保存为 PNG 文件
            img.save(f"{path}_{i}.png")

class VideoProcessor:
    def __init__(self, save_mask=False, re_split=False) -> None:
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.video_path = "test.mp4"
        self.text_prompt = "object."
        self.output_video_path = "./object.mp4"
        self.source_video_frame_dir = "./object_frame_n/"
        self.save_tracking_results_dir = "./object_result_n/"
        self.prompt_type_for_video = "box" # choose from ["point", "box", "mask"]

        """
        Step 1: Environment settings and model initialization for SAM 2
        """
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # init sam image predictor and video predictor model
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # build grounding dino from huggingface
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.save_mask = save_mask
        self.re_split = re_split

    def split_video(self):
        """
        Custom video input directly using video files
        """
        video_info = sv.VideoInfo.from_video_path(self.video_path)  # get video info
        print(video_info)
        frame_generator = sv.get_video_frames_generator(self.video_path, stride=1, start=0, end=None)

        # saving video to frames
        source_frames = Path(self.source_video_frame_dir)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="saving video frames"):
                sink.save_image(frame)

    def check_if_need_split(self):
        if self.re_split:
            self.split_video()
        else:
            if not os.path.exists(self.source_video_frame_dir):
                self.split_video()
            elif os.path.exists(self.source_video_frame_dir) and len(os.listdir(self.source_video_frame_dir)) == 0:
                self.split_video()


    def load_frame_name(self):
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(self.source_video_frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        if len(frame_names) ==  0:
            frame_names = [
                p for p in os.listdir(self.source_video_frame_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        # init video predictor state

        self.frame_names = frame_names
        self.inference_state = self.video_predictor.init_state(video_path=self.source_video_frame_dir)

        return frame_names
    def set_ref_idx(self, idx=0):
        self.ann_frame_idx = idx  # the frame index we interact with

    def set_better_ref(self):
        idx = 0
        self.set_ref_idx(idx)
    # prompt grounding dino to get the box coordinates on specific frame
    def gdino_process(self):
        assert hasattr(self, 'ann_frame_idx')
        img_path = os.path.join(self.source_video_frame_dir, self.frame_names[self.ann_frame_idx])
        image = Image.open(img_path)
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.20,
            target_sizes=[image.size[::-1]]
        )

        input_boxes = results[0]["boxes"].cpu().numpy()

        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        converted_image = np.array(image.convert("RGB"))
        h, w = converted_image.shape[:2]
        print('before processed', input_boxes.shape)
        self.input_boxes, self.class_names, self.confidences = filte_big_box(input_boxes, class_names, confidences, w * h)
        print(input_boxes.shape)
        self.objects = self.class_names
        self.converted_image = converted_image
    def sam2_image_process(self):

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(self.converted_image)

        # process the detection results

        print(self.objects)
        # prompt SAM 2 image predictor to get the mask for the object
        self.image_masks, self.image_scores, self.image_logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=self.input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if self.image_masks.ndim == 4:
            self.image_masks = self.image_masks.squeeze(1)
        if self.save_mask:
            save_mask(self.image_masks)

    def prepare_sam2_video(self):

        """
        Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        """

        assert self.prompt_type_for_video in ["point", "box", "mask"], "sam 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if self.prompt_type_for_video == "point":
            # sample the positive points from mask for each self.objects
            all_sample_points = sample_points_from_masks(masks=self.image_masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(self.objects, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif self.prompt_type_for_video == "box":
            for object_id, (label, box) in enumerate(zip(self.objects, self.input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif self.prompt_type_for_video == "mask":
            for object_id, (label, mask) in enumerate(zip(self.objects, self.image_masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

        return self.objects

    def propagate_in_video(self):
        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        # prev_seg = 
        self.video_segments = {}  # self.video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state,reverse=False):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    def vis_and_save(self):
        """
        Step 5: Visualize the segment results across the video and save them
        """
        if not os.path.exists(self.save_tracking_results_dir):
            os.makedirs(self.save_tracking_results_dir)

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(self.objects, start=1)}

        for frame_idx, segments in self.video_segments.items():
            img = cv2.imread(os.path.join(self.source_video_frame_dir, self.frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(self.save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


        """
        Step 6: Convert the annotated frames to video
        """

        create_video_from_images(self.save_tracking_results_dir, self.output_video_path)
    def update_files(self, video_path, output_video_path='processed_video.mp4', text_prompt='object.', source_video_frame_dir='./tmp/source_video_frame', save_tracking_results_dir='./tmp/save_tracking_results'):
        self.video_path = video_path
        self.text_prompt = text_prompt
        self.output_video_path = output_video_path
        self.source_video_frame_dir = source_video_frame_dir
        self.save_tracking_results_dir = save_tracking_results_dir
        print('updated video path is ' , self.video_path)
    
    def update_and_process(self, video_path, output_video_path='processed_video.mp4', text_prompt='object.', source_video_frame_dir='./tmp/source_video_frame', save_tracking_results_dir='./tmp/save_tracking_results'):
        self.update_files(video_path, output_video_path,text_prompt, source_video_frame_dir, save_tracking_results_dir)
        
        self.check_if_need_split()
        self.load_frame_name()
        
        self.set_better_ref()

        self.gdino_process()

        self.sam2_image_process()

        self.prepare_sam2_video()

        self.propagate_in_video()

        self.vis_and_save()

if __name__=='__main__':
    process_model = VideoProcessor(re_split=True)
    process_model.update_and_process('./test.mp4')
