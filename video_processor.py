import os
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
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
VIDEO_PATH = "test.mp4"
TEXT_PROMPT = "object."
OUTPUT_VIDEO_PATH = "./object.mp4"
SOURCE_VIDEO_FRAME_DIR = "./object_frame_n/"
SAVE_TRACKING_RESULTS_DIR = "./object_result_n/"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]

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

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


"""
Custom video input directly using video files
"""
def save_mask(masks, path="mask_object"):
    # 遍历每个物体的掩码并保存为 PNG 文件
    for i, mask in enumerate(masks):
        # 将掩码转换为图片格式 (0, 255)
        img = Image.fromarray((mask * 255).astype(np.uint8))
        
        # 保存为 PNG 文件
        img.save(f"{path}_{i}.png")

def split_video(video_path, source_video_frame_dir):
    video_info = sv.videoinfo.from_video_path(video_path)  # get video info
    print(video_info)
    frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)

    # saving video to frames
    source_frames = Path(source_video_frame_dir)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.imagesink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="saving video frames"):
            sink.save_image(frame)
def load_frame_name(source_video_frame_dir):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(source_video_frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    if len(frame_names) ==  0:
        split_video(VIDEO_PATH, source_video_frame_dir)
        frame_names = [
            p for p in os.listdir(source_video_frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names
    # init video predictor state
frame_names = load_frame_name(SOURCE_VIDEO_FRAME_DIR)
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)


ann_frame_idx = 15  # the frame index we interact with
"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
def gd_process(source_video_frame_dir, text_prompt='object.'):
    img_path = os.path.join(source_video_frame_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
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
    input_boxes, class_names, confidences = filte_big_box(input_boxes, class_names, confidences, w * h)
    print(input_boxes.shape)
    return converted_image, input_boxes, class_names, confidences
def sam2_image_process(source_video_frame_dir, text_prompt):

    # prompt SAM image predictor to get the mask for the object
    converted_image, input_boxes, class_names, confidences = gd_process(SOURCE_VIDEO_FRAME_DIR, TEXT_PROMPT)
    image_predictor.set_image(converted_image)

    # process the detection results
    OBJECTS = class_names

    print(OBJECTS)

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    save_mask(masks)
    return masks, scores, logits, OBJECTS, input_boxes

def prepare_sam2_video(source_video_frame_dir, text_prompt):
    masks, scores, logits, objects, input_boxes = sam2_image_process(source_video_frame_dir, text_prompt)

    """
    Step 3: Register each object's positive points to video predictor with seperate add_new_points call
    """

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(objects, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(objects, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

    return objects

def propagate_in_video(source_video_frame_dir, text_prompt):
    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    # prev_seg = 
    objects = prepare_sam2_video(source_video_frame_dir, text_prompt)
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments, objects
def vis_and_save(source_video_frame_dir, text_prompt, save_frames_dir, output_video_path):
    """
    Step 5: Visualize the segment results across the video and save them
    """
    video_segments, objects = propagate_in_video(source_video_frame_dir, text_prompt)
    if not os.path.exists(save_frames_dir):
        os.makedirs(save_frames_dir)

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(objects, start=1)}

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
        
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
        cv2.imwrite(os.path.join(save_frames_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


    """
    Step 6: Convert the annotated frames to video
    """

    create_video_from_images(save_frames_dir, output_video_path)

if __name__=='__main__':
    vis_and_save(SOURCE_VIDEO_FRAME_DIR, TEXT_PROMPT, SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)
