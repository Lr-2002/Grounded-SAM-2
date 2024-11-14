import os
from collections import OrderedDict
import re
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
VIDEO_SUFFIX = 'rgb.mp4'
PATH_SUFFIX = 'videos'
PATH_REPLACE = PATH_SUFFIX + '_seged'
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
def calculate_area(bbox):
    # Calculates the area of a single bounding box
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def update_bboxes(bboxes, labels, confidences, image_width, image_height):
    height_threshold = 0.01 * image_height
    width_threshold = 0.01 * image_width
    agents = []
    
    # Find agents: bounding boxes close to any image border
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        if (
            x1 <= width_threshold or                # Close to left border
            x2 >= (image_width - width_threshold) or # Close to right border
            y1 <= height_threshold or               # Close to top border
            y2 >= (image_height - height_threshold) # Close to bottom border
        ):
            agents.append(i)
    # Filter agents based on IoU

    agents_to_remove = set()
    for i in agents:
        for j in range(len(bboxes)):
            if i == j :
                continue 
            ious = calculate_iou(bboxes[i], bboxes[j]) 
            if i != j and (ious[0]> 0.9 or ious[1] > 0.9):
                agents_to_remove.add(i)
                break
    
    # Calculate mean area of non-agent bounding boxes
    non_agent_bboxes = [bbox for i, bbox in enumerate(bboxes) if i not in agents_to_remove]
    non_agent_areas = [calculate_area(bbox) for bbox in non_agent_bboxes]
    mean_area = np.mean(non_agent_areas) if non_agent_areas else 0
    
    # Remove non-agents that are too large
    final_bboxes = []
    final_labels = []
    final_confidences = []
    
    for i, bbox in enumerate(bboxes):
        if i not in agents_to_remove:
            # Check if the area exceeds 4 times the mean area
            area = calculate_area(bbox)
            if (i in agents and i not in agents_to_remove) or area <= 5 * mean_area:
                final_bboxes.append(bbox)
                final_labels.append(labels[i])
                final_confidences.append(confidences[i])
    
    # Parameters to return
    agents_remaining = bool(len(agents) - len(agents_to_remove))
    num_bboxes = len(final_bboxes)
    
    return final_bboxes, final_labels, final_confidences, agents_remaining, num_bboxes


def filte_big_box(boxes, labels, confidences,total_area, w, h) : 
    assert boxes.shape[1] == 4 
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) 
    mask =( areas  / total_area )  < 0.5
    filtered_boxes = boxes[mask]
    filtered_labels = [label for i, label in enumerate(labels) if mask[i]]
    filtered_confid = [conf for i, conf in enumerate(confidences) if mask[i]]
    
    filtered_boxes, filtered_labels, filtered_confid, remain, num_len = update_bboxes(filtered_boxes, filtered_labels, filtered_confid, w, h)

    return filtered_boxes, filtered_labels, filtered_confid, remain, num_len

def save_bbox_vis(bboxes, width, height, path="bbox_object"):
    # Iterate over each bounding box and save as a PNG file
    for i, bbox in enumerate(bboxes):
        # Create a blank white image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw the bounding box
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        # Save the image with bounding box
        img.save(f"saved_mask/{path}_{i}.png")

def save_mask_vis(masks, path="mask_object"):
        # 遍历每个物体的掩码并保存为 PNG 文件
        for i, mask in enumerate(masks):
            # 将掩码转换为图片格式 (0, 255)
            img = Image.fromarray((mask * 255).astype(np.uint8))
            
            # 保存为 PNG 文件
            img.save(f"{path}_{i}.png")

class VideoProcessor:
    def __init__(self, save_video=True, save_bbox=True, save_mask=False, save_mask_vis=False, re_split=False, save_bbox_vis=True) -> None:
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
        self.save_mask_vis = save_mask_vis
        self.need_save_mask = save_mask

        self.need_save_bbox = save_bbox
        self.save_video = save_video
        self.re_split = re_split

    def split_video(self):
        """
        Custom video input directly using video files
        """
        video_info = sv.VideoInfo.from_video_path(self.video_path)  # get video info
        self.video_info = video_info
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
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
        ]
        if len(frame_names) ==  0:
            frame_names = [
                p for p in os.listdir(self.source_video_frame_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
            ]
        if '_' in frame_names[0]:
            frame_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        else: 
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        # init video predictor state

        self.frame_names = frame_names
        self.video_len = len(frame_names)
        self.inference_state = self.video_predictor.init_state(video_path=self.source_video_frame_dir)

        return frame_names
    def set_ref_idx(self, idx=0):
        self.ann_frame_idx = idx  # the frame index we interact with

    def gdino_inner_process(self,img_path):
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

        return image, input_boxes, confidences, class_names


    def set_better_ref(self, stride=3):
        stride = stride if self.video_len < stride * 10 else self.video_len // 10
        max_num = 0 
        flag = None
        for idx in range(0, self.video_len, stride):
            img_path = os.path.join(self.source_video_frame_dir, self.frame_names[idx])
            img, input_boxes, class_names, confidences  = self.gdino_inner_process(img_path)
            w, h = self.video_info.resolution_wh
            input_boxes, class_names, confidences, remain, num_len = filte_big_box(input_boxes, class_names, confidences, w * h, w,h)
            if remain and num_len > max_num:
                flag = remain
                max_num = num_len
                self.set_ref_idx(idx)
        if flag== None :
            self.set_ref_idx(-1)


    # prompt grounding dino to get the box coordinates on specific frame
    def gdino_process(self):
        assert hasattr(self, 'ann_frame_idx')
        if self.ann_frame_idx == -1:
            return False
        img_path = os.path.join(self.source_video_frame_dir, self.frame_names[self.ann_frame_idx])
        image, input_boxes, confidences, class_names = self.gdino_inner_process(img_path)

        converted_image = np.array(image.convert("RGB"))
        h, w = converted_image.shape[:2]
        self.input_boxes, self.class_names, self.confidences,_, _ = filte_big_box(input_boxes, class_names, confidences, w * h, w,h)
        self.objects = self.class_names
        self.converted_image = converted_image
        return True
    def sam2_image_process(self):

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(self.converted_image)

        # process the detection results

        self.image_masks, self.image_scores, self.image_logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=self.input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if self.image_masks.ndim == 4:
            self.image_masks = self.image_masks.squeeze(1)
        if self.save_mask_vis:
            save_mask_vis(self.image_masks)

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
        return self.video_segments is 
        frame_idx : obj_id : masks 
        """
        # prev_seg = 
        self.video_segments = {}  # self.video_segments contains the per-frame segmentation results

        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state,reverse=False):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        reverse = self.ann_frame_idx != 0 
        if reverse:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state,reverse=reverse):
                self.video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        return self.video_segments
    def vis_and_save(self, mask_save_dir=None, save_all=True):
        """
        Step 5: Visualize the segment results across the video and save them
        """
        if not os.path.exists(self.save_tracking_results_dir):
            os.makedirs(self.save_tracking_results_dir)

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(self.objects, start=1)}

        old_frame_idx = 0
        self.video_segments = OrderedDict(sorted(self.video_segments.items()))

        all_detections = []
        all_masks = []
        for frame_idx, segments in self.video_segments.items():
            assert frame_idx >= old_frame_idx , f'{frame_idx}, {old_frame_idx}'
            old_frame_idx = frame_idx
            img = cv2.imread(os.path.join(self.source_video_frame_dir, self.frame_names[frame_idx]))
            
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            all_detections.append(detections)
            if save_all:
                if self.need_save_bbox:
                    self.save_bbox(detections, frame_idx, save_dir=mask_save_dir)
                if self.need_save_mask:
                    self.save_mask(masks, frame_idx, save_dir = mask_save_dir)

                if self.save_video:
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
        if save_all:
            if self.save_video:
                create_video_from_images(self.save_tracking_results_dir, self.output_video_path)
        return all_detections
    def update_files(self, video_path, output_video_path='processed_video.mp4', text_prompt='object.', source_video_frame_dir='./tmp/source_video_frame', save_tracking_results_dir='./tmp/save_tracking_results'):
        self.video_path = video_path
        self.text_prompt = text_prompt
        self.output_video_path = output_video_path
        self.source_video_frame_dir = source_video_frame_dir
        self.save_tracking_results_dir = save_tracking_results_dir

        import shutil
        if os.path.exists(save_tracking_results_dir):
            shutil.rmtree(save_tracking_results_dir)
    def makedir(self, paths):
        if paths[0] != '/':
            path_parts = paths.split('/')

            current_path = ''
        else :
            path_parts = paths.split('/')[1:]
            current_path='/'
        for part in path_parts:
            if part =='':
                continue 
            current_path = os.path.join(current_path, part)
            if not os.path.exists(current_path):
                os.makedirs(current_path, exist_ok=True)

    def save_mask(self, masks, frame_idx, save_dir=None):

        # Construct the new save directory for masks
        save_dir = self.video_path.replace(PATH_SUFFIX, PATH_REPLACE).replace(VIDEO_SUFFIX, 'masks').replace('images', 'masks')  if save_dir is None else save_dir
        self.makedir(save_dir)
        # Construct the filename for the frame
        
        frame_filename = f'{frame_idx:05d}.npz'
        frame_path = os.path.join(save_dir, frame_filename)
        masks_shape = masks.shape 
        masks = np.resize(masks_shape, (masks_shape[0], 180, 320))
        np.savez_compressed(frame_path, masks)



    def save_bbox(self, detections, frame_idx, save_dir=None):

        save_dir = self.video_path.replace(PATH_SUFFIX, PATH_REPLACE).replace(VIDEO_SUFFIX, 'bbox').replace('image', 'bbox') if save_dir is None else save_dir
        self.makedir(save_dir)

        frame_filename = f'{frame_idx:05d}.npy'
        frame_path = os.path.join(save_dir, frame_filename)
        image_size = ( 1, *self.video_info.resolution_wh)   # Example image size, adjust as needed 
        bbox_data = {'image_size': image_size}
        
        for idx, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            x1, y1, x2, y2 = bbox
            class_id = str(class_id)
            bbox_data[class_id] = [x1, y1, x2, y2]

        data = {'arr_0': bbox_data}
        np.save(frame_path, bbox_data)


    def check_dir_len(self, path):
        return len(os.listdir(path))

    def check_if_continue(self, images_dir):
        mask_dir = images_dir.replace('images', 'masks')
        save_path = self.video_path.replace(PATH_SUFFIX, PATH_REPLACE).replace(VIDEO_SUFFIX, 'masks.npy') 
        if os.path.exists(save_path):
            return False
        if os.path.exists(mask_dir) and os.path.exists(images_dir):
            if self.check_dir_len(mask_dir) == self.check_dir_len(images_dir) and self.check_dir_len(mask_dir) != 0:
                return True

        return False
    def generate_from_split_images_with_bbox(self, video_path, output_video_path='processed_video.mp4', text_prompt='object.', source_video_frame_dir='./tmp/source_video_frame', save_tracking_results_dir='./tmp/save_tracking_results', mask_save_dir=None, split_for_metric=False):
        # self.update_files(video_path, output_video_path,text_prompt, source_video_frame_dir, save_tracking_results_dir)
        self.text_prompt = text_prompt
        self.source_video_frame_dir = source_video_frame_dir
        # self.check_if_need_split()

        self.load_frame_name()
        test_image = Image.open(os.path.join(source_video_frame_dir, self.frame_names[0]))
        image_size = test_image.size
        self.video_info = sv.VideoInfo(width=image_size[0], height=image_size[1], fps=25)
        # self.set_better_ref()

        # remain = self.gdino_process()
        # if remain:

        self.input_boxes = [np.array([33,38, 52, 89]), np.array([105, 121, 125, 170])]
        converted_image = np.array(test_image.convert("RGB"))
        h, w = converted_image.shape[:2]
        self.objects = [0,1]
        self.converted_image = converted_image
        self.ann_frame_idx = 0
        self.sam2_image_process()

        self.prepare_sam2_video()

        self.propagate_in_video()


        return self.vis_and_save(save_all=True)

 

    def get_detections(self, video_path, output_video_path='processed_video.mp4', text_prompt='object.', source_video_frame_dir='./tmp/source_video_frame', save_tracking_results_dir='./tmp/save_tracking_results', mask_save_dir=None, split_for_metric=False):
        # self.update_files(video_path, output_video_path,text_prompt, source_video_frame_dir, save_tracking_results_dir)

        self.text_prompt = text_prompt
        self.source_video_frame_dir = source_video_frame_dir
        self.check_if_need_split()

        self.load_frame_name()
        test_image = Image.open(os.path.join(source_video_frame_dir, self.frame_names[0]))
        image_size = test_image.size
        self.video_info = sv.VideoInfo(width=image_size[0], height=image_size[1], fps=25)
        self.set_better_ref()

        remain = self.gdino_process()
        if remain:
            self.sam2_image_process()

            self.prepare_sam2_video()

            self.propagate_in_video()


            return self.vis_and_save(save_all=False)

   
    def update_and_process(self, video_path, output_video_path='processed_video.mp4', text_prompt='object.', source_video_frame_dir='./tmp/source_video_frame', save_tracking_results_dir='./tmp/save_tracking_results', mask_save_dir=None, split_for_metric=False):
        source_video_frame_dir = video_path.replace(PATH_SUFFIX, PATH_REPLACE).replace(VIDEO_SUFFIX, 'images') if '0-th' not in video_path else source_video_frame_dir
        if_continue = self.check_if_continue(source_video_frame_dir) if split_for_metric is False else False 
        if if_continue:
            print('----> skip the dir ', source_video_frame_dir)
            return 
        self.update_files(video_path, output_video_path,text_prompt, source_video_frame_dir, save_tracking_results_dir)
        save_dir = self.video_path.replace(PATH_SUFFIX, PATH_REPLACE).replace(VIDEO_SUFFIX, 'bbox')
        if os.path.exists(save_dir):
            return 

        self.check_if_need_split()

        

        self.load_frame_name()
        
        self.set_better_ref()

        remain = self.gdino_process()
        if remain:
            self.sam2_image_process()

            self.prepare_sam2_video()

            self.propagate_in_video()


            self.vis_and_save(mask_save_dir=mask_save_dir)
        else: 
            print('-----> deparched video', video_path)
if __name__=='__main__':
    # process_model = VideoProcessor(save_video=False, re_split=True, save_bbox=True, save_mask=True, save_mask_vis=False, save_bbox_vis=False)
    # dir_path = '/ssd/lt/processed_dataset/lt_sim/train/'
    # dir_path = './dataset/videos/train/'
    # #
    # videos = os.listdir(dir_path)
    # cnt = 0
    # from tqdm import tqdm
    # for video in tqdm(videos):
    #     cnt +=1 
    #     video_path =  dir_path + video + f'/{VIDEO_SUFFIX}'
    #     process_model.update_and_process(video_path, output_video_path='output_dir/' +str(cnt) + '.mp4', text_prompt='object.')
    #

    # dir_path = '/home/lr-2002/code/IRASim/generate_video'
    # videos = os.listdir(dir_path)
    # for video in tqdm(videos):
    #     if video.endswith('.mp4'):
    #         video_id = video.split('_')[0]
    #         if len(video_id) != 6 :
    #             continue 
    #         mask_dir = os.path.join(dir_path, str(video_id) + '/')
    #         if not os.path.exists(mask_dir):
    #             os.mkdir(mask_dir)
    #         print('video dir is ', video_id, 'mask_dir is', mask_dir)
    #         process_model.update_and_process(video_path=os.path.join(dir_path, video), mask_save_dir=mask_dir, save_tracking_results_dir=os.path.join(dir_path, video_id, 'annotate'), split_for_metric=True)
    #

    
    # data_boxing_hard 

    process_model = VideoProcessor(save_video=False, re_split=True, save_bbox=True, save_mask=True, save_mask_vis=False, save_bbox_vis=False)
    dir_path = '/ssd/data_boxing_hard/'
    for video in tqdm(os.listdir(dir_path)):
        video_path = os.path.join(dir_path,  video, 'images') 
        process_model.generate_from_split_images_with_bbox(None, source_video_frame_dir=video_path)
