import numpy as np
import re
import os
import shutil
from tqdm import tqdm

    # Function to track objects in subsequent frames
def track_objects(previous_frame, current_frame, tracked_objects, frame_index):
    for obj_id, position in enumerate(current_frame):
        # Logic to determine if the object is the same as in the previous frame
        # Here, we can use a simple distance metric or more complex methods (e.g., IoU)
        for tracked_id, tracked_position in tracked_objects.items():
            # Example: Check if the positions are similar (you may want to refine this)
            if np.array_equal(position, tracked_position['positions'][-1]):
                tracked_objects[tracked_id]['positions'].append(position)
                tracked_objects[tracked_id]['frame_indices'].append(frame_index)
                break
        else:
            # New object detected, assign a new ID
            tracked_objects[len(tracked_objects)] = {
                'positions': [position],
                'frame_indices': [frame_index]
            }

def calculate_area(points):
    x1 , y1 ,x2, y2 = points
    return (x2-x1) * (y2-y1)

def get_frame_objects(frame, filte_agent=True, reference_id=None):
    """
    return objests: 
    - obj id 
        - position : numpy 4  for x1 , y1, x2, y2 
        - frame_indices 



    """
    ans = {}
    mean_area= 0 
    for obj_id in frame.keys():
        if obj_id == 'image_size': 
            continue 
        sum_obj = np.sum(frame[str(obj_id)])
        if sum_obj == 0 :
            continue 
            
        obj_id = int(obj_id)
        points = frame[str(obj_id)]
        area = calculate_area(points)
        # print(area)

        if filte_agent:
            if area> 6000:
                continue    
        if reference_id is not None:
            if obj_id not in  reference_id:
                continue
        mean_area += area
        ans[obj_id] = {
            'positions': [frame[str(obj_id)]],  # Store the initial position
            'frame_indices': [0]  # Store the frame index
        }
    #print('mean_area is ', mean_area/len(ans.keys()))
    return ans   

def load_arr(filepath):
    data = np.load(filepath, allow_pickle=True)
    frame_data = data['arr_0'].item()  # Assuming 'arr_0' contains the relevant data

    return frame_data
def check_first_frame(mask_dir):
    filepath =  os.path.join(mask_dir, 'frame000000.npz')
    if os.path.exists(filepath):
        data = load_arr(filepath)
        obj_list = get_frame_objects(data).keys()
        obj_num = len(obj_list)
        if obj_num ==8:
            # print('mask_dir' , mask_dir)
            return True 
    return False

def calculate_iou(a, b):
    a = a[0]
    b = b[0]
    # Unpack the bounding boxes
    
    # print(a, b ) 
    x_min_a, y_min_a, x_max_a, y_max_a = a
    x_min_b, y_min_b, x_max_b, y_max_b = b

    # Calculate the coordinates of the intersection rectangle
    x_min_intersection = max(x_min_a, x_min_b)
    y_min_intersection = max(y_min_a, y_min_b)
    x_max_intersection = min(x_max_a, x_max_b)
    y_max_intersection = min(y_max_a, y_max_b)

    # Calculate the area of intersection
    intersection_width = max(0, x_max_intersection - x_min_intersection)
    intersection_height = max(0, y_max_intersection - y_min_intersection)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of each bounding box
    area_a = (x_max_a - x_min_a) * (y_max_a - y_min_a)
    area_b = (x_max_b - x_min_b) * (y_max_b - y_min_b)

    # Calculate the IoU
    iou = intersection_area / area_a if area_a > 0 else 0

    return iou


def copy_available_data(mask_dir, video_id, store_dir='./no_hide/', ori_dir=None):
    if check_first_frame(mask_dir):
        if ori_dir is None :
            video_dir = os.path.dirname(mask_dir)
        else: 
            video_dir = os.path.join(ori_dir, video_id, 'images/00000.jpg')
        shutil.copy(video_dir, os.path.join(store_dir, str(video_id)+'.jpg'))

def find_delta_ids(id1, id2):
    """ 
    will only count all the missing part from the first frame 

    """
    # print(set(id1), set(id2))
    delta_list = list(set(id1) - set(id2))
    delta_from_previous = [dd for dd in delta_list if dd in id1]
    return delta_from_previous
def check_missing(pre_frame, this_frame, missing_id):
    miss, hide = 0, 0
    if len(missing_id) >= 1 : 
        
        # print(missing_id)
        for miss in missing_id:
            miss_bbx = pre_frame[miss]['positions']
            for alter_id,  alter in this_frame.items(): 
                alter = alter['positions']
                iou = calculate_iou(miss_bbx, alter)
                # print('iou is ', iou)
                if iou > 0.6:
                    hide += 1 
                    break
        miss = len(missing_id) - hide
    assert miss + hide == len(missing_id)
    return miss, hide



def update_metrics(pre_frame, this_frame, history_ids):
    pre_len = len(pre_frame.keys())
    this_len = len(this_frame.keys())
    pre_ids = pre_frame.keys()
    this_ids = this_frame.keys()
    missing_id = find_delta_ids(pre_ids, this_ids)
    delta_addon_id = find_delta_ids(this_ids, pre_ids)
    addon = 0
    for i in delta_addon_id: 
        if i not in history_ids:
            addon+=1 
            history_ids.add(i)
    miss, hide = check_missing(pre_frame, this_frame, missing_id)

    # if this_len < pre_len:
    #     assert hide + miss + this_len == pre_len, f'{hide, miss, this_len, pre_len}'
    return {'hide': hide, 'miss': miss, 'addon':addon}, history_ids

# Directory containing the mask files
def object_num_metric(mask_dir):
    # Step 1: Load all frame data, history_ids
    frame_data = {} 
    sorted_dir = [x for x in os.listdir(mask_dir) if x.endswith('npz')]
    def get_frame_id(name):
        idd = name.split('.')[0]
        idd = int(idd[5:])
        return idd
    sorted_filenames = sorted(sorted_dir, key=lambda x: get_frame_id(x))
    if len(sorted_filenames) == 0 :
        return 
    itt = 0
    for filename in sorted_filenames:
        if '.npz' in filename:
            filepath = os.path.join(mask_dir, filename)
            frame_data[itt] = load_arr(filepath)
            itt +=1 

    # Step 2: Track objects across frames
    # Initialize a dictionary to hold the tracking information
    tracked_objects = {}
    
    # Assume the first frame is the reference for tracking
    first_frame = frame_data[0]
    # Initialize tracked objects with the first frame
    obj_num_list = []
    objs_list = []
    miss_items = 0 
    hide_items = 0
    addon_items = 0
    reference_id = None 
    history_ids = set()
    for frame_id, frame in enumerate(frame_data.values()):
        
        objs = get_frame_objects(frame, filte_agent=False, reference_id=reference_id)
        if frame_id == 0 : 
            reference_id = objs.keys()
            history_ids.update(set(reference_id))
            # print(history_ids)
        objs_list.append(objs)
        
        obj_num_list.append(len(objs.keys()))
        if frame_id >=1   : 
            prev_len = obj_num_list[frame_id-1]
            this_len = obj_num_list[frame_id]
            prev_frame = objs_list[frame_id-1]
            this_frame = objs_list[frame_id]
            if prev_len != this_len or len(find_delta_ids(prev_frame.keys(), this_frame.keys())): 
                
               # update_metrics(prev_frame, this_frame)
                # print('now processing the id', frame_id)
                miss_dict, history_ids= update_metrics(prev_frame, this_frame, history_ids)
                # print(miss_dict)
                miss_items += miss_dict['miss']
                hide_items += miss_dict['hide']
                addon_items += miss_dict['addon']
    # print(obj_num_list)
    if miss_items +  hide_items + addon_items != 0 : 
        print(miss_items, hide_items, addon_items, mask_dir, mask_dir.split('/')[-1])
if __name__=='__main__':

    # mask_dir = './dataset/mask_data/val/'  # Update this to your mask directory
    # for i in tqdm(os.listdir(mask_dir)):
    #     test_dir = os.path.join(mask_dir, i, 'masks')
    #     # check_first_frame(test_dir)
    #     copy_available_data(test_dir, i, ori_dir=mask_dir)
    mask_dir = '/home/lr-2002/code/IRASim/generate_video/'  # Update this to your mask directory
    all_dirs = os.listdir
    for video in tqdm(sorted(os.listdir(mask_dir))):
        path = os.path.join(mask_dir, video)
        # input('next?')
        if os.path.isdir(path):

            # test_dir = os.path.join(mask_dir, i, 'masks')

            test_dir = path
            # check_first_frame(test_dir)
            # copy_available_data(test_dir, i, ori_dir=mask_dir)
            object_num_metric(test_dir)
