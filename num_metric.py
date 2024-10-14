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

def get_frame_objects(frame, filte_agent=True):
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

def copy_available_data(mask_dir, video_id, store_dir='./no_hide/', ori_dir=None):
    if check_first_frame(mask_dir):
        if ori_dir is None :
            video_dir = os.path.dirname(mask_dir)
        else: 
            video_dir = os.path.join(ori_dir, video_id, 'images/00000.jpg')
        shutil.copy(video_dir, os.path.join(store_dir, str(video_id)+'.jpg'))

# Directory containing the mask files
def object_num_metric(mask_dir):
    # Step 1: Load all frame data
    frame_data = {}
    sorted_dir = os.listdir(mask_dir)
    sorted_filenames = sorted(sorted_dir, key=lambda x: int(re.search(r'\d+', x).group()))
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
    for frame in frame_data.values():
        obj_list = get_frame_objects(frame).keys()
        print(obj_list)
    
if __name__=='__main__':

    mask_dir = './dataset/mask_data/train/'  # Update this to your mask directory
    for i in tqdm(os.listdir(mask_dir)):
        test_dir = os.path.join(mask_dir, i, 'masks')
        # check_first_frame(test_dir)
        copy_available_data(test_dir, i, ori_dir=mask_dir)
        # object_num_metric(test_dir)
