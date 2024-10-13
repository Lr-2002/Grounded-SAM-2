import numpy as np
import re
import os

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

def get_frame_objects(frame):
    ans = {}
    for obj_id in frame.keys():
        if obj_id == 'image_size': 
            continue 
        sum_obj = np.sum(frame[str(obj_id)])
        if sum_obj == 0 :
            continue 
        obj_id = int(obj_id)
        ans[obj_id] = {
            'positions': [frame[str(obj_id)]],  # Store the initial position
            'frame_indices': [0]  # Store the frame index
        }

    return ans   


# Directory containing the mask files
def object_num_metric(mask_dir):
    # Step 1: Load all frame data
    frame_data = {}
    sorted_dir = os.listdir(mask_dir)
    sorted_filenames = sorted(sorted_dir, key=lambda x: int(re.search(r'\d+', x).group()))
    itt = 0
    for filename in sorted_filenames:
        if filename.endswith('.npz'):
            
            filepath = os.path.join(mask_dir, filename)
            data = np.load(filepath, allow_pickle=True)
            frame_data[itt] = data['arr_0'].item()  # Assuming 'arr_0' contains the relevant data
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

    mask_dir = './dataset/mask_data/test/'  # Update this to your mask directory
    for i in os.listdir(mask_dir):
        test_dir = os.path.join(mask_dir, i, 'masks')
        print(test_dir)
        object_num_metric(test_dir)
