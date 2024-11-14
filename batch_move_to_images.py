import os
import shutil

from tqdm import tqdm
# 根目录路径
root_dir = '/ssd/data_boxing_hard/'  # 替换为实际路径

# 遍历根目录中的所有 'episode' 文件夹
for episode_folder in tqdm(os.listdir(root_dir)):
    episode_path = os.path.join(root_dir, episode_folder)
    
    # 检查是否是目录
    if os.path.isdir(episode_path):
        # 创建 'images' 文件夹路径
        images_folder = os.path.join(episode_path, 'images')
        
        # 如果 'images' 文件夹不存在，则创建
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)
        
        # 遍历 episode 文件夹中的所有文件
        for file_name in os.listdir(episode_path):
            file_path = os.path.join(episode_path, file_name)
            
            # 如果文件是 .png 文件，则移动到 images 文件夹
            if os.path.isfile(file_path) and file_name.endswith('.png'):
                shutil.move(file_path, os.path.join(images_folder, file_name))

print("所有图像文件已成功移动到各自的 images 文件夹中。")

