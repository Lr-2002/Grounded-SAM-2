import os

# 定义要遍历的根目录
root_dir = './dataset/mask_data/train/'

# 遍历所有子文件夹
for subdir, dirs, files in os.walk(root_dir):
    if 'masks' in subdir or  'images' in subdir:
         continue
    images_path = os.path.join(subdir, 'images')
    masks_path = os.path.join(subdir, 'masks')

    # 检查 images 和 masks 文件夹是否存在
    if os.path.exists(images_path) and os.path.exists(masks_path):
        # 计算 images 和 masks 文件夹中的文件数量
        image_count = len(os.listdir(images_path))
        mask_count = len(os.listdir(masks_path))

        # 输出结果
        if image_count == mask_count:
            print(f"{subdir}: images 和 masks 的文件数量相等 ({image_count})")
        else:
            print(f"{subdir}: images 和 masks 的文件数量不相等 (images: {image_count}, masks: {mask_count})")
    else:
        print(f"{subdir}: images 或 masks 文件夹不存在")

