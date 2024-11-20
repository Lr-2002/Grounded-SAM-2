import numpy as np
import matplotlib.pyplot as plt
import cv2
def load_and_save(file_path):
    data = np.load(file_path)

    print("Data shape:", data.shape)

    num_objects = data.shape[0]
    images = []

    for i in range(num_objects):
        img = data[i, :, :]  # 形状为 (h, w)
        
        images.append(img)
        cv2.imwrite(f'image_{i+1}.png', img * 255)  # 将图像值缩放到 [0, 255]

        # plt.imshow(img, cmap='gray')  # 使用灰度色图显示
        # plt.title(f'Image {i+1}')
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
        #
file_path = './dataset/mask_data/train/000016/masks/00002.npy'  # 替换为你的文件路径
load_and_save(file_path)
