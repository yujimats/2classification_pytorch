import os
from PIL import Image
import numpy as np
import cv2

import torch.utils.data as data
from torchvision import transforms

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, image):
        return self.data_transform(image)

class MyDataset(data.Dataset):
    def __init__(self, list_file, transform=None, phase='train'):
        self.list_file = list_file
        self.transform = transform
        self.phase = phase

    def __len__(self):
        # ファイル数を返す
        return len(self.list_file)

    def __getitem__(self, index):
        # 画像をPillowsで開く
        path_image = self.list_file[index][0]
        pil_image = Image.open(path_image).convert('RGB')

        # 画像の前処理
        image_transformed = self.transform(pil_image)

        # ラベルを取得
        label_class = self.list_file[index][1]
        label_type = self.list_file[index][2]
        return image_transformed, label_class

class MyDataset_path(data.Dataset):
    def __init__(self, list_file, path_input, transform=None, phase='train'):
        self.list_file = list_file
        self.path_input = path_input
        self.transform = transform
        self.phase = phase

    def __len__(self):
        # ファイル数を返す
        return len(self.list_file)

    def __getitem__(self, index):
        # 画像のパスを取得
        path_image = os.path.join(self.path_input, self.list_file[index][0])

        # ラベルを取得
        label_class = self.list_file[index][1]
        label_type = self.list_file[index][2]
        return path_image, label_class
