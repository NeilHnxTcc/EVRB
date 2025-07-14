import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image




class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        image_list, query_list, label_list = [], [], []
        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        # return 1500
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")

        # # 使用PIL的size属性获取维度信息
        # width, height = raw_image.size  # (宽度, 高度)
        # # 计算裁剪高度范围（基于原代码逻辑）
        # h_width = width // 4  # 使用宽度的1/4作为裁剪半高
        # top = max(0, (height // 2) - h_width)
        # bottom = min(height, (height // 2) + h_width)
        # # 使用PIL的crop方法进行区域裁剪
        # # 参数格式：(左, 上, 右, 下)
        # raw_image = raw_image.crop((0, top, width, bottom))


        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label, 'raw_img':np.array(raw_image), 'image_path':image_path}