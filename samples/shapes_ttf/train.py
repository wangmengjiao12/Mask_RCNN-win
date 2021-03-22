# Using Tensorflow-2.4.x
import tensorflow as tf
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

import os
import sys
from PIL import Image
import tensorflow.keras as keras
import numpy as np
import random
import tensorflow as tf
import yaml

# 项目根目录 E:\Mask_RCNN-tf2
ROOT_DIR = os.path.abspath("../../")

# 导入 Mask RCNN
sys.path.append(ROOT_DIR)  # 查找库的本地版本

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import matplotlib.pyplot as plt

# 保存日志和训练的模型的目录
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# 训练权重文件的本地路径
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class ShapesConfig(Config):
    # 新的配置名字
    NAME = "shapes"

    # 应该通过设置IMAGES_PER_GPU来设置BATCH的大小
    # BATCHS_SIZE = IMAGES_PER_GPU*GPU_COUNT
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 类数目
    NUM_CLASSES = 3 + 1  # 3类 + 1个背景

    #
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    #
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # 训练集和验证集长度已经自动计算

    # #
    STEPS_PER_EPOCH = 50


class ShapesDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] =1
        return mask

    # 重写load_shapes，里面包含子集的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, img_floder, mask_floder, imglist, yaml_floder):
        # Add Classes
        self.add_class("shapes", 1, "circle")
        self.add_class("shapes", 2, "square")
        self.add_class("shapes", 3, "triangle")

        for i in range(count):

            img = imglist[i]
            if img.endswith(".jpg"):
                img_name = img.split(".")[0]
                img_path = img_floder + img
                mask_path = mask_floder + img_name + ".png"
                yaml_path = yaml_floder + img_name + ".yaml"
                self.add_image("shapes", image_id=i, path=img_path, mask_path=mask_path,
                               yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)

        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):

            if labels[i].find("circle") != -1:
                labels_form.append("circle")
            elif labels[i].find("square") != -1:
                labels_form.append("square")
            elif labels[i].find("triangle") != -1:
                labels_form.append("triangle")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


'''
def get_ax(rows=1, cols=1, size=8):
    """返回Matplotlib Axes数组，以用于笔记本中的所有可视化。
        提供一个中心点来控制图形大小.

    调整大小属性以控制渲染图像的大小
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
'''


# ###########################################################################
# # Training
# #
# ###########################################################################

# 您的数据路径
dataset_root_path = "E:\\Mask_RCNN-tf2\\samples\\shapes_ttf\\dataset\\"
img_floder = dataset_root_path + "imgs/"
mask_floder = dataset_root_path + "mask/"
yaml_floder = dataset_root_path + "yaml/"
imglist = os.listdir(img_floder)
count = len(imglist)

np.random.seed(10101)
np.random.shuffle(imglist)
train_imglist = imglist[:int(count*0.9)]
val_imglist = imglist[int(count*0.9):]

config = ShapesConfig()
config.display()

# COCO_MODEL_PATH = "E:\\Mask_RCNN-tf2\\mask_rcnn_coco.h5"

# 计算训练集和验证集长度
config.STEPS_PER_EPOCH = len(train_imglist)//config.IMAGES_PER_GPU
config.VALIDATION_STEPS = len(val_imglist)//config.IMAGES_PER_GPU
# config.display()

# 训练数据集准备
dataset_train = ShapesDataset()
dataset_train.load_shapes(len(train_imglist), img_floder, mask_floder, train_imglist, yaml_floder)
dataset_train.prepare()
# 验证数据集准备
dataset_val = ShapesDataset()
dataset_val.load_shapes(len(val_imglist), img_floder, mask_floder, val_imglist, yaml_floder)
dataset_val.prepare()

# 获得训练模型
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# 加载权重
model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

# 训练head分支Passing layer =“ heads”冻结除head层以外的所有层.
# 您还可以传递一个正则表达式以按名称模式选择要训练的图层。
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=2,
            layers='heads')

# # 微调所有图层
# # 通过layers =“ all”训练所有层. 您还可以传递一个正则表达式以按名称模式选择要训练的图层。
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")


