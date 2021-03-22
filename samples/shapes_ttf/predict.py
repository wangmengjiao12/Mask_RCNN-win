# Using Tensorflow-2.4.x
import tensorflow as tf
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

import os
from PIL import Image
import tensorflow.keras as keras
import numpy as np
import random
import tensorflow as tf
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
import yaml
from mrcnn.model import log
import matplotlib.pyplot as plt

class ShapesDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self,image_id):
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

    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, img_floder, mask_floder, imglist, yaml_floder):
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
                self.add_image("shapes", image_id=i, path=img_path, mask_path=mask_path, yaml_path=yaml_path)

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
            if labels[i].find("circle")!=-1:
                labels_form.append("circle")
            elif labels[i].find("square")!=-1:
                labels_form.append("square")
            elif labels[i].find("triangle")!=-1:
                labels_form.append("triangle")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


# def log(text, array=None):
#     if array is not None:
#         text = text.ljust(25)
#         text += ("shape: {:20}  ".format(str(array.shape)))
#         if array.size:
#             text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
#         else:
#             text += ("min: {:10}  max: {:10}".format("",""))
#         text += "  {}".format(array.dtype)
#     print(text)

def get_ax(rows=1, cols=1, size=8):
    """返回Matplotlib Axes数组，以用于笔记本中的所有可视化。
        提供一个中心点来控制图形大小.

    调整大小属性以控制渲染图像的大小
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    # 应该通过设置IMAGES_PER_GPU来设置BATCH的大小
    # BATCHS_SIZE = IMAGES_PER_GPU*GPU_COUNT
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # 训练集和验证集长度已经自动计算


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


if __name__ == "__main__":
    learning_rate = 1e-5
    init_epoch = 0
    epoch = 100

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

    MODEL_DIR = "logs"
    config = ShapesConfig()
    inference_config = InferenceConfig()

    # 计算训练集和验证集长度
    config.STEPS_PER_EPOCH = len(train_imglist)//config.IMAGES_PER_GPU
    config.VALIDATION_STEPS = len(val_imglist)//config.IMAGES_PER_GPU
    config.display()

    # 训练数据集准备
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(len(train_imglist), img_floder, mask_floder, train_imglist, yaml_floder)
    dataset_train.prepare()

    # 验证数据集准备
    dataset_val = ShapesDataset()
    dataset_val.load_shapes(len(val_imglist), img_floder, mask_floder, val_imglist, yaml_floder)
    dataset_val.prepare()

    # 在 inference 模式下重新创建模型
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # 获取保存权重的路径
    model_path = "E:\\Mask_RCNN-tf2\\samples\\shapes_ttf\\mask_rcnn_shapes_0020.h5"

    # 加载训练权重
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # 测试随机图像
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config, image_id)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())



    # 计算VOC样式的mAP @ IoU = 0.5
    # 在100张图像上运行。 增加以获得更好的精度。
    image_ids = np.random.choice(dataset_val.image_ids, 5)
    APs = []
    for image_id in image_ids:
        # 加载图像和地面 ground truth 数据
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   #                                image_id, use_mini_mask=False)
                                   image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # 运行目标检测
        results = model.detect([image], verbose=0)
        r = results[0]
        # 计算AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))
