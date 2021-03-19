"""
Mask R-CNN
在 toy balloon 数据集上训练并实现 颜色飞溅 效果.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
train  --dataset "E:\\Mask_RCNN-tf2\\samples\\balloon"  --weights coco
splash  --weights "E:\\Mask_RCNN-tf2\\mask_rcnn_balloon.h5"  --image "E:\\Mask_RCNN-tf2\\samples\\balloon\\train\\120853323_d4788431b9_b.jpg"
splash  --weights "E:\\Mask_RCNN-tf2\\logs\\balloon20210318T1851\\mask_rcnn_balloon_0030.h5"  --image "E:\\Mask_RCNN-tf2\\samples\\balloon\\train\\120853323_d4788431b9_b.jpg"



Usage: 导入模块 (see Jupyter notebooks for examples), 或从命令行运行，如下所示:

    # 从预先训练的COCO重量开始训练新模型
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # 恢复训练您之前训练过的模型
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # 从 InageNet 权重开始训练新模型
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # 将 颜色飞溅 应用于图像
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # 使用您训练的最后权重将 颜色飞溅 应用于视频
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf

# GPU限制
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 项目的根目录
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)  # E:\Mask_RCNN-tf2
# 导入 Mask RCNN
sys.path.append(ROOT_DIR)  # 查找库的本地版本
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# 训练后的权重文件的路径
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_WEIGHTS_PATH)  # E:\Mask_RCNN-tf2\mask_rcnn_coco.h5

# 用于保存日志和模型检查点的目录 (如果未通过命令行参数 --logs 提供)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
print(DEFAULT_LOGS_DIR)  # E:\Mask_RCNN-tf2\logs

# 数据集文件路径
BALLOON_DIR = os.path.join(ROOT_DIR, "samples/balloon")
print(BALLOON_DIR)


############################################################
#  Configurations
############################################################
class BalloonConfig(Config):
    """用于对 toy balloon 数据集进行训练的配置.
    派生自基本的Config类， 并覆盖一些值.
    """
    # 为配置指定一个可识别的名称
    NAME = "balloon"

    # 我们使用具有12GB内存的GPU，该内存可以容纳两个图像
    # IMAGES_PER_GPU = 2.
    # 若使用较小的GPU，请向下调整.
    # 本机是8GB内存的GPU，该内存使用一个图像.
    IMAGES_PER_GPU = 1

    # 类别数 (包括背景)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # 每个 epoch 的训练步骤数
    STEPS_PER_EPOCH = 100

    # < 90% 置信度的跳过检测
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################
class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """加载 Balloon 数据集的子集.
        dataset_dir: 数据集的根目录.
        要加载的子集: train or val
        """
        # 添加 classes. 我们仅添加一个类.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }

        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # 即使图像没有任何注释，VIA工具也会将图像保存在JSON中.
        # 跳过未注释的图像
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """生成图像的实例 mask.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # 如果不是 balloon dataset 图像, 则委托给父类.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # 将多边形转化为位图形状的mask
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # 获取多边形内像素的索引并将其设置为1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # 返回掩码，以及每个实例的类ID数组。
        # 由于我们只有一个类ID，因此我们返回1的数组
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """返回图像的路径."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """训练模型."""
    # 训练数据.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # 验证数据
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** 这个培训表只是一个例子. 更新您的需求 ***
    # 由于我们使用的数据集非常小，并且从COCO训练的权重开始，因此我们不需要训练太长时间.
    # 另外，不需要训练所有层，只要训练头部就可以完成
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """应用 颜色飞溅 效果.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns 结果图像.
    """
    # 制作图像的灰度副本.
    # 尽管灰度副本仍然具有3个通道
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # 从设置了mask的原始色彩图像中复制彩色像素
    if mask.shape[-1] > 0:
        # 我们将所有实例视为一个，因此将mask折叠为一层
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # 运行模型检测并生成颜色飞溅效果
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define code and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################
if __name__ == '__main__':
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # 验证参数
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # 设置 batch size 为 1 因此我们一次将对一幅图像进行推理.
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # 创建模型
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # 选择要加载的权重文件
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # 下载 weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # 查找最近的训练权重
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # 从 ImageNet 的权重开始训练
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # 加载 weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # 排除最后一层，应为特们需要匹配数量的类
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # 训练或评估
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
