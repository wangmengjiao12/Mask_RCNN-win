"""
Mask R-CNN
MS COCO 的配置 和 数据加载代码.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
evaluate  --dataset "E:\\Mask_RCNN-tf2\\samples\\coco\\dataset"  --model "E:\\Mask_RCNN-tf2\\mask_rcnn_coco.h5"
train  --dataset "E:\\Mask_RCNN-tf2\\samples\\coco\\dataset"  --model "E:\\Mask_RCNN-tf2\\mask_rcnn_coco.h5"




Usage: 导入模块 (see Jupyter notebooks for examples), 或从命令行运行，如下所示

    # 从预先训练的COCO权重开始训练新模型
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # 从ImageNet权重开始训练新模型. 同时自动下载COCO数据集.
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # 继续训练您先前训练过的模型
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # 继续训练您训练的最后一个模型
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # 对您训练的最后一个模型进行COCO评估
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

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
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# 下载pycocotools来实现对MS-COCO数据集的操作
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# 项目的根目录
ROOT_DIR = os.path.abspath("../../")

# 导入 Mask RCNN
sys.path.append(ROOT_DIR)  # 查找库的本地版本
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# 训练后的权重文件的路径
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# 用于保存日志和模型检查点的目录（如果未提供）
# ＃通过命令行参数  --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"  # COCO2014

############################################################
#  Configurations
############################################################
class CocoConfig(Config):
    """MS COCO 训练的配置.
    派生自基本Config类，并覆盖特定于COCO 数据集的值.
    """
    # 为配置指定一个可识别的名称
    NAME = "coco"

    # 我们使用具有12GB内存的GPU，该内存可以容纳两个图像.
    # IMAGES_PER_GPU = 2
    # 如果使用较小的GPU，请向下调整.
    # 我们使用的是RTX3070，8G内存的GPU，该内存容纳单个图像
    IMAGES_PER_GPU = 1

    # 取消注释即可以在 8个GPU上训练 (默认值为 1)
    # GPU_COUNT = 8

    # 类别数目 (包括背景)
    NUM_CLASSES = 1 + 80  # COCO 有80个类


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """加载 COCO 数据集的子集.
        dataset_dir: COCO 数据集的根目录.
        subset: 加载什么？ (train, val, minival, valminusminival)
        year: 那个数据年份 (2014, 2017) 以字符串而不是整数的形式加载
        class_ids: 若提供, 仅加载具有给定类别的图像.
        class_map: TODO: 尚未执行.
                支持从不同数据集到相同类ID的映射类.
        return_coco: 若为真，则返回 COCO 对象.
        auto_download: 自动下载和解压缩 MS-COCO 图像和注释
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # 加载所有类或子集?
        if not class_ids:
            # 所有类别
            class_ids = sorted(coco.getCatIds())

        # 所有图像或子集?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # 删除重复项
            image_ids = list(set(image_ids))
        else:
            # 所有图像
            image_ids = list(coco.imgs.keys())

        # 新增类
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # 新增图片
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """若需要，则下载 COCO dataset/annotations .
        dataDir: COCO 数据集的根目录.
        dataType: 加载什么(train, val, minival, valminusminival)
        dataYear: 字符串（而不是整数）要加载的数据集年份（2014、2017）
                注意:
                    For 2014, use "train", "val", "minival", or "valminusminival"
                    For 2017, only "train" and "val" annotations are available
        """

        # 设置路径和文件名
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # 创建主文件夹(如果不存在)
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # 下载图像(如果本地不可用)
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # 设置注释数据路径
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # 下载注释(如果本地不可用)
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """加载给定图像的实例 mask .

        不同的数据集使用不同的方式存储蒙版.
        此函数以位图的形式将不同的掩码格式转换为一种格式 [height, width, instances]

        Returns:
        masks: 一个形状为[height，width，instance count]的布尔数组，每个实例具有一个 mask.
        class_ids: 实例mask 的类ID的一维数组
        """
        # 颗不是 COCO image, 则委托给父类.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # 构建形状为[height，width，instance_count]的mask以及
        # 与该mask的每个通道对应的类ID列表.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # 有些对象太小，以至于它们小于1像素面积，并且最终被舍入,跳过那些对象.
                if m.max() < 1:
                    continue
                # 是 crowd 吗? 若是则使用 negative class ID.
                if annotation['iscrowd']:
                    # 为 crowds 使用 negative class ID.
                    class_id *= -1
                    # 对于 crowd masks, annToMask() 有时会返回小于给定尺寸的mask.
                    # 若是这样，请调整大小.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # 将实例 masks 打包到数组中
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # 调用 super class 来返回空 mask.
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """ 返回 COCO 网站中图像的链接."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # 以下两个功能来自 pycocotools ，但有一些更改.

    def annToRLE(self, ann, height, width):
        """
        将注释（可以是未压缩的RLE的多边形）转换为RLE.
        return: 二进制 mask（numpy 2D数组）
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- 单个对象可能包含多个部分
            # 我们将所有部分合并到一个蒙版文件中
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # 解压缩 RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        将可以是多边形，未压缩的RLE或RLE的注释转换为二进制 mask
        return: 二进制 mask（numpy 2D数组）
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """安排结果以匹配 http://cocodataset.org/#format 中的COCO规范
    """
    # 如果无结果，则返回一个空列表
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # 遍历 detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """进行官方 COCO 评估.
    dataset: 具有验证数据的数据集对象
    eval_type: "bbox" or "segm" 用于 边界框 或 分割 评估
    limit: 若不为0，则是用于评估的图像数
    """
    # 从数据集中选择COCO图像
    image_ids = image_ids or dataset.image_ids

    # 得到一个限制的子集
    if limit:
        image_ids = image_ids[:limit]

    # 得到相应的 COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # 导入 image
        image = dataset.load_image(image_id)

        # 运行 detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # 将结果转换为COCO格式
        # 将蒙版转换为uint8，因为COCO工具在bool上出错
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # 加载结果. 这会使用其他属性修改结果.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # 将批处理大小设置为1，因为我们一次只处理一张图像
            # Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # 创建模型
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # 选择要加载的权重
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # 查找上一个训练好的权重
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # 从 ImageNet 训练的权重开始
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # 导入 weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # 训练 或 评估
    if args.command == "train":
        # 训练数据集.
        # 就像Mask RCNN论文一样，使用训练集和验证集中的35K.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # 验证数据集
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # 图像增强
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** 该培训表就是一个例子. 更新您的需求 ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    # epochs=40,
                    epochs=2,  # 测试用
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    # epochs=120,
                    epochs=4,  # 测试用
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    # epochs=160,
                    epochs=8,  # 测试用
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # 验证数据集
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
