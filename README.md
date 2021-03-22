# Mask_RCNN-win
* Win10, RTX3070, CUDA11.1, cudnn8.0.5, keras2.4.3, tensorflow2.4.0, Mask_RCNN
* code从这改写 `https://github.com/matterport/Mask_RCNN`

# environment configuration：
* `https://github.com/huitudou/Environment_configuration`

# 文件介绍
* assets -- `Readme.md`中使用的图片所在目录
* BEFORE_DATASET -- 原始标注的数据文件, 例如 0.png, 0.json
* DATASET -- 经`json_to_dataset`处理后的文件存放目录, imgs, mask, yaml, label_viz
* images -- 一些示例文件，在`demo.py`中使用
* logs -- 保存日志和训练的模型的目录
* mrcnn -- ------------------------------------------
*     __init__.py  --  构成python库
*     config.py  --  基本配置类
*     model.py  -- Mask R-CNN模型的主要实现
*     parallel_model.py  --  对Keras的多GPU支持
*     utils.py  --  通用实用程序功能和类
*     visualize.py  --  显示和可视化功能
* samples -- ----------------------------------------
*     balloon
*     coco
*     kangaroo
*     nucleus
*     shapes
*     shapes_ttf
*     demo.ipynb
* json_to_dataset  --  将 Labelme 标注的文件转化为几个文件用于MASK R-CNN 训练与预测

# 
