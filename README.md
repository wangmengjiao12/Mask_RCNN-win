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
*     balloon  --  气球色彩飞溅，目前可完美运行
*     coco  --  coco数据集训练测试，目前仅对于跑几十个epoch可用，过大硬件out of memory
*     kangaroo  --  分割出袋鼠，目前可训练测试，但是测试结果不理想，未知原因
*     nucleus  --  有2个bug存在， 但整体可使用
*     shapes  --  分割三角，方形吗圆形， 目前可以完美运行
*     shapes_ttf  -- 使用labelme自制图形数据集， 分割三角，方形吗圆形， 目前可以完美运行
*     demo.ipynb  --  MASK R_CNN的基础测试，目前可完美运行
* json_to_dataset  --  将 Labelme 标注的文件转化为几个文件用于MASK R-CNN 训练与预测

# 
