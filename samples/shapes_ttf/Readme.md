# 文件介绍：
* 文件夹 shapes_ttf：(仅提交一张，其余需从相应链接下载)
  * 数据集下载: `链接: https://pan.baidu.com/s/14dBd1Lbjw0FCnwKryf9taQ 提取码: 9457`
    * 数据集来源自B站UP：`https://www.bilibili.com/video/BV1CE411g78W?from=search&seid=15069132654799607424`
    * 项目来自：`https://github.com/bubbliiiing/mask-rcnn-keras`
  * 训练权重下载: `https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5`
    * 权重放在ROOT_DIR下
    * 使用COCO权重预训练

* train.py 是 shapes_ttf的训练文件

* predict.py 是 shapes_ttf的测试文件



# 项目来源：
* 链接：`https://github.com/matterport/Mask_RCNN`, `https://github.com/bubbliiiing/mask-rcnn-keras`
* 自制标注数据集使用


# 可视化：
* 在根目录输入：`tensorboard --logdir=training:logs/shapes20210322T2055` 即可打开相应的浏览器端口对训练得到的模型进行可视化
`
