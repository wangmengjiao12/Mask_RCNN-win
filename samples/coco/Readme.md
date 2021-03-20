# 文件介绍：
* 文件夹 coco：(仅提交一张，其余需从相应链接下载)
  * 数据集下载：详见 `https://github.com/huitudou/Environment_configuration`
    * 数据集解压放在samples/coco/dataset下
  * 训练权重下载: `https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5`
    * 权重放在ROOT_DIR下

* coco.py 是 coco的训练预测python文件，在Edit Configuration 键入(做训练，预测)：
  * evaluate  --dataset "E:\\Mask_RCNN-tf2\\samples\\coco\\dataset"  --model "E:\\Mask_RCNN-tf2\\mask_rcnn_coco.h5"
  * train  --dataset "E:\\Mask_RCNN-tf2\\samples\\coco\\dataset"  --model "E:\\Mask_RCNN-tf2\\mask_rcnn_coco.h5"

* inspect_data.ipynb 是coco的data的逐步解析代码
* inspect_model.ipynb 是coco的model的逐步解析代码
* inspect_weights.ipynb 是coco的weights的逐步解析代码

# 项目来源：
* 链接：`https://github.com/matterport/Mask_RCNN`
* Training on MS COCO
