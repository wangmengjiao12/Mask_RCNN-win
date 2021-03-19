# 文件介绍：
* 文件夹 train, val：
  * 数据集下载：`https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip`
    * 数据集解压放在samples/balloon下
  * 训练权重下载: `https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5`
    * 权重放在ROOT_DIR下

* balloon.py 是 balloon的训练预测python文件，在Edit Configuration 键入(做训练，预测)：
  * train  --dataset   "E:\\Mask_RCNN-tf2\\samples\\balloon"  --weights coco
  * splash  --weights  "E:\\Mask_RCNN-tf2\\logs\\balloon20210318T1851\\mask_rcnn_balloon_0030.h5"  --image  "E:\\Mask_RCNN-tf2\\samples\\balloon\\train\\120853323_d4788431b9_b.jpg"

* inspect_balloon_data.ipynb 是balloon的data的逐步解析代码
* inspect_balloon_model.ipynb 是balloon的model的逐步解析代码

# 项目来源：
* 链接：`https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46`
* A blog post explaining how to train this model from scratch and use it to implement a color splash effect
