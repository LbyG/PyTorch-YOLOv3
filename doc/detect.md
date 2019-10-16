# 检测图像(`/detect.py`)
#### 参数
- `image_folder`: 图像的文件夹路径(`data/samples`)
- `model_def`: 模型结构定义文件(`config/yolov3.cfg`)
- `weights_path`: 模型权重文件(`weights/yolov3.weights`)
- `class_path`: 检测目标的种类(`data/coco.name`)
- `conf_thres`:
- `nms_thres`:
- `batch_size`: 模型每次预测的batch大小(`1`)
- `n_cpu`: dataloader是否使用多进程加载数据(`n_cpu=0:`单进程, `n_cpu>0:n_cpu`进程加载数据)
- `img_size`: 模型接收的图片大小(`416`)
