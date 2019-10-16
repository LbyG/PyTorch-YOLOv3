# 模型测试(`/test.py`)
#### 参数
- `batch_size`: 模型每次预测的batch大小(`8`)
- `model_def`: 模型结构定义文件(`config/yolov3.cfg`)
- `data_config`：数据train, valid, test的相关定义文件(`config.coco.data`)
- `weights_path`: 模型权重文件(`weights/yolov3.weights`)
- `class_path`: 检测目标的种类(`data/coco.name`)
- `iou_thres`: 判断预测框是否匹配真实框的IOU阈值(`0.5`)
- `conf_thres`:
- `nms_thres`:
- `n_cpu`: dataloader是否使用多进程加载数据(`n_cpu=0:`单进程, `n_cpu>0:n_cpu`进程加载数据)
- `img_size`: 模型接收的图片大小(`416`)

#### main
- 通过`opt.model_def, opt.weights_path`得到要测试的模型`model`
- `precision, recall, AP, f1, ap_class = `[evaluate()](####evaluate)
- `mAP = AP.mean()`

#### evaluate
