# 模型测试(`/test.py`)
#### 参数
- `opt.batch_size:` 模型每次预测的batch大小(`8`)
- `opt.model_def:` 模型结构定义文件(`config/yolov3.cfg`)
- `opt.data_config:` 数据train, valid, test的相关定义文件(`config/coco.data`)
- `opt.weights_path:` 模型权重文件(`weights/yolov3.weights`)
- `opt.class_path:` 检测目标的种类(`data/coco.name`)
- `opt.iou_thres:` 判断预测框是否匹配真实框的IOU阈值(`0.5`)
- `opt.conf_thres:`
- `opt.nms_thres:`
- `opt.n_cpu:` dataloader是否使用多进程加载数据(`n_cpu=0:`单进程, `n_cpu>0:n_cpu`进程加载数据)
- `opt.img_size:` 模型接收的图片大小(`416`)

#### main()
- 通过`opt.model_def, opt.weights_path`得到要测试的模型`model`
- `precision, recall, AP, f1, ap_class = `[evaluate()](test.md#evaluate)
- `mAP = AP.mean()`

#### evaluate()
- 参数
  - `model: Darknet(opt.model_def).load_darknet_weights(opt.weights_path)`
  - `path: opt.data_config["valid"]`(`data/coco/5k.txt`)
  - `iou_thres: opt.iou_thres`(`0.5`)
  - `conf_thres: opt.conf_thres`(`0.01`)
  - `nms_thres: opt.nms_thres`(`0.5`)
  - `img_size: opt.img_size`(`416`)
  - `batch_size: 8`
- 代码细节
  - dataset = [utils.utils.ListDataset(path, img_size=img_size, augment=False, multiscale=False)][utils.utils.ListDataset]初始化`path`数据中的图片和真实框。

[utils.utils.ListDataset]:<>
