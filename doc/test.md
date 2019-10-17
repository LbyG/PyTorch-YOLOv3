# 模型测试(`/test.py`)
#### 参数
- `opt.batch_size:` 模型每次预测的batch大小(`8`)
- `opt.model_def:` 模型结构定义文件(`config/yolov3.cfg`)
- `opt.data_config:` 数据train, valid, test的相关定义文件(`config/coco.data`)
- `opt.weights_path:` 模型权重文件(`weights/yolov3.weights`)
- `opt.class_path:` 检测目标的种类(`data/coco.name`)
- `opt.iou_thres:` 判断预测框是否匹配真实框的IOU阈值(`0.5`)
- `opt.conf_thres:` `预测框置信度 < opt.conf_thres`则直接筛除(`0.001`)
- `opt.nms_thres:` `NMS`时，`iou > opt.nms_thres`的边界框会被抑制(`0.5`)
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
  - `dataset =` [utils.datasets.ListDataset(path, img_size=img_size, augment=False, multiscale=False)][utils.datasets.ListDataset]初始化`path`数据中的图片和真实框。
  - `dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)`。遍历`dataloader`，每次会返回`batch_size`个图像和相关标注的信息。
  - `for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):`
    - `imgs:` `batch`个图片的信息(`四维tensor数组，tensor([batch_size, RGB, max(高, 宽), max(高, 宽)])`)
    - `targets:` `batch`个图片对应的真实目标框信息(`三维tensor数组，tensor([batch_size, 目标框数, 6])。targets[batch_i, bbox_i] = tensor([batch_i, 目标类别id, center_x, center_y, width, height])，且值在[0, 1]的范围内`)
    - `label:` 保存图片目标框的种类信息(`list类型，len(label) = 所有图像的真实目标框总数`)
    - `target:` 真实目标框的`(center_x, center_y, width, height)`信息转化成`(x1, y1, x2, y2)`信息，并且由`[0, 1]`的范围转化为`[0, img_size]`
    - `outputs = model(imgs):` `yolov3`会输出`13*13, 26*26, 52*52`的特征矩阵，特征矩阵每个cell会预测3个`anchor`，每个`anchor`是`85`维向量，`[0:4]`为是预测框的`x, y, w, h`, `[5]`是置信度, `[5:85]`是对类别的预测(`三维tensor, torch.Size([8, 10647(13*13*3 + 26*26*3 + 52*52*3), 85(5 + 80)])`) 
    - `outputs = `[utils.utils.non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)][utils.utils.non_max_suppression]，非极大值抑制筛选后得到，置信度从大到小的预测框(`list(batch_size * tensor([m, 7(x1, y1, x2, y2, c, class_conf, class_pred)]))`)

[utils.datasets.ListDataset]:<utils/datasets.md#def-__init__self-list_path-img_size416-augmenttrue-multiscaletrue-normalized_labelstrue>
[utils.utils.non_max_suppression]:<utils/utils.md#def-non_max_suppressionprediction-conf_thres05-nms_thres04>
