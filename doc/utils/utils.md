# `utils.utils(utils/utils.py)`
#### `def get_batch_statistics(outputs, targets, iou_threshold)`
- 功能：参考[COCO mAP]的计算方法，计算`tp, fp, 预测置信度和预测类别id`
- 函数参数
  - `outputs:` `batch`个图片经过`NMS`过滤后的预测框(`list(batch_size * tensor([m, 7(x1, y1, x2, y2, c, class_conf, class_pred)]))`)
  - `targets:` `batch`个图片的真实目标框(`三维tensor数组，tensor([batch_size, 目标框数, 6])。targets[batch_i, bbox_i] = tensor([batch_i, 目标类别id, x1, y1, x2, y2])，且值在[0, img_size]的范围内`)
  - `iou_threshold:` 如果预测目标框与真实目标框之间的`IOU > iou_threshold`则认为预测匹配了相关真实目标框(`int`)
- 函数返回
  - `batch_metrics:` 类型为`list(batch_size * list[true_positives, pred_scores, pred_labels])`。
    - `true_positives:` 记录了预测框是否正确。如果`true_positives[pred_i] == 1`，则表明预测框`pred_id`正确检测了目标，否则为错误预测(`list(m * int)`)
    - `pred_score:` 记录了预测框的置信度，该向量不一定是从大到小有序的，因为是按pred_score*label_score从大到小排序的(`tensor([m])`)
    - `pred_labels:` 记录了预测框检测到的类别(`tensor([m])`)

#### `def bbox_iou(box1, box2, x1y1x2y2=True)`
- 功能：计算`box1`与`box2`之间的`IOU`
- 函数参数
  - `box1:` 目标框1信息(`box1.shape = torch.Size([1, 4])`)
  - `box2:` 目标框2信息(`box2.shape = torch.Size([n, 4])`)
  - `x1y1x2y2:` 目标框的信息是否是`(x1, y1, x2, y2)`的形式，如果不是，会转化为相应形式
- 函数返回
  - `iou:` 目标框1与目标框2之间的IOU值(`iou.shape = torch.Size([n, 1])`)
- 代码细节
  - `b1_x1, b1_y1, b1_x2, b1_y2:` 目标框1的左上角点和右下角点信息。
  - `b2_x1, b2_y1, b2_x2, b2_y2:` 目标框2的左上角点和右下角点信息。
  - `inter_area:` 目标框1与目标框2之间的交叉面积(`inter_area.shape = torch.Size([n, 1])`)
  - `b1_area:` 目标框1的面积(`b1_area.shape = torch.Size([n, 1])`)
  - `b2_area:` 目标框2的面积(`b2_area.shape = torch.Size([n, 1])`)
  - `iou:` 目标框1与目标框2之间的IOU值(`iou.shape = torch.Size([n, 1])`)

#### `def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4)`
- 功能：非极大值抑制，过滤掉部分预测框。
- 函数参数
  - `prediction:` 模型的预测框结果(`三维tensor, torch.Size([8, 10647(13*13*3 + 26*26*3 + 52*52*3), 85(5 + 80)])`)
  - `conf_thres:` `预测框置信度 < opt.conf_thres`则直接筛除(`0.001`)
  - `nms_thres:` `NMS`时，`iou > opt.nms_thres`的边界框会被抑制(`0.5`)
- 函数返回
  - `output:` 非极大值抑制筛选后得到，置信度从大到小的预测框(`list(batch_size * tensor([m, 7(x1, y1, x2, y2, c, class_conf, class_pred)]))`)
- 代码细节
  - 将预测框的`(center_x, center_y, width, height)`信息转化成`(x1, y1, x2, y2)`信息
  - `image_pred:` 遍历`batch`的预测框`prediction`，得到每个图的预测框`image_pred`(`image_pred.shape = torch.Size([10647, 85])`)
    - `image_pred:` 过滤掉置信度小于conf_thres的预测框(`image_pred.shape = torch.Size([n(n<10647), 85])`)
    - `score:` 置信度*最大类别预测值得到最终预测框的置信度(`score.shape = torch.Size([n, 1])`)
    - `image_pred:` 根据`score`从大到小排序。
    - `class_confs:` 类别预测值中最大的值(`class_confs.shape = torch.Size([n, 1])`)
    - `class_preds:` 类别预测值中最大值的下标(`class_preds.shape = torch.Size([n, 1])`)
    - `detections:` 拼接预测框，类别置信度，类别id(`class_preds.shape = torch.Size([n, 7(x1, y1, x2, y2, c, class_conf, class_pred)])`)
    - `keep_boxes:` 一张图片经过`NMS`后的预测框(`keep_boxes.shape = torch.Size([m(m<n), 7])`)
    - 持续循环直到`detections`为空：
      - `detections[0]`加入到`keep_boxes`中
      - `large_overlap:` `detection`中与`detections[0]`的`iou`值是否大于`nms_thres`
      - `label_match:` `detection`的类别是否与`detections[0]`相同
      - `detections:` 剔除掉`detections`中与`detections[0]`类别相同且`iou > nms_thres`的`detection`
    - `output[image_i] = keep_boxes:` 保存不同图片的最终预测框

[COCO mAP]:<https://github.com/LbyG/MOT-Paper-Notes/blob/master/evaluate-metric.md#map%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B>
