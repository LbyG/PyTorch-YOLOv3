# `model(model.py)`

#### class YOLOLayer(torch.nn.Module)
##### 类变量`(__init__(self, anchors, num_classes, img_dim=416))`
- `self.anchors:` 特征图的锚
- `self.num_anchors:` 锚的数量
- `self.num_classes:` 类别数量
- `self.ignore_thres:` TODO
- `self.mse_loss:` 均方误差损失函数`(x_i - y_i)^2`
- `self.bce_loss:` 交叉熵损失函数`(- x * log(y) - (1 - x) * log(1 - y))`
- `self.obj_scale:` 损失函数中有目标的权重
- `self.noobj_scale:` 损失函数中没有目标的权重
- `self.metrics:` 记录模型运行中的参数
- `self.img_dim:` 图片尺寸
- `self.grid_size:` 特征图尺寸

##### `def compute_grid_offsets(self, grid_size, cuda=True)`
- 功能：更新不同特征图尺寸下的参数。
- 函数参数
  - `grid_size:` 特征图尺寸。
  - `cuda:` 是否使用`cuda`加速
- 代码细节
  - `self.gird_size = grid_size:` 更新特征图尺寸。
  - `self.stride = img_dim / grid_size:` 更新特征图中每个格子的像素值。
  - `self.grid_x:` 特征图中格子的横向偏移像素值`(self.grid_x[:, x, y, :] = x)`。
  - `self.grid_y:` 特征图中格子的纵向偏移像素值`(self.grid_x[:, x, y, :] = y)`。
  - `self.scaled_anchors = anchors / stride:` 更新计算锚在特征图下边长`([0, grid_size])`
  - `self.anchor_w:` 锚的宽
  - `self.anchor_h:` 锚的高
  

##### `def forward(self, x, targets=None, img_dim=None)`
- 功能：对Yolov3的最后一层进行处理，返回预测目标框和损失。
- 函数参数
  - `x:` Yolov3最后一层的特征图。
  - `targets:` 应该输出的真实目标框。
  - `img_dim:` 图片像素尺寸。
- 参数返回
  - `output:` 预测结果目标框。
  - `total_loss:` 损失值。
- 代码细节
  - `grid_size:` 特征图尺寸
  - `prediction:` 调整Yolov3最后一层特征图的形式`(图片数量, 锚数量， 预测向量(类别数量+5), 特征图尺寸, 特征图尺寸)`
  - `x = torch.sigmoid(prediction[..., 0]):` 预测框左上角横坐标偏移`(值的范围在[0, 1]之间)`
  - `y = torch.sigmoid(prediction[..., 1]):` 预测框左上角纵坐标偏移`(值的范围在[0, 1]之间)`
  - `w = prediction[..., 2]:` 预测框的宽偏移`(之后需要进行(exp(w) * anchor_w) * stride缩放)`
  - `h = prediction[..., 3]:` 预测框的高偏移`(之后需要进行(exp(h) * anchor_h) * stride缩放)`
  - `pred_boxes[..., 0] = (x + grid_x) * stride:` 预测框在图片中左上角的横坐标`(值的范围在[0, img_size]之间)`
  - `pred_boxes[..., 1] = (y + grid_y) * stride:` 预测框在图片中左上角的纵坐标`(值的范围在[0, img_size]之间)`
  - `pred_boxes[..., 2] = (exp(w) * anchor_w) * stride:` 预测框在图片中的宽`(宽是在[0, img_size]坐标下)`
  - `pred_boxes[..., 3] = (exp(h) * anchor_h) * stride:` 预测框在图片中的高`(高是在[0, img_size]坐标下)`
  - 如果`target is None`
    - `return output, 0`
  - 否则，计算损失函数
    - `iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets:` 调用[utils.build_targets]()将真实目标框与网络特征图进行匹配, 得到相关参数
  
