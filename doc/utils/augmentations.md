# `utils.augmentations(utils/augmentations.py)`
#### def horisontal_flip(images, targets)
- 功能：水平翻转图像和目标框
- 函数参数
  - `images:` 图像的tensor表示(`三维tensor数组，tensor([RGB, 高, 宽])`)
  - `targets:` 图像中目标框信息(`二维tensor数组，tensor([目标框数, 5])。targets[i] = tensor([置信度, 中心点的纵坐标, 中心点的横坐标, 高, 宽])，且值在[0, 1]的范围内`)
