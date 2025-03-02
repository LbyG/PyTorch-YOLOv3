# `utils.datasets(utils/dataset.py)`

#### def pad_to_square(img, pad_value)
- 功能：将图像`padding`填充成长宽一致的图像。
- 函数参数
  - `img:` `padding`前的图像tensor(`三维tensor数组，tensor([RGB, 高, 宽])`)
  - `pad_value: `用于`padding`时填充的值(`int`)
- 函数返回
  - `img:` `padding`后长宽一致的图像tensor(`三维tensor数组，tensor([RGB, 高, 宽])`)
  - `pad:` 将图像变成长宽一样时，需要设置的四维tuple，其中记录的是`(top_padding, bottom_padding, left_padding, right_padding)`
- 函数细节
  - `pad: `需要通过[torch.nn.functional.pad()][torch.nn.functional.pad]将图像变成长宽一样时，需要设置的四维tuple，其中记录的是`(top_padding, bottom_padding, left_padding, right_padding)`
  - `img = F.pad(img, pad, "constant", value=pad_value)`得到填充后的长宽一致的tensor

#### class ListDataset(torch.utils.data.Dataset)
##### 类变量
- `self.img_files:` 保存着图像路径(`string数组`)
- `self.label_files:` 保存着图像所对应标签的文件路径(`string数组`)
- `self.img_size:` 不进行多尺度缩放时的图片大小(`int`)
- `self.max_objects:` **没有被用到**
- `self.augment:` 是否进行数据增强(`bool`)
- `self.multiscale: `是否进行多尺度变换(`bool`)
- `self.normalized_labels: `标签数据中的坐标是否正则化到了`[0, 1]`的范围，如果有需要令`self.normalized_labels = True`，否则`self.normalized_labels = False`(`COCO`数据集标签的范围是`[0, 1]`)
- `self.min_size = self.img_size - 3 * 32:` 进行多尺度变换时，图片缩小的最小边界。
- `self.max_size = self.img_size + 3 * 32:` 进行多尺度变换时，图片放大的最大边界。
- `self.batch_count:` 用于统计`collate_fn`的次数，如果`self.multiscale == True`则每10次，即`self.batch_count % 10 == 0`成立时。改变`self.img_size`的大小。
##### `def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True)`
- 函数参数
  - `list_path:` 数据文件的路径，其中记录了每张图片的位置，如`/home/lbyg/object_detection/input/coco/images/val2014/COCO_val2014_000000000164.jpg`
  - `img_size:` 模型处理图片的大小。
  - `augment:` 是否进行数据增强。
  - `multiscale:` 是否进行不同尺度的缩放。
  - `normalized_labels:` 获得图片坐标的时候，是否正则化到`[0, 1]`的范围内。
- 代码细节
  - `self.img_files:` 读取`list_path`文件中保存所有的图像路径，如`/home/lbyg/object_detection/input/coco/images/val2014/COCO_val2014_000000000164.jpg`。
  - `self.label_files` 将图片路径中的`"images"`替换成`"labels"`,`".jpg"`替换成`".txt"`得到相关标签文件的路径，如`/home/lbyg/object_detection/input/coco/labels/val2014/COCO_val2014_000000000164.txt`。
  - `self.img_size = img_size(416)`
  - `self.max_objects = 100`
  - `self.augment = augment(True)`
  - `self.multiscale = multiscale(True)`
  - `self.normalized_labels = normalized_labels(True)`
  - `self.min_size = self.img_size - 3 * 32(320)`
  - `self.max_size = self.img_size + 3 * 32(512)`
  - `self.batch_count = 0`
##### `def __getitem__(self, index)`
- 功能：获得`index`对应的图像和真实目标框信息。
- 函数参数
  - `index:` 要获取图片和标注的下标。
- 函数返回
  - `img_path:` 图片的所在路径(`string`)
  - `img:` `padding`后长宽一致的图像tensor(`三维tensor数组，tensor([RGB, max(高, 宽), max(高, 宽)])`)
  - `targets:` 图片对应的真实目标框(`二维tensor数组，tensor([目标框数, 6])。targets[i] = tensor([0, 目标类别id, 中心点的横坐标, 中心点的纵坐标, 宽, 高])，且值在[0, 1]的范围内`)
- 代码细节
  - `img:` 读取图像数据到`img`当中，如果`img`是黑白图像(即`len(img.shape) == 2`)，则需要将其扩展为`RGB`三通道的形式(`三维tensor数组，tensor([RGB, 高, 宽])`)
  - `img, pad = pad_to_square(img, 0): `如果图像的长宽不相等，则通过`padding`操作填充`0`，将图像转化为长宽相等的图像(`三维tensor数组，tensor([RGB, max(高, 宽), max(高, 宽)])`)
  - `boxes:` 读取标签框到`boxes`中，并将标签框的`x, y, w, h`信息转化成`padding`过后的，且进行正则化操作，即值的范围在`[0, 1]`中。
  - `targets: ``target`为`tensor([n, 6])`的`tensor`变量，`target[:, 0]`留着之后保存`batch`中的图片`id`，`target[:, 0] = boxes`存储真实目标框信息。
  - 如果`self.augment == True`，则`50%`的概率调用[utils.utils.horisontal_flip()][utils.utils.horisontal_flip]水平翻转图片和目标框。
  
##### `def collate_fn(self, batch)`
- 功能：[torch.utils.data.dataloader]的到一组图像和标签数据后，会调用该函数将图像和标签数据分别转换成一个tensor:
- 函数参数
  - `batch:` 用一个长度为`self.batch_size`的数组保存了图片位置，图片信息，图片对应的真实检测框(`list(tuple(path, img, target))`)
- 函数返回
  - `paths:` 保存着图像路径的`string数组`(`一维string数组，string[batch_size]`)
  - `imgs:` batch个图像的信息，如果`self.multiscale == True`则图像会经过多尺寸缩放。(`三维tensor数组，tensor([RGB * batch_size, self.img_size, self.img_size])`)
  - `targets:` batch个图像对应的真实目标框信息(`二维tensor数组，tensor([sum_Batch(目标框数), 6])。targets[i] = tensor([图像id, 目标类别id, 中心点的横坐标, 中心点的纵坐标, 宽, 高])。图像id的值在[0, batch_size-1]的范围内，其他值在[0, 1]的范围内`)
- 代码细节
  - `for i, boxes in enumerate(targets): boxes[:, 0] = i`将boxes[i]的第一个维度设置为图像id。
  - 如果`self.multiscale == True`，则每10个`batch`会进行多尺寸采样，即`self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))`
  
[torch.utils.data.dataloader]:<https://pytorch.org/docs/stable/data.html>
[torch.nn.functional.pad]:<https://pytorch.org/docs/stable/nn.functional.html>
[utils.utils.horisontal_flip]:<augmentations.md#def-horisontal_flipimages-targets>
