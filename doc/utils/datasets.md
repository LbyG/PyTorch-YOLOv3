# `utils.datasets(utils/dataset.py)`

#### class ListDataset(torch.utils.data.Dataset)
##### 类变量
- `self.img_files:` 保存着图像路径(`string数组`)
- `self.label_files:` 保存着图像所对应标签的文件路径(`string数组`)
- `self.img_size:` 不进行多尺度缩放时的图片大小(`int`)
- `self.max_objects:` **没有被用到**
- `self.augment:` 是否进行数据增强(`bool`)
- `self.multiscale: `是否进行多尺度变换(`bool`)
- `self.normalized_labels: `是否将图片的`x, y, w, h`正则化到`[0, 1]`的范围内
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
