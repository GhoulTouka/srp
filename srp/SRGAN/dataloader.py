import tensorflow as tf
import os

from tensorflow.python.ops import array_ops, math_ops


class DataLoader(object):
    """SRGAN的数据加载器，为训练准备tf数据对象"""

    def __init__(self, image_dir, hr_image_size):
        """
        初始化 数据加载器
        参数:
            image_dir: 包含高分辨率图像的目录的 路径
            hr_image_size: 整数，待训练的图像的裁剪大小(高分辨率图像将裁剪为此宽度和高度）).
        Returns:
            The 数据加载器对象
        """
        self.image_paths = [os.path.join(image_dir, x)
                            for x in os.listdir(image_dir)]
        self.image_size = hr_image_size

    def _parse_image(self, image_path):
        """
        加载给定路径的图像的函数
        Args:
            image_path: 图像文件路径
        Returns:
            image: 加载图像的tf张量
        """

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # 检查图像是否足够大
        if tf.keras.backend.image_data_format() == 'channels_last':
            shape = array_ops.shape(image)[:2]
        else:
            shape = array_ops.shape(image)[1:]
        cond = math_ops.reduce_all(shape >= tf.constant(self.image_size))

        image = tf.cond(cond, lambda: tf.identity(image),
                        lambda: tf.image.resize(image, [self.image_size, self.image_size]))

        return image

    def _random_crop(self, image):
        """
        根据定义的宽度和高度裁剪图像的函数
        Args:
            image: 图像的tf张量
        Returns:
            image: 包含裁剪图像的tf张量
        """

        image = tf.image.random_crop(
            image, [self.image_size, self.image_size, 3])

        return image

    def _high_low_res_pairs(self, high_res):
        """
        给定高分辨率图像时生成低分辨率图像的函数，下采样系数是4
        Args:
            high_res: 高分辨率图像的tf张量
        Returns:
            low_res: 低分辨率图像的tf张量
            high_res: 高分辨率图像的tf张量
        """

        low_res = tf.image.resize(high_res,
                                  [self.image_size // 4, self.image_size // 4],
                                  method='bicubic')

        return low_res, high_res

    def _rescale(self, low_res, high_res):
        """
        将像素值缩放到-1到1范围的函数，与生成器出tanh函数一起使用
        Args:
            low_res: 低分辨率图像的tf张量
            high_res: 高分辨率图像的tf张量
        Returns:
            low_res: 重新缩放的低分辨率图像的tf张量
            high_res: 重新缩放的高分辨率图像的tf张量
        """
        high_res = high_res * 2.0 - 1.0

        return low_res, high_res

    def dataset(self, batch_size, threads=4):
        """
        返回具有指定映射的tf数据集对象
        Args:
            batch_size: 数据集返回的批次中的元素个数
            threads: 线程数.
        Returns:
            dataset: tf数据集对象
        """

        # 从高分辨率图像路径生成tf数据集
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)

        # 读取图像
        dataset = dataset.map(
            self._parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # 剪出一块来训练
        dataset = dataset.map(
            self._random_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # 通过下采样图像产生低分辨率。
        dataset = dataset.map(self._high_low_res_pairs,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # 重新缩放输入
        dataset = dataset.map(
            self._rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # 批处理输入，删除剩余部分以获得定义的批处理大小
        # 预取数据以优化GPU利用率
        dataset = dataset.shuffle(30).batch(
            batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset