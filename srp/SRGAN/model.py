from tensorflow import keras
import tensorflow as tf


class FastSRGAN(object):
    """用于快速超分辨率的SRGAN"""

    def __init__(self, args):
        """
        初始化 Mobile SRGAN.
        参数:
            建立模型的CLI参数
        返回:
            None
        """
        self.hr_height = args.hr_size
        self.hr_width = args.hr_size
        self.lr_height = self.hr_height // 4  # 低分辨率高度
        self.lr_width = self.hr_width // 4  # 低分辨率宽度
        self.lr_shape = (self.lr_height, self.lr_width, 3)
        self.hr_shape = (self.hr_height, self.hr_width, 3)
        self.iterations = 0

        # mobilenet生成器中的倒转残差块的数量
        self.n_residual_blocks = 6

        # 定义学习率的衰减时间表
        self.gen_schedule = keras.optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=100000,
            decay_rate=0.1,
            staircase=True
        )

        self.disc_schedule = keras.optimizers.schedules.ExponentialDecay(
            args.lr * 5,  # TTUR - 2个时间尺度的更新规则
            decay_steps=100000,
            decay_rate=0.1,
            staircase=True
        )

        self.gen_optimizer = keras.optimizers.Adam(
            learning_rate=self.gen_schedule)
        self.disc_optimizer = keras.optimizers.Adam(
            learning_rate=self.disc_schedule)

        # 用预训练的VGG19模型从高分辨率图像和生成的高分辨率图像中提取图像特征，并最小化它们之间的mse
        self.vgg = self.build_vgg()
        self.vgg.trainable = False

        # 计算D的输出尺寸大小 (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # G和D第一层中的滤波器数量
        self.gf = 32  # 实时图像增强GAN Galteri等
        self.df = 32

        # 构建并编译鉴别器
        self.discriminator = self.build_discriminator()

        # 建立并编译预训练的生成器
        self.generator = self.build_generator()

    @tf.function
    def content_loss(self, hr, sr):
        sr = keras.applications.vgg19.preprocess_input(
            ((sr + 1.0) * 255) / 2.0)
        hr = keras.applications.vgg19.preprocess_input(
            ((hr + 1.0) * 255) / 2.0)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return tf.keras.losses.MeanSquaredError()(hr_features, sr_features)

    def build_vgg(self):
        """
        建立一个预先训练的VGG19模型，其输出为在模型的第三个块提取的图像特征
        """
        # 获取vgg网络，从块5提取特征，最后卷积
        vgg = keras.applications.VGG19(
            weights="imagenet", input_shape=self.hr_shape, include_top=False)
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False

        # 创建模型并编译
        model = keras.models.Model(
            inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

        return model

    def build_generator(self):
        """构建将执行超分辨率任务的生成器。基于Mobilenet设计和来自Galteri等的想法。"""

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # 确保四舍五入的降幅不超过10%
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def residual_block(inputs, filters, block_id, expansion=6, stride=1, alpha=1.0):
            """倒转残差快 使用深度卷积提高参数效率
            参数:
                inputs: 输入feature map.
                filters: 块中每个卷积中的滤波器数
                block_id: 图像内块的整数id.
                expansion: 通道膨胀系数
                stride: 卷积的步长
                alpha: 深度膨胀系数
            返回:
                x: 倒转残差快的输出
            """
            channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

            in_channels = keras.backend.int_shape(inputs)[channel_axis]
            pointwise_conv_filters = int(filters * alpha)
            pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
            x = inputs
            prefix = 'block_{}_'.format(block_id)

            if block_id:
                # 扩张
                x = keras.layers.Conv2D(expansion * in_channels,
                                        kernel_size=1,
                                        padding='same',
                                        use_bias=True,
                                        activation=None,
                                        name=prefix + 'expand')(x)
                x = keras.layers.BatchNormalization(axis=channel_axis,
                                                    epsilon=1e-3,
                                                    momentum=0.999,
                                                    name=prefix + 'expand_BN')(x)
                x = keras.layers.Activation(
                    'relu', name=prefix + 'expand_relu')(x)
            else:
                prefix = 'expanded_conv_'

            # 纵向
            x = keras.layers.DepthwiseConv2D(kernel_size=3,
                                             strides=stride,
                                             activation=None,
                                             use_bias=True,
                                             padding='same' if stride == 1 else 'valid',
                                             name=prefix + 'depthwise')(x)
            x = keras.layers.BatchNormalization(axis=channel_axis,
                                                epsilon=1e-3,
                                                momentum=0.999,
                                                name=prefix + 'depthwise_BN')(x)

            x = keras.layers.Activation(
                'relu', name=prefix + 'depthwise_relu')(x)

            # 投射
            x = keras.layers.Conv2D(pointwise_filters,
                                    kernel_size=1,
                                    padding='same',
                                    use_bias=True,
                                    activation=None,
                                    name=prefix + 'project')(x)
            x = keras.layers.BatchNormalization(axis=channel_axis,
                                                epsilon=1e-3,
                                                momentum=0.999,
                                                name=prefix + 'project_BN')(x)

            if in_channels == pointwise_filters and stride == 1:
                return keras.layers.Add(name=prefix + 'add')([inputs, x])
            return x

        def deconv2d(layer_input, filters):
            """上采样层增加输入的高度和宽度。使用PixelShuffle进行上采样。
            Args:
                layer_input: 上采样的输入张量
                filters: 扩张滤波器数量
            Returns:
                u: 将输入上采样2倍
            """
            u = keras.layers.Conv2D(
                filters, kernel_size=3, strides=1, padding='same')(layer_input)
            u = tf.nn.depth_to_space(u, 2)
            u = keras.layers.PReLU(shared_axes=[1, 2])(u)
            return u

        # 输入的低分辨率图像
        img_lr = keras.Input(shape=self.lr_shape)

        # 前-残差快
        c1 = keras.layers.Conv2D(
            self.gf, kernel_size=3, strides=1, padding='same')(img_lr)
        c1 = keras.layers.BatchNormalization()(c1)
        c1 = keras.layers.PReLU(shared_axes=[1, 2])(c1)

        # 通过残差快传播
        r = residual_block(c1, self.gf, 0)
        for idx in range(1, self.n_residual_blocks):
            r = residual_block(r, self.gf, idx)

        # 后-残差快
        c2 = keras.layers.Conv2D(
            self.gf, kernel_size=3, strides=1, padding='same')(r)
        c2 = keras.layers.BatchNormalization()(c2)
        c2 = keras.layers.Add()([c2, c1])

        # 上采样
        u1 = deconv2d(c2, self.gf * 4)
        u2 = deconv2d(u1, self.gf * 4)

        # 生成高分辨率输出
        gen_hr = keras.layers.Conv2D(
            3, kernel_size=3, strides=1, padding='same', activation='tanh')(u2)

        return keras.models.Model(img_lr, gen_hr)

    def build_discriminator(self):
        """基于SRGAN设计构建鉴别器的网络"""

        def d_block(layer_input, filters, strides=1, bn=True):
            """鉴别器层块
            参数:
                layer_input: 卷积块的输入特征映射
                filters: 卷积中的滤波器数
                strides: 卷积的步长
                bn: 是否使用批量标准
            """
            d = keras.layers.Conv2D(
                filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = keras.layers.BatchNormalization(momentum=0.8)(d)
            d = keras.layers.LeakyReLU(alpha=0.2)(d)

            return d

        # 输入图像
        d0 = keras.layers.Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df)
        d4 = d_block(d3, self.df, strides=2)
        d5 = d_block(d4, self.df * 2)
        d6 = d_block(d5, self.df * 2, strides=2)
        d7 = d_block(d6, self.df * 2)
        d8 = d_block(d7, self.df * 2, strides=2)

        validity = keras.layers.Conv2D(
            1, kernel_size=1, strides=1, activation='sigmoid', padding='same')(d8)

        return keras.models.Model(d0, validity)