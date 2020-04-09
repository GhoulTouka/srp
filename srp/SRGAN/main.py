from argparse import ArgumentParser
from dataloader import DataLoader
from model import FastSRGAN
import tensorflow as tf
import os

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str,
                    help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training.')
parser.add_argument('--epochs', default=1, type=int,
                    help='Number of epochs for training')
parser.add_argument('--hr_size', default=384, type=int,
                    help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')


@tf.function
def pretrain_step(model, x, y):
    """
    对生成器进行单步预训练
    参数:
        model: 带有tf keras编译的生成器的模型对象
        x: 低分辨率图像张量
        y: 高分辨率图像张量
    """
    with tf.GradientTape() as tape:
        fake_hr = model.generator(x)
        loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)

    grads = tape.gradient(loss_mse, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(
        zip(grads, model.generator.trainable_variables))

    return loss_mse


def pretrain_generator(model, dataset, writer):
    """稍微对生成器进行预训练的函数, 以避免局部极小值
    参数:
        model: 待训练的keras模型
        dataset: 用于预训练的tf数据集对象(低分辨率图像 + 高分辨率图像)
        writer: 1个writer对象(总结性).
    返回值:
        无
    """
    with writer.as_default():
        iteration = 0
        for _ in range(1):
            try:
                for x, y in dataset:
                    loss = pretrain_step(model, x, y)
                    if iteration % 20 == 0:
                        tf.summary.scalar('MSE Loss', loss,
                                          step=tf.cast(iteration, tf.int64))
                        writer.flush()
                    iteration += 1
            except:
                print("iteration", iteration)


@tf.function
def train_step(model, x, y):
    """单步对SRGAN进行训练
    参数:
        model: 包含tf keras编译的鉴别器模型的对象
        x: 输入的低分辨率图像
        y: 所需的输出(高分辨率图像)

    返回值:
        d_loss: 鉴别器的平均loss
    """
    # Label 平滑以求更好的梯度流
    valid = tf.ones((x.shape[0],) + model.disc_patch)
    fake = tf.zeros((x.shape[0],) + model.disc_patch)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 用低分辨率图片生成高分辨率图片
        fake_hr = model.generator(x)

        # 训练鉴别器 (original images = real / generated = Fake)
        valid_prediction = model.discriminator(y)
        fake_prediction = model.discriminator(fake_hr)

        # 生成器 loss
        content_loss = model.content_loss(y, fake_hr)
        adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
        mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
        perceptual_loss = content_loss + adv_loss + mse_loss

        # 鉴别器 loss
        valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
        fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
        d_loss = tf.add(valid_loss, fake_loss)

    # 生成器上的反向传播
    gen_grads = gen_tape.gradient(
        perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(
        zip(gen_grads, model.generator.trainable_variables))

    # 鉴别器上的反向传播
    disc_grads = disc_tape.gradient(
        d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(
        zip(disc_grads, model.discriminator.trainable_variables))

    return d_loss, adv_loss, content_loss, mse_loss


def train(model, dataset, log_iter, writer):
    """
    定义了对SR-GAN进行单步训练的函数
    参数:
        model: 包含tf keras编译的生成器和鉴别器的对象
        dataset: 包含低分辨率图像和高分辨率图像的tf数据对象
        log_iter: 向tensorboard添加日志(迭代次数)
        writer: 1个writer对象(总结性).
    """
    with writer.as_default():
        # 在数据集上迭代
        for x, y in dataset:
            disc_loss, adv_loss, content_loss, mse_loss = train_step(
                model, x, y)
            # 每迭代1轮log_iter，将结果保存
            if model.iterations % log_iter == 0:
                tf.summary.scalar('Adversarial Loss',
                                  adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss,
                                  step=model.iterations)
                tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss',
                                  disc_loss, step=model.iterations)
                tf.summary.image('Low Res', tf.cast(
                    255 * x, tf.uint8), step=model.iterations)
                tf.summary.image('High Res', tf.cast(
                    255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                model.generator.save('models/generator.h5')
                model.discriminator.save('models/discriminator.h5')
                writer.flush()
            model.iterations += 1


def main():
    # 分析/处理CLI参数.
    args = parser.parse_args()

    # 创建目录以存储训练的模型
    if not os.path.exists('models'):
        os.makedirs('models')

    # 创建tensorflow数据集
    ds = DataLoader(args.image_dir, args.hr_size).dataset(args.batch_size)

    # 初始化GAN对象
    gan = FastSRGAN(args)

    # 定义保存预训练loss tensorboard摘要的目录
    pretrain_summary_writer = tf.summary.create_file_writer('logs/pretrain')

    # 执行预训练
    pretrain_generator(gan, ds, pretrain_summary_writer)

    # 定义保存SRGAN训练tensorbaord摘要的目录
    train_summary_writer = tf.summary.create_file_writer('logs/train')

    # 执行训练
    for _ in range(args.epochs):
        train(gan, ds, args.save_iter, train_summary_writer)


if __name__ == '__main__':
    main()