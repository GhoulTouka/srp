from argparse import ArgumentParser
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import os
import time

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')


def main():
    args = parser.parse_args()

    # 获得所有图像的路径
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    # 更改模型输入shape以接受所有size的输入
    model = keras.models.load_model('models/generator.h5')
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)

    i = 0
    start_time = []
    start_time.append(time.time())
    # print(start_time[0])

    # 遍历所有图像
    for image_path in image_paths:
        if (i % 100 == 0):
            print("{}\t".format(i))

        # 读取图像
        low_res = cv2.imread(image_path, 1)

        # if low_res.empty():
        #     break
        i += 1

        # 图像转为RGB (opencv默认使用BGR)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # 重塑为0~1之间.
        low_res = low_res / 255.0

        # 获得超分辨率图像
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        # 将值重塑在0~255范围内
        sr = ((sr + 1) / 2.) * 255

        # 将图像转回BGR一遍opencv方便操作
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        # 保存结果(输出文件):
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), sr)

        # start_time.append(time.time())

    time_used = time.time() - start_time[0]
    print()
    print("Time used: \t", time_used)
    print("#images: \t", i)
    #print("FPS: \t\t", i/time_used)

    # for i in range(10):
    #     print(start_time[i], sep="\t")


if __name__ == '__main__':
    main()