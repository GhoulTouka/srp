import cv2


class Anime4K(object):
    def __init__(
        self, passes=1, strengthColor=1/6, strengthGradient=1/2, fastMode=False,
    ):
        # 执行次数
        self.ps = passes
        # strengthColor范围[0, 1]，越大线条越细薄
        self.sc = strengthColor
        # strengthGradient范围[0, 1], 越大锐化度越高
        self.sg = strengthGradient
        # 更快但质量可能会更低
        self.fm = fastMode

    def loadImage(self, path="./Anime4K/pic/p1.png"):
        self.srcFile = path
        self.dstImg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.H = self.dstImg.shape[0]
        self.W = self.dstImg.shape[1]

    # 执行Anime4K
    def process(self):
        for i in range(self.ps):
            self.getGray()
            self.pushColor()
            self.getGradientByScharr()
            self.pushGradient()
            
            self.getGray()
            self.pushColor()
            self.getGradientBySobel()
            self.pushGradient()

    # 计算图像的灰度并将其存储到Alpha通道
    def getGray(self):
        B, G, R, A = 0, 1, 2, 3

        def callBack(i, j, pixel):
            pixel[A] = 0.299 * pixel[R] + 0.587 * pixel[G] + 0.114 * pixel[B]
            return pixel

        self.changeEachPixel(self.dstImg, callBack)

    #使图像的线条在Alpha通道中的灰度引导下变细
    def pushColor(self):
        B, G, R, A = 0, 1, 2, 3

        def getLightest(mc, a, b, c):
            mc[R] = mc[R] * (1 - self.sc) + (a[R] / 3 + b[R] / 3 + c[R] / 3) * self.sc
            mc[G] = mc[G] * (1 - self.sc) + (a[G] / 3 + b[G] / 3 + c[G] / 3) * self.sc
            mc[B] = mc[B] * (1 - self.sc) + (a[B] / 3 + b[B] / 3 + c[B] / 3) * self.sc
            mc[A] = mc[A] * (1 - self.sc) + (a[A] / 3 + b[A] / 3 + c[A] / 3) * self.sc

        def callBack(i, j, pixel):
            iN, iP, jN, jP = -1, 1, -1, 1
            #第一行
            if i == 0:
                iN = 0
            #最后一行
            elif i == self.H - 1:
                iP = 0
            #第一列
            if j == 0:
                jN = 0
            #最后一列
            elif j == self.W - 1:
                jP = 0

            tl, tc, tr = (  #左上方    上方  右上方
                self.dstImg[i + iN, j + jN],
                self.dstImg[i + iN, j],
                self.dstImg[i + iN, j + jP],
            )
            ml, mc, mr = (  #左方    中间  右方
                self.dstImg[i, j + jN],
                pixel,
                self.dstImg[i, j + jP],
            )
            bl, bc, br = (  #左下方    下方  右下方
                self.dstImg[i + iP, j + jN],
                self.dstImg[i + iP, j],
                self.dstImg[i + iP, j + jP],
            )

            # 上下侧
            maxD = max(bl[A], bc[A], br[A])
            minL = min(tl[A], tc[A], tr[A])
            if minL > mc[A] and mc[A] > maxD:
                getLightest(mc, tl, tc, tr)
            else:
                maxD = max(tl[A], tc[A], tr[A])
                minL = min(bl[A], bc[A], br[A])
                if minL > mc[A] and mc[A] > maxD:
                    getLightest(mc, bl, bc, br)

            # 次对角
            maxD = max(ml[A], mc[A], bc[A])
            minL = min(tc[A], tr[A], mr[A])
            if minL > maxD:
                getLightest(mc, tc, tr, mr)
            else:
                maxD = max(tc[A], mc[A], mr[A])
                minL = min(ml[A], bl[A], bc[A])
                if minL > maxD:
                    getLightest(mc, ml, bl, bc)

            # 左右侧
            maxD = max(tl[A], ml[A], bl[A])
            minL = min(tr[A], mr[A], br[A])
            if minL > mc[A] and mc[A] > maxD:
                getLightest(mc, tr, mr, br)
            else:
                maxD = max(tr[A], mr[A], br[A])
                minL = min(tl[A], ml[A], bl[A])
                if minL > mc[A] and mc[A] > maxD:
                    getLightest(mc, tl, ml, bl)

            # 对角
            maxD = max(tc[A], mc[A], ml[A])
            minL = min(mr[A], br[A], bc[A])
            if minL > maxD:
                getLightest(mc, mr, br, bc)
            else:
                maxD = max(bc[A], mc[A], mr[A])
                minL = min(ml[A], tl[A], tc[A])
                if minL > maxD:
                    getLightest(mc, ml, tl, tc)

            return pixel

        self.changeEachPixel(self.dstImg, callBack)

    # 计算图像的梯度并将其存储到Alpha通道
    def getGradientBySobel(self):
        B, G, R, A = 0, 1, 2, 3
        if self.fm == True:

            def callBack(i, j, pixel):
                if i == 0 or j == 0 or i == self.H - 1 or j == self.W - 1:
                    return pixel

                Grad = abs(
                    self.dstImg[i + 1, j - 1][A]
                    + 2 * self.dstImg[i + 1, j][A]
                    + self.dstImg[i + 1, j + 1][A]
                    - self.dstImg[i - 1, j - 1][A]
                    - 2 * self.dstImg[i - 1, j][A]
                    - self.dstImg[i - 1, j + 1][A]
                ) + abs(
                    self.dstImg[i - 1, j - 1][A]
                    + 2 * self.dstImg[i, j - 1][A]
                    + self.dstImg[i + 1, j - 1][A]
                    - self.dstImg[i - 1, j + 1][A]
                    - 2 * self.dstImg[i, j + 1][A]
                    - self.dstImg[i + 1, j + 1][A]
                )

                rst = self.unFloat(Grad / 2)
                pixel[A] = 255 - rst
                return pixel

        else:

            def callBack(i, j, pixel):
                if i == 0 or j == 0 or i == self.H - 1 or j == self.W - 1:
                    return pixel
                
                #sobel算子
                Grad = (
                    (
                        self.dstImg[i + 1, j - 1][A]
                        + 2 * self.dstImg[i + 1, j][A]
                        + self.dstImg[i + 1, j + 1][A]
                        - self.dstImg[i - 1, j - 1][A]
                        - 2 * self.dstImg[i - 1, j][A]
                        - self.dstImg[i - 1, j + 1][A]
                    )
                    ** 2
                    + (
                        self.dstImg[i - 1, j - 1][A]
                        + 2 * self.dstImg[i, j - 1][A]
                        + self.dstImg[i + 1, j - 1][A]
                        - self.dstImg[i - 1, j + 1][A]
                        - 2 * self.dstImg[i, j + 1][A]
                        - self.dstImg[i + 1, j + 1][A]
                    )
                    ** 2
                ) ** (0.5)
                
                rst = self.unFloat(Grad)
                pixel[A] = 255 - rst
                return pixel

        self.changeEachPixel(self.dstImg, callBack)
        
        
    # 计算图像的梯度并将其存储到Alpha通道
    def getGradientByScharr(self):
        B, G, R, A = 0, 1, 2, 3
        if self.fm == True:

            def callBack(i, j, pixel):
                if i == 0 or j == 0 or i == self.H - 1 or j == self.W - 1:
                    return pixel

                Grad = abs(
                    3 * self.dstImg[i + 1, j - 1][A]
                    + 10 * self.dstImg[i + 1, j][A]
                    + 3 * self.dstImg[i + 1, j + 1][A]
                    - 3 * self.dstImg[i - 1, j - 1][A]
                    - 10 * self.dstImg[i - 1, j][A]
                    - 3 * self.dstImg[i - 1, j + 1][A]
                ) + abs(
                    3 * self.dstImg[i - 1, j - 1][A]
                    + 10 * self.dstImg[i, j - 1][A]
                    + 3 * self.dstImg[i + 1, j - 1][A]
                    - 3 * self.dstImg[i - 1, j + 1][A]
                    - 10 * self.dstImg[i, j + 1][A]
                    - 3 * self.dstImg[i + 1, j + 1][A]
                )

                rst = self.unFloat(Grad / 2)
                pixel[A] = 255 - rst
                return pixel

        else:
            def callBack(i, j, pixel):
                if i == 0 or j == 0 or i == self.H - 1 or j == self.W - 1:
                    return pixel
                
                #scharr算子
                Grad = (
                    (
                        3 * self.dstImg[i + 1, j - 1][A]
                        + 10 * self.dstImg[i + 1, j][A]
                        + 3 * self.dstImg[i + 1, j + 1][A]
                        - 3 * self.dstImg[i - 1, j - 1][A]
                        - 10 * self.dstImg[i - 1, j][A]
                        - 3 * self.dstImg[i - 1, j + 1][A]
                    )
                    ** 2
                    + (
                        3 * self.dstImg[i - 1, j - 1][A]
                        + 10 * self.dstImg[i, j - 1][A]
                        + 3 * self.dstImg[i + 1, j - 1][A]
                        - 3 * self.dstImg[i - 1, j + 1][A]
                        - 10 * self.dstImg[i, j + 1][A]
                        - 3 * self.dstImg[i + 1, j + 1][A]
                    )
                    ** 2
                ) ** (0.5)
                
                rst = self.unFloat(Grad)
                pixel[A] = 255 - rst
                return pixel

        self.changeEachPixel(self.dstImg, callBack)

    # 将使图像的线条在Alpha通道中的梯度引导下锐化
    def pushGradient(self):
        B, G, R, A = 0, 1, 2, 3

        def getLightest(mc, a, b, c):
            mc[R] = mc[R] * (1 - self.sg) + (a[R] / 3 + b[R] / 3 + c[R] / 3) * self.sg
            mc[G] = mc[G] * (1 - self.sg) + (a[G] / 3 + b[G] / 3 + c[G] / 3) * self.sg
            mc[B] = mc[B] * (1 - self.sg) + (a[B] / 3 + b[B] / 3 + c[B] / 3) * self.sg
            mc[A] = 255
            return mc

        def callBack(i, j, pixel):
            iN, iP, jN, jP = -1, 1, -1, 1
            if i == 0:
                iN = 0
            elif i == self.H - 1:
                iP = 0
            if j == 0:
                jN = 0
            elif j == self.W - 1:
                jP = 0

            tl, tc, tr = (
                self.dstImg[i + iN, j + jN],
                self.dstImg[i + iN, j],
                self.dstImg[i + iN, j + jP],
            )
            ml, mc, mr = (
                self.dstImg[i, j + jN],
                pixel,
                self.dstImg[i, j + jP],
            )
            bl, bc, br = (
                self.dstImg[i + iP, j + jN],
                self.dstImg[i + iP, j],
                self.dstImg[i + iP, j + jP],
            )

            # 上下侧
            maxD = max(bl[A], bc[A], br[A])
            minL = min(tl[A], tc[A], tr[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, tl, tc, tr)

            maxD = max(tl[A], tc[A], tr[A])
            minL = min(bl[A], bc[A], br[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, bl, bc, br)

            # 次对角
            maxD = max(ml[A], mc[A], bc[A])
            minL = min(tc[A], tr[A], mr[A])
            if minL > maxD:
                return getLightest(mc, tc, tr, mr)
            maxD = max(tc[A], mc[A], mr[A])
            minL = min(ml[A], bl[A], bc[A])
            if minL > maxD:
                return getLightest(mc, ml, bl, bc)

            # 左右侧
            maxD = max(tl[A], ml[A], bl[A])
            minL = min(tr[A], mr[A], br[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, tr, mr, br)
            maxD = max(tr[A], mr[A], br[A])
            minL = min(tl[A], ml[A], bl[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, tl, ml, bl)

            # 对角
            maxD = max(tc[A], mc[A], ml[A])
            minL = min(mr[A], br[A], bc[A])
            if minL > maxD:
                return getLightest(mc, mr, br, bc)
            maxD = max(bc[A], mc[A], mr[A])
            minL = min(ml[A], tl[A], tc[A])
            if minL > maxD:
                return getLightest(mc, ml, tl, tc)

            pixel[A] = 255
            return pixel

        self.changeEachPixel(self.dstImg, callBack)

    # 展示生成的图像
    def show(self):
        cv2.imshow("dstImg", self.dstImg)
        cv2.waitKey()

    # 遍历图像的所有像素，通过回调函数对其更改, 所有更改将在遍历后应用
    def changeEachPixel(self, img, callBack):
        tmp = img.copy()
        for i in range(self.H):
            for j in range(self.W):
                tmp[i, j] = callBack(i, j, img[i, j].copy())
        self.dstImg = tmp

    # 将float转为uint8,范围[0-255]
    def unFloat(self, n):
        n += 0.5
        if n >= 255:
            return 255
        elif n <= 0:
            return 0
        return n

    # 展示图像基本信息
    def showInfo(self):
        print("Width: %d, Height: %d" % (self.dstImg.shape[1], self.dstImg.shape[0]))
        print("----------------------------------------------")
        print(
            "Input: %s\nPasses: %d\nFast Mode: %r\nStrength color: %g\nStrength gradient: %g"
            % (self.srcFile, self.ps, self.fm, self.sc, self.sg)
        )
        print("----------------------------------------------")

    # save image to disk
    def saveImage(self, filename="./output.png"):
        cv2.imwrite(filename, self.dstImg)
