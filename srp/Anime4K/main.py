from Anime4KPython.Anime4K import Anime4K
import time

if __name__ == "__main__":
    #give arguments
    anime4k = Anime4K(passes=1,strengthColor=1/6,strengthGradient=1/2,fastMode=False)
    #load your image
    anime4k.loadImage("input/test.png")
    #show basic infomation
    anime4k.showInfo()
    time_start = time.time()
    #main process
    anime4k.process()
    time_end = time.time()
    print("Total time:", time_end - time_start, "s")
    #show thr result by opencv
    #anime4k.show()
    #save to disk
    anime4k.saveImage("output/output.png")
