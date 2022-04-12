import sys
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
from yolo import YOLO
from pypylon import pylon
#import redis

class Main():

    def __init__(self):
        self.test_interval = 100
        self.win = tk.Tk()
        self.win.title('MOCO自动驾驶目标检测系统 V1.0')
        self.win.geometry('1280x720')
        self.win.resizable(False, False)
        self.winisvisible = 1
        canvas_root = tk.Canvas(self.win, width=1280, height=720)
        im_root = self.get_image("car 1.png", 1280, 720)
        canvas_root.create_image(640, 360, image=im_root)
        canvas_root.pack()
        self.yolo = YOLO()
        #redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0)
        #self.r = redis.StrictRedis(connection_pool=redis_pool)

        #INTEGER
        self.integer = tk.StringVar()
        self.integer.set('0')

        #Buttons
        im_button0 = self.get_image('7.png', 160, 100)
        self.Button0 = tk.Button(self.win, text='image detection', image=im_button0, compound=tk.CENTER,
                            font=('Time New Roman', 12, 'bold'), fg='#00F5FF',
                            width=10, height=3, relief=RAISED, command=lambda: self.image_detection())
        self.Button0.place(x=3, y=3, width=160, height=100)
        im_button1 = self.get_image('8.png', 160, 100)
        self.Button1 = tk.Button(self.win, text='Video detection', image=im_button1, compound=tk.CENTER,
                            font=('Time New Roman', 12, 'bold'), fg='#00F5FF',
                            width=10, height=3, relief=RAISED, command=lambda: self.video_detection())
        self.Button1.place(x=3, y=103, width=162, height=100)
        im_button2 = self.get_image('9.png', 160, 100)
        self.Button2 = tk.Button(self.win, text='Camera detection', image=im_button2, compound=tk.CENTER,
                            font=('Time New Roman', 12, 'bold'), fg='#00F5FF',
                            width=10, height=3, relief=RAISED, command=lambda: self.camera_detection())
        self.Button2.place(x=3, y=203, width=162, height=100)
        im_button3 = self.get_image('10.png', 160, 100)
        self.Button3 = tk.Button(self.win, text='FPS test', image=im_button3, compound=tk.CENTER,
                            font=('Time New Roman', 12, 'bold'), fg='#00F5FF',
                            width=10, height=3, relief=RAISED, command=lambda: self.fps_test())
        self.Button3.place(x=3, y=303, width=162, height=100)
        im_button4 = self.get_image('11.png', 160, 100)
        self.Button4 = tk.Button(self.win, text='Visible', image=im_button4, compound=tk.CENTER,
                            font=('Time New Roman', 12, 'bold'), fg='#00F5FF',
                            width=10, height=3, relief=RAISED, command=lambda:self.visible())
        self.Button4.place(x=3, y=403, width=160, height=100)
        im_button5 = self.get_image('12.png', 160, 100)
        self.Button5 = tk.Button(self.win, text='Exit', image=im_button5, compound=tk.CENTER, font=('Time New Roman', 12, 'bold'),
                            fg='red',
                            width=10, height=3, relief=RAISED, command=lambda: exit())
        self.Button5.place(x=3, y=503, width=160, height=100)
        #ENTRY
        self.Entry1 = tk.Entry(self.win, textvariable=self.integer, justify="center", font=('Time New Roman', 8))
        self.Entry1.place(x=163, y=610, width=600, height=50)
        mainloop()

    def visible(self):
        self.winisvisible=-1*self.winisvisible
        self.integer.set(self.winisvisible)
        print("visible:"+str(self.winisvisible))

    def get_image(self, filename, width, height):
        im = Image.open(filename).resize((width, height))
        return ImageTk.PhotoImage(im)

    '''
    image_detection()是进行单个图片检测的程序
    '''

    def image_detection(self):

        file = filedialog.askopenfilename()
        while True:
            img = file
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                break
            else:
                r_image = self.yolo.detect_image(image)
                r_image.show()
                r_image.save("img.jpg")
                print('Successfully saved!')
                cv2.imshow(r_image)

    '''
    video_detection是进行视频的目标检测程序
    '''

    def video_detection(self):

        print('进行视频图像目标检测')
        file = filedialog.askopenfilename()
        print("path:", file)

        video_path = file
        video_save_path = ""
        video_fps = 25.0
        capture = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            label = self.yolo.get_label(frame)
            self.integer.set(label)
            # self.r.hset("camera", "left", str(label))
            print(label)
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            if self.winisvisible==1:
                frame = np.array(self.yolo.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if c == 27:
                    capture.release()
                    break
            if video_save_path != "":
                out.write(frame)


        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()



    '''
    camera_detection 会调用vision.py
    这个程序是使用Basler相机进行实时目标检测的程序
    '''

    def camera_detection(self):
        print('进行摄像头实时目标检测')
        # 连接Basler相机列表的第一个相机1
        capture = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        # 开始读取图像
        capture.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()

        # 转换为OpenCV的BGR彩色格式
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while capture.IsGrabbing():
            grabResult = capture.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # 转换为OpenCV图像格式
                image = converter.Convert(grabResult)
                frame = image.GetArray()
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                label = self.yolo.get_label(frame)
                self.integer.set(label)
                # self.r.hset("camera", "left", str(label))
                print(label)
                # 进行检测
                if self.winisvisible == 1:
                    frame = np.array(self.yolo.detect_image(frame))
                    # RGBtoBGR满足opencv显示格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.namedWindow('title', cv2.WINDOW_NORMAL)
                    cv2.imshow('title', frame)
                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break
            grabResult.Release()

        # 关闭相机2
        capture.StopGrabbing()
        # 关闭窗口
        cv2.destroyAllWindows()

    '''
    进行fps测试，检测机器性能
    '''

    def fps_test(self):
        img = Image.open('img/street.jpg')
        tact_time = self.yolo.get_FPS(img, self.test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    def exit(self):
        sys.exit()
if __name__=='__main__':
    Main()






