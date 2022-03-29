import os
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
import threading



yolo=YOLO()
test_interval=100
win = tk.Tk()
win.title('MOCO自动驾驶目标检测系统 V1.0')
win.geometry('1280x720')
win.resizable(False, False)
'''
get_image 是对图片大小进行变换，便于设置背景图片
'''

def get_image(filename,width,height):
    im = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(im)
'''
image_detection()是进行单个图片检测的程序
'''
def image_detection():

    file = filedialog.askopenfilename()
    while True:
        img = file
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            r_image.save("img.jpg")
            print('Successfully saved!')
            cv2.imshow(r_image)
'''
video_detection是进行视频的目标检测程序
'''
def video_detection():

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
        frame = np.array(yolo.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if video_save_path != "":
            out.write(frame)

        if c == 27:
            capture.release()
            break

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

def camera_detection():
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
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
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
提取检测标签，置信度等信息
'''
def info_extraction():
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
        label = yolo.get_label(frame)
        print(label)
'''
在entry窗口中显示信息
'''


'''
进行fps测试，检测机器性能
'''
def fps_test():
    img = Image.open('img/street.jpg')
    tact_time = yolo.get_FPS(img, test_interval)
    print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')



def exit():
    sys.exit()

canvas_root = tk.Canvas(win,width=1280, height=720)
im_root = get_image("car 1.png", 1280, 720)
canvas_root.create_image(640, 360, image=im_root)
canvas_root.pack()
#entry1 = tk.Entry(win, justify='left', bg='#d3fbfb')
#entry1.place(x=3, y=400, width=200, height=200)
im_button0=get_image('1.png', 160, 100)
Button0 = tk.Button(win, text='图片检测', image=im_button0, compound=tk.CENTER, font=('黑体', 12, 'bold'), fg='blue',
                    width=10, height=3, relief=RAISED, command=lambda :image_detection())
Button0.place(x=3, y=3, width=160, height=100)
im_button1=get_image('2.png', 160, 100)
Button1 = tk.Button(win, text='视频检测', image=im_button1, compound=tk.CENTER, font=('黑体', 12, 'bold '), fg='blue',
                    width=10, height=3, relief=RAISED, command=lambda:video_detection())
Button1.place(x=3, y=103, width=162, height=100)
im_button2=get_image('3.png', 160, 100)
Button2 = tk.Button(win, text='摄像头检测', image=im_button2, compound=tk.CENTER,font=('黑体', 12, 'bold'), fg='blue',
                    width=10, height=3, relief=RAISED, command=lambda:camera_detection())
Button2.place(x=3, y=203, width=162, height=100)
im_button3=get_image('4.png', 160, 100)
Button3 = tk.Button(win, text='FPS测试', image=im_button3, compound=tk.CENTER,font=('黑体', 12, 'bold'), fg='blue',
                    width=10, height=3, relief=RAISED, command=lambda:fps_test())
Button3.place(x=3, y=303, width=162, height=100)
im_button4=get_image('5.png', 160, 100)
Button4 = tk.Button(win,text='检测信息提取', image=im_button4, compound=tk.CENTER,font=('黑体', 12, 'bold'), fg='blue',
                    width=10, height=3, relief=RAISED)
Button4.place(x=3, y=403, width=160, height=100)
im_button5=get_image('6.png', 160, 100)
Button5 = tk.Button(win, text='退出程序', image=im_button5, compound=tk.CENTER,font=('黑体', 12, 'bold'),
                    width=10, height=3, relief=RAISED, command=lambda:exit())
Button5.place(x=3, y=503, width=160, height=100)

win.mainloop()
