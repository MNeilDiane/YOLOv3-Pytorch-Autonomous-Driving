#this file will new a window 
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
import tkinter.messagebox

win = tk.Tk()
win.title('基于YOLO的自动驾驶目标检测程序')
win.geometry('800x600')
win.resizable(False,False)
def get_image(filename,width,height):
    im = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(im)
def video_detection():
    yolo = YOLO()
    print('进行视频图像目标检测')
    file = filedialog.askopenfilename()
    print("path:", file)

    video_path = file
    video_save_path = ""
    video_fps = 25.0
    test_interval = 100
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



def camera_detection():
    print('进行摄像头实时目标检测')
    os.system('python vision.py')
def image_detection():
    yolo = YOLO()
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
def exit():
    sys.exit()

canvas_root = tk.Canvas(win,width=800, height=600)
im_root = get_image("photo.png", 800, 600)
canvas_root.create_image(400, 300, image=im_root)
canvas_root.pack()

Button0 = tk.Button(win, text='图片检测', font=('宋体', 12, 'bold italic'), bg='#d3fbfb', fg='blue',
                    width=10, height=3, relief=RAISED, command=lambda :image_detection())
Button0.place(x=3, y=3, width=100, height=40)
Button1 = tk.Button(win, text='视频检测', font=('宋体', 12, 'bold italic'), bg='#d3fbfb', fg='blue',
                    width=10, height=3, relief=RAISED, command=lambda:video_detection())
Button1.place(x=3, y=43, width=100, height=40)
Button2 = tk.Button(win, text='摄像头检测', font=('宋体', 12, 'bold italic'), bg='#d3fbfb', fg='blue',
                    width=10, height=3, relief=RAISED, command=lambda:camera_detection())
Button2.place(x=3, y=83, width=100, height=40)

Button3 = tk.Button(win, text='退出程序', font=('宋体', 12, 'bold italic'),bg='red',
                    width=10, height=3, relief=RAISED, command=lambda:exit())
Button3.place(x=3, y=123, width=100, height=40)
win.mainloop()
