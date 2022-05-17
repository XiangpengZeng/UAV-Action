# -*- coding: utf-8 -*- 

from cgitb import text
from email.mime import image
from modulefinder import packagePathMap
import tkinter
from tkinter import NW, Button, font
import tkinter.messagebox
from turtle import bgcolor
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2

from tkinter.filedialog import askdirectory
import run_zeng_video

# 姿态估计展示界面

# 打开指定图片并缩放
def get_image(filename, width, height):
    img = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(img)

def exit_1():
    root.destroy()

def sixth_frame(last_window, path_1):
    global path1
    # 初始传参
    path1 = path_1

    last_window.destroy()
    # 一个程序只能有一个Tk对象
    global root 
    root = tkinter.Tk()
    root.title('Vision for UAV')
    root.geometry('500x500+300+150')
    root.resizable(False, False)

    # 创建画布，设置要显示的图片，把画布添加至窗口,设置名字
    path = './star1.jpg'
    canvas_root = tkinter.Canvas(root, width=500, height=500)
    img_root = get_image(path, 500, 500)
    canvas_root.create_image(250, 250, image=img_root)    #默认坐标指的是图片中心在画布中的位置,anchor=NW则可指定图片左上角的位置
    canvas_root.create_text(250,35, text='基于多无人机视角的行为识别系统', font=("宋体", 18), fill='white')
    canvas_root.create_text(250,75, text='当前功能：人体姿态估计', font=("宋体", 18), fill='white')
    canvas_root.pack()

    global image1
    image1 = tkinter.Label(root, text='展示界面', bg='white', font=('宋体', 17))
    image1.place(relx=0.5, rely=0.4, width=320, height=180, anchor=tkinter.CENTER)

    # 展示
    disp = tkinter.Button(root, text='展示', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=video_play)
    disp.place(relx=0.35, rely=0.72, width=130, height=37, anchor=tkinter.CENTER)

    # 退出
    sign_out = tkinter.Button(root, text='退出', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=exit_1)
    sign_out.place(relx=0.65, rely=0.72, width=130, height=37, anchor=tkinter.CENTER)

    root.mainloop()

def video_play():
    video= cv2.VideoCapture(path1)  
    while video.isOpened():
        ret, frame = video.read()  # 读取照片
     	# print('读取成功')
        if ret == True:
            img = run_zeng_video.play(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  #转换颜色使播放时保持原有色彩
            current_image = Image.fromarray(img).resize((320, 180))  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            image1.imgtk = imgtk    # image1,2,3  Label
            image1.config(image=imgtk)
            cv2.waitKey(30)
            image1.update()   #每执行一次只显示一张图片，需要更新窗口实现视频播放
        else: break

    video.release()

# sixth_frame()