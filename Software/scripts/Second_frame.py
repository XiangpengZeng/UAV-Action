# -*- coding: utf-8 -*- 

from cgitb import text
from email.mime import image
import tkinter
from tkinter import NW, Button, font
import tkinter.messagebox
from turtle import bgcolor
from PIL import Image, ImageTk

from scripts import Third_frame, Fourth_frame

# 功能选择界面


def get_image(filename, width, height):
    img = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(img)

def exit_1():
    root.destroy()

def pose_display():
    Third_frame.third_frame(root)

def action_display():
    Fourth_frame.fourth_frame(root)

def second_frame(last_window):
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
    canvas_root.create_text(250,40, text='基于多无人机视角的行为识别系统', font=("宋体", 18), fill='white')
    canvas_root.pack()

    # 请选择模式
    mode = tkinter.Label(root, text='请选择功能:', bg='LightYellow', font=('宋体', 17))
    mode.place(relx=0.5, rely=0.25, width=150, height=32, anchor=tkinter.CENTER)

    # Button
    pose = tkinter.Button(root, text='姿态估计', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=pose_display)
    pose.place(relx=0.5, rely=0.4, width=150, height=37, anchor=tkinter.CENTER)
    action = tkinter.Button(root, text='行为识别', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=action_display)
    action.place(relx=0.5, rely=0.55, width=150, height=37, anchor=tkinter.CENTER)
    sign_out = tkinter.Button(root, text='退出', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=exit_1)
    sign_out.place(relx=0.5, rely=0.7, width=150, height=37, anchor=tkinter.CENTER)


    root.mainloop()

