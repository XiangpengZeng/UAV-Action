# -*- coding: utf-8 -*- 

from cgitb import text
from email.mime import image
import tkinter
from tkinter import NW, Button, font
import tkinter.messagebox
from turtle import bgcolor
from PIL import Image, ImageTk

from scripts import Second_frame

# 登陆界面

# 打开指定图片并缩放
def get_image(filename, width, height):
    img = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(img)

root = tkinter.Tk()
root.title('Vision for UAV')
root.geometry('500x500+300+150')
root.resizable(False, False)

# 创建画布，设置要显示的图片，把画布添加至窗口,设置名字
path = './star1.jpg'
canvas_root = tkinter.Canvas(root, width=500, height=500)
img_root = get_image(path, 500, 500)
canvas_root.create_image(250, 250, image=img_root)    #默认坐标指的是图片中心在画布中的位置,anchor=NW则可指定图片左上角的位置
canvas_root.create_text(250,100, text='基于多无人机视角的行为识别系统', font=("宋体", 18), fill='white')
canvas_root.pack()


# 用户名、密码文字
user = tkinter.Label(root, text='用户名:', bg='white', font=('宋体', 17))
user.place(relx=0.32, rely=0.4, width=85, height=32, anchor=tkinter.CENTER)
passeord = tkinter.Label(root, text='密码:', bg='white', font=('宋体', 17))
passeord.place(relx=0.32, rely=0.5, width=85, height=32, anchor=tkinter.CENTER)

# 用户名、密码输入，用Entry来做, bd是边框
user_name = tkinter.Entry(root, bd=3, font=('Times', 17))
user_name.place(relx=0.62, rely=0.4, width=200, height=32, anchor=tkinter.CENTER)
password_name = tkinter.Entry(root, bd=3, font=('Times', 17), show='*')   # 密码加密
password_name.place(relx=0.62, rely=0.5, width=200, height=32, anchor=tkinter.CENTER)

# 登录，退出的函数
def second():
    Second_frame.second_frame(root)

def exit_1():
    root.destroy()

# 登录，退出，Button来做
sign_in = tkinter.Button(root, text='登录', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=second)
sign_in.place(relx=0.43, rely=0.65, width=85, height=37, anchor=tkinter.CENTER)
sign_out = tkinter.Button(root, text='退出', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=exit_1)
sign_out.place(relx=0.63, rely=0.65, width=85, height=37, anchor=tkinter.CENTER)

root.mainloop()