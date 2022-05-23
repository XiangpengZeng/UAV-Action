# -*- coding: utf-8 -*- 

from cgitb import text
from email.mime import image
import tkinter
from tkinter import NW, Button, font
import tkinter.messagebox
from turtle import bgcolor
from PIL import Image, ImageTk
from tkinter import filedialog

from tkinter.filedialog import askdirectory, askopenfilename
from scripts import Fifth_frame

# 行为识别选择文件界面

# 打开指定图片并缩放
def get_image(filename, width, height):
    img = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(img)

def exit_1():
    root.destroy()

def selectPath1():
    path_ = askopenfilename()
    path_file1.set(path_)
    temp = path_file1.get()
    linux_path = '/'.join(temp.split('\\'))
    print(linux_path)

def selectPath2():
    path_ = askopenfilename()
    path_file2.set(path_)
    temp = path_file2.get()
    linux_path = '/'.join(temp.split('\\'))
    print(linux_path)

def selectPath3():
    path_ = askopenfilename()
    path_file3.set(path_)
    temp = path_file3.get()
    linux_path = '/'.join(temp.split('\\'))
    print(linux_path)

def frame_display():
    path1 = path_file1.get()
    path2 = path_file2.get()
    path3 = path_file3.get()
    Fifth_frame.fifth_frame(root, path1, path2, path3)

def fourth_frame(last_window):
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
    canvas_root.create_text(250,75, text='当前功能：人体行为识别', font=("宋体", 18), fill='white')
    canvas_root.pack()

    # 请上传视频
    mode = tkinter.Label(root, text='请选择上传视频:', bg='LightYellow', font=('宋体', 17))
    mode.place(relx=0.5, rely=0.27, width=190, height=32, anchor=tkinter.CENTER)

    # 选择文件
    global path_file1, path_file2, path_file3
    path_file1 = tkinter.StringVar()
    path_file2 = tkinter.StringVar()
    path_file3 = tkinter.StringVar()
    tkinter.Label(root,text = "视角1路径:",bg='white', font=('宋体', 12)).place(relx=0.15, rely=0.40, width=85, height=25, anchor=tkinter.CENTER)
    tkinter.Entry(root, textvariable = path_file1, bd=3,font=('Times', 12)).place(relx=0.504, rely=0.40, width=265, height=25, anchor=tkinter.CENTER)
    tkinter.Button(root, text = "路径选择", font=('宋体', 12), bd=4, command = selectPath1).place(relx=0.86, rely=0.402, width=79, height=25, anchor=tkinter.CENTER)

    tkinter.Label(root,text = "视角2路径:",bg='white', font=('宋体', 12)).place(relx=0.15, rely=0.50, width=85, height=25, anchor=tkinter.CENTER)
    tkinter.Entry(root, textvariable = path_file2, bd=3,font=('Times', 12)).place(relx=0.504, rely=0.50, width=265, height=25, anchor=tkinter.CENTER)
    tkinter.Button(root, text = "路径选择", font=('宋体', 12), bd=4, command = selectPath2).place(relx=0.86, rely=0.502, width=79, height=25, anchor=tkinter.CENTER)

    tkinter.Label(root,text = "视角3路径:",bg='white', font=('宋体', 12)).place(relx=0.15, rely=0.60, width=85, height=25, anchor=tkinter.CENTER)
    tkinter.Entry(root, textvariable = path_file3, bd=3,font=('Times', 12)).place(relx=0.504, rely=0.60, width=265, height=25, anchor=tkinter.CENTER)
    tkinter.Button(root, text = "路径选择", font=('宋体', 12), bd=4, command = selectPath3).place(relx=0.86, rely=0.602, width=79, height=25, anchor=tkinter.CENTER)
    # 效果展示
    tkinter.Button(root, text = "效果展示", font=('宋体', 17), bd=5, command = frame_display, bg='SkyBlue').place(relx=0.33, rely=0.72, width=150, height=37, anchor=tkinter.CENTER)
    # 退出
    sign_out = tkinter.Button(root, text='退出', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=exit_1)
    sign_out.place(relx=0.67, rely=0.72, width=150, height=37, anchor=tkinter.CENTER)

    root.mainloop()
# fourth_frame()