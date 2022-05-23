# -*- coding: utf-8 -*- 

from cgitb import text
from email.mime import image
from modulefinder import packagePathMap
import tkinter
from tkinter import NW, Button, font
import tkinter.messagebox
from turtle import bgcolor
from PIL import Image, ImageTk
from tkinter import ttk
import csv
from tkinter.filedialog import asksaveasfilename, askdirectory

# 打开指定图片并缩放
def get_image(filename, width, height):
    img = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(img)

def exit_1():
    root.destroy()

def selectSavePath():
    path_ = askdirectory()
    # path_file1.set(path_)
    # temp = path_file1.get()
    linux_path = '/'.join(path_.split('\\'))
    print(linux_path)
    return linux_path

 
def save_parameter():
    savePath = selectSavePath()
    with open(savePath + '/saveData.csv', mode='w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(list1)
        w.writerow(list2)
        w.writerow(list3)
        w.writerow(list_fuison)


def seventh_frame(last_window, list_1, list_2, list_3, listFuison):
    # 初始声明，方便其他调用
    global list1, list2, list3, list_fuison
    list1 = list_1
    list2 = list_2
    list3 = list_3
    list_fuison = listFuison  

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
    canvas_root.create_image(250, 250, image=img_root)
    canvas_root.create_text(250,35, text='基于多无人机视角的行为识别系统', font=("宋体", 18), fill='white')
    canvas_root.create_text(250,83, text='当前功能：人体行为识别--参数查看', font=("宋体", 18), fill='white')
    canvas_root.pack()  

    style1 = ttk.Style()
    style1.configure("Treeview.Heading", font=('宋体', 11))
    style1.configure("Treeview", font=('宋体', 11))
    style1.configure("Treeview", font=('Times', 11))
    # 创建表格
    tree_date = ttk.Treeview(root, show='headings')

    # 定义列
    tree_date['columns'] = ['view','category','area','length','position']
    tree_date.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

    # 设置列宽度
    tree_date.column('view',width=80, anchor='center')  # 居中
    tree_date.column('category',width=80, anchor='center')
    tree_date.column('area',width=125, anchor='center')
    tree_date.column('length',width=80, anchor='center')
    tree_date.column('position',width=80, anchor='center')

    # 添加列名
    tree_date.heading('view',text='视角')
    tree_date.heading('category',text='行为类别')
    tree_date.heading('area',text='最小外接矩形面积')
    tree_date.heading('length',text='双肩距离')
    tree_date.heading('position',text='目标位置')

    tree_date.insert('',0, text='date1', values=list1)
    tree_date.insert('',1, text='date2', values=list2)
    tree_date.insert('',2, text='date3', values=list3)
    tree_date.insert('',3, text='date4', values=list_fuison)
    # 退出
    sign_out = tkinter.Button(root, text='退出', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=exit_1)
    sign_out.place(relx=0.35, rely=0.8, width=120, height=37, anchor=tkinter.CENTER)
    # 保存
    save_file = tkinter.Button(root, text='保存数据', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=save_parameter)
    save_file.place(relx=0.65, rely=0.8, width=120, height=37, anchor=tkinter.CENTER)


    root.mainloop()

# seventh_frame(list__1, list__2, list__3, list__1fusion)
