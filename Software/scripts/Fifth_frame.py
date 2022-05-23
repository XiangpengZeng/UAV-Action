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
import numpy as np

from tkinter.filedialog import askdirectory

from scripts import run_zeng_3video, action_att, Seventh_frame

# 行为识别展示界面（3视频）
save_list_total = []
save_list_total2 = []
save_list_total3 = []

# 打开指定图片并缩放
def get_image(filename, width, height):
    img = Image.open(filename).resize((width, height))
    return ImageTk.PhotoImage(img)

def exit_1():
    root.destroy()

def seventh_table_frame():
    Seventh_frame.seventh_frame(root, list_result, list_result2, list_result3, list_result_fusion)

def fifth_frame(last_window, path_1, path_2, path_3):
    global path1, path2, path3
    # 初始传参
    path1 = path_1
    path2 = path_2
    path3 = path_3

    last_window.destroy()
    # 一个程序只能有一个Tk对象
    global root 
    root = tkinter.Tk()
    root.title('Vision for UAV')
    root.geometry('1000x600+300+150')
    root.resizable(False, False)

    # 创建画布，设置要显示的图片，把画布添加至窗口,设置名字
    path = './star1.jpg'
    canvas_root = tkinter.Canvas(root, width=1000, height=600)
    img_root = get_image(path, 1000, 600)
    canvas_root.create_image(500, 300, image=img_root)    #默认坐标指的是图片中心在画布中的位置,anchor=NW则可指定图片左上角的位置
    canvas_root.create_text(500,35, text='基于多无人机视角的行为识别系统', font=("宋体", 18), fill='white')
    canvas_root.create_text(500,83, text='当前功能：人体行为识别', font=("宋体", 18), fill='white')
    canvas_root.pack()

    global image1, image2, image3
    image1 = tkinter.Label(root, text='视角1', bg='white', font=('宋体', 17))
    image1.place(relx=0.17, rely=0.39, width=320, height=180, anchor=tkinter.CENTER)
    image2 = tkinter.Label(root, text='视角2', bg='white', font=('宋体', 17))
    image2.place(relx=0.5, rely=0.39, width=320, height=180, anchor=tkinter.CENTER)
    image3 = tkinter.Label(root, text='视角3', bg='white', font=('宋体', 17))
    image3.place(relx=0.83, rely=0.39, width=320, height=180, anchor=tkinter.CENTER)

    # 当前动作
    current = tkinter.Label(root, text='当前动作:', bg='white', font=('宋体', 17))
    current2 = tkinter.Label(root, text='当前动作:', bg='white', font=('宋体', 17))
    current3 = tkinter.Label(root, text='当前动作:', bg='white', font=('宋体', 17))
    current.place(relx=0.135, rely=0.60, width=105, height=32, anchor=tkinter.CENTER)
    current2.place(relx=0.465, rely=0.60, width=105, height=32, anchor=tkinter.CENTER)
    current3.place(relx=0.795, rely=0.60, width=105, height=32, anchor=tkinter.CENTER)

    current4 = tkinter.Label(root, text='融合决策:', bg='white', font=('宋体', 17))
    current4.place(relx=0.465, rely=0.69, width=105, height=32, anchor=tkinter.CENTER)

    global var, var2, var3, var4
    var = tkinter.StringVar()
    var2 = tkinter.StringVar()
    var3 = tkinter.StringVar()
    var4 = tkinter.StringVar()

    # 显示类别
    category = tkinter.Entry(root, textvariable=var, bd=3, font=('宋体', 17))
    category2 = tkinter.Entry(root, textvariable=var2, bd=3, font=('宋体', 17))
    category3 = tkinter.Entry(root, textvariable=var3, bd=3, font=('宋体', 17))
    category4 = tkinter.Entry(root, textvariable=var4, bd=3, font=('宋体', 17))
    category.place(relx=0.223, rely=0.60, width=60, height=32, anchor=tkinter.CENTER)
    category2.place(relx=0.553, rely=0.60, width=60, height=32, anchor=tkinter.CENTER)
    category3.place(relx=0.883, rely=0.60, width=60, height=32, anchor=tkinter.CENTER)
    category4.place(relx=0.553, rely=0.69, width=60, height=32, anchor=tkinter.CENTER)

    # 展示
    disp = tkinter.Button(root, text='展示', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=video_play)
    disp.place(relx=0.25, rely=0.79, width=150, height=37, anchor=tkinter.CENTER)

    # 查看识别参数
    check_para = tkinter.Button(root, text='查看参数', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=seventh_table_frame)
    check_para.place(relx=0.5, rely=0.79, width=150, height=37, anchor=tkinter.CENTER)

    # 退出
    sign_out = tkinter.Button(root, text='退出', font=('宋体', 17), bd=5, bg='SkyBlue', activebackground='Olive', command=exit_1)
    sign_out.place(relx=0.75, rely=0.79, width=150, height=37, anchor=tkinter.CENTER)

    root.mainloop()

def video_play():
    video= cv2.VideoCapture(path1)  
    video2 = cv2.VideoCapture(path2)
    video3 = cv2.VideoCapture(path3)
    s_mean = s_mean2 = s_mean3 = 0
    l_mean = l_mean2 = l_mean3 = 0
    x_mean = x_mean2 = x_mean3 = 0
    y_mean = y_mean2 = y_mean3 = 0
    while video.isOpened():
        ret, frame = video.read()  # 读取照片
        ret2, frame2 = video2.read()
        ret3, frame3 = video3.read()
     	# print('读取成功')
        if ret == True and ret2 == True and ret3 == True:
            frame, frame2, frame3, savelist, savelist2, savelist3 = run_zeng_3video.play(frame, frame2, frame3)
            save_list_total.append(savelist)    # save_list_total是18个点
            save_list_total2.append(savelist2)
            save_list_total3.append(savelist3)

            rect = cv2.minAreaRect(np.array(save_list_total[:28]).reshape(-1,2))
            s = rect[1][0] * rect[1][1]
            x = (cv2.boxPoints(rect)[1][0] + cv2.boxPoints(rect)[3][0]) / 2
            y = (cv2.boxPoints(rect)[1][1] + cv2.boxPoints(rect)[3][1]) / 2

            s_mean = s_mean + s
            x_mean = x_mean + x
            y_mean = y_mean + y

            rect2 = cv2.minAreaRect(np.array(save_list_total2[:28]).reshape(-1,2))
            s2 = rect2[1][0] * rect2[1][1]
            x2 = (cv2.boxPoints(rect2)[1][0] + cv2.boxPoints(rect2)[3][0]) / 2
            y2 = (cv2.boxPoints(rect2)[1][1] + cv2.boxPoints(rect2)[3][1]) / 2

            s_mean2 = s_mean2 + s2
            x_mean2 = x_mean2 + x2
            y_mean2 = x_mean2 + y2

            rect3 = cv2.minAreaRect(np.array(save_list_total3[:28]).reshape(-1,2))
            s3 = rect3[1][0] * rect3[1][1]
            x3 = (cv2.boxPoints(rect3)[1][0] + cv2.boxPoints(rect3)[3][0]) / 2
            y3 = (cv2.boxPoints(rect3)[1][1] + cv2.boxPoints(rect3)[3][1]) / 2

            s_mean3 = s_mean3 + s3
            x_mean3 = x_mean3 + x3
            y_mean3 = y_mean3 + y3
            
            l = abs(savelist[10]-savelist[4])
            l2 = abs(savelist2[10]-savelist2[4])
            l3 = abs(savelist3[10]-savelist3[4])

            l_mean = l_mean + l
            l_mean2 = l_mean2 + l2
            l_mean3 = l_mean3 + l3

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  #转换颜色使播放时保持原有色彩
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
            img3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
            current_image = Image.fromarray(img).resize((320, 180))  # 将图像转换成Image对象
            current_image2 = Image.fromarray(img2).resize((320, 180))
            current_image3 = Image.fromarray(img3).resize((320, 180))
            imgtk = ImageTk.PhotoImage(image=current_image)
            imgtk2 = ImageTk.PhotoImage(image=current_image2)
            imgtk3 = ImageTk.PhotoImage(image=current_image3)
            image1.imgtk = imgtk   
            image1.config(image=imgtk)
            image2.imgtk = imgtk2
            image2.config(image=imgtk2)
            image3.imgtk = imgtk3
            image3.config(image=imgtk3)
            cv2.waitKey(30)
            image1.update()   #每执行一次只显示一张图片，需要更新窗口实现视频播放
            image2.update()
            image3.update()
        else: break

    length = len(save_list_total)
    length2 = len(save_list_total2)
    length3 = len(save_list_total3)
    s_mean = s_mean // length
    x_mean = int(x_mean // length)
    y_mean = int(y_mean // length)
    l_mean = l_mean // length

    s_mean2 = s_mean2 // length2
    x_mean2 = int(x_mean2 // length2)
    y_mean2 = int(y_mean2 // length2)
    l_mean2 = l_mean2 // length2

    s_mean3 = s_mean3 // length3
    x_mean3 = int(x_mean3 // length3)
    y_mean3 = int(y_mean3 // length3)
    l_mean3 = l_mean3 // length3

    video.release()
    video2.release()
    video3.release()
    view1, view2, view3, view_fusion = action_att.play_action(save_list_total, save_list_total2, save_list_total3)
    var.set(view1)
    var2.set(view2)
    var3.set(view3)
    var4.set(view_fusion)
    
    global list_result, list_result2, list_result3, list_result_fusion
    list_result = ["视角1", view1, s_mean, l_mean, str((x_mean, y_mean))]
    list_result2 = ["视角2", view2, s_mean2, l_mean2, str((x_mean2, y_mean2))]
    list_result3 = ["视角3", view3, s_mean3, l_mean3, str((x_mean3, y_mean3))]
    list_result_fusion = ["融合", view_fusion, '-', '-', '-']
    
    # cv2.destroyAllWindows()
# fifth_frame()