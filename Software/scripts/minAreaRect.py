import cv2
import numpy as np


img = np.zeros((1080,1920,3), np.uint8)   
#fill the image with white
img.fill(255)
list = [1088,396,1084,428,1058,428,1048,472,1044,506,1110,428,1110,468,1096,490,1066,516,1066,574,1072,624,1098,514,1100,572,1106,632,1082,390,1094,390,1072,392,1100,394]
to_np = np.array(list)
#to_np.reshape((len(list)//2,2))
#print(to_np.reshape((len(list)//2,2)))
rect = cv2.minAreaRect(to_np.reshape(len(list)//2,2))
print(rect[1][1])
print(rect[1][0])
print(rect[1][1]*rect[1][0])
# calculate coordinate of the minimum area rectangle
box = cv2.boxPoints(rect)
# # normalize coordinates to integers
box =np.int0(box)
# # 注：OpenCV没有函数能直接从轮廓信息中计算出最小矩形顶点的坐标。所以需要计算出最小矩形区域，
# # 然后计算这个矩形的顶点。由于计算出来的顶点坐标是浮点型，但是所得像素的坐标值是整数（不能获取像素的一部分），
# # 所以需要做一个转换
# # draw contours
cv2.drawContours(img, [box], 0, (0, 0, 255), 3)  # 画出该矩形
s1 = np.array([[1],[2],[3]])
s2 = np.array([[2],[3],[4]])
w1 = s1/(s1+s2)
print(w1)
cv2.imshow("1111",img)

# s1 = [1,2,3,4,5,6]
# s2 = [2,3,4,5,6,7]
# s3 = [3,4,5,6,7,8]
# w1 = s1/(s1+s2+s3)
# print(w1)
cv2.waitKey(0)