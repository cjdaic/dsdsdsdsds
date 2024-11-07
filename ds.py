import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance
import cv2
import threading
import time
import numpy as np

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Camera Processing")

        # 创建左右两个面板
        self.left_panel = tk.Frame(root)
        self.left_panel.pack(side="left", padx=10, pady=10)

        self.right_panel = tk.Frame(root)
        self.right_panel.pack(side="right", padx=10, pady=10)

        # 左侧按钮：选文件和读取摄像头
        self.select_file_button = tk.Button(self.left_panel, text="Select File", command=self.select_file)
        self.select_file_button.pack()

        self.read_camera_button = tk.Button(self.left_panel, text="Read Camera", command=self.read_camera)
        self.read_camera_button.pack()

        # 显示原始图像和处理后图像的面板
        self.left_image_label = tk.Label(self.left_panel)
        self.left_image_label.pack()

        self.right_image_label = tk.Label(self.right_panel)
        self.right_image_label.pack()

        self.process_options = ["Original", "Original Gray", "Original Blurred", "Original Edged", "Outline", "Thresh Binary", "Thresh mean", "Thresh gauss", "Otsu", "dst"]
        self.selected_option = tk.StringVar(value=self.process_options[0])

        self.rotate_button = tk.Button(self.right_panel, text="↻", command=self.rotate_image)
        self.rotate_button.pack()
        
        self.option_menu = tk.OptionMenu(self.right_panel, self.selected_option, *self.process_options, command=self.start_processing)
        self.option_menu.pack()

        self.current_image = None
        self.processed_image = None
        self.is_running = False

    def select_file(self):
        # 选择文件并显示图片
        file_path = filedialog.askopenfilename()
        if file_path:
            pil_img = Image.open(file_path)  # PIL 图像
            self.current_image = np.array(pil_img)  # 转换为 NumPy 数组
            self.show_image(self.current_image, self.left_image_label)
            self.start_processing(self.selected_option.get())  # 处理选择的图像

    def read_camera(self):
        # 从摄像头读取图片
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
                self.current_image = frame  # OpenCV 的 cv2.VideoCapture 返回的是 NumPy 数组
                self.show_image(self.current_image, self.left_image_label)
                self.start_processing(self.selected_option.get())
        cap.release()

    def rotate_image(self):
        if self.processed_image is not None:
            self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.show_image(self.processed_image, self.right_image_label)
    
    def show_image(self, img, label):
        # 显示图像在指定标签上
        img_resized = cv2.resize(img, (800, 800))  # 调整图像大小
        img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        tk_image = ImageTk.PhotoImage(img_pil)
        label.config(image=tk_image)
        label.image = tk_image


    def rectify(self,h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew

    def process_image(self,img,option):
        image = cv2.resize(img, (1500, 880))

        # 不破坏原图
        orig = image.copy()

        # 获得灰度图和高斯滤波图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #blurred = cv2.medianBlur(gray, 5)#中值滤波

        # 用Canny算子对滤波后的图进行边缘检测
        edged = cv2.Canny(blurred, 0, 50)
        orig_edged = edged.copy()

        # 寻找图中最大的轮廓，使用cv2.RETR_LIST（无轮廓等级hierarchy_区分）来寻找轮廓，cv2.CHAIN_APPROX_NONE即获取轮廓的每个像素，将轮廓排序
        (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        #x,y,w,h = cv2.boundingRect(contours[0]) #获得一个图像的最小矩形边框一些信息
        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0) #配合cv2.rectangle()可以画出该最小边框

        # 获取近似四边形轮廓(以 0.02*周长p 为逼近精度)
        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                target = approx
                break


        # 将轮廓近似矩阵矫正为800x800像素的浮点数矩形矩阵(见rectify函数)
        approx = self.rectify(target)
        pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]]) #四个顶点
        M = cv2.getPerspectiveTransform(approx,pts2)        #得到最关键的透视变换矩阵
        dst = cv2.warpPerspective(orig,M,(800,800))         #透视变换
        
        #画出处理后的矩形轮廓
        cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)


        # 通过调节不同阈值获得不同扫描效果
        ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if self.current_image is not None:
            if option == "Original Gray":
                self.processed_image = gray
            elif option == "Original Blurred":
                self.processed_image = blurred
            elif option == "Original Edged":
                self.processed_image = orig_edged
            elif option == "Outline":
                self.processed_image = image
            elif option == "Thresh Binary":
                self.processed_image = th1
            elif option == "Thresh mean":
                self.processed_image = th2
            elif option == "Thresh gauss":
                self.processed_image = th3
            elif option == "Otsu":
                self.processed_image = th4
            elif option == "dst":
                self.processed_image = dst                                           
            else:  # "Original"
                self.processed_image = self.current_image
            self.show_image(self.processed_image, self.right_image_label)
        

    def start_processing(self, option):
        # 根据选择处理图像
        if self.current_image is not None:
            self.process_image(self.current_image, option)


# 创建并运行主程序
root = tk.Tk()
app = App(root)
root.mainloop()