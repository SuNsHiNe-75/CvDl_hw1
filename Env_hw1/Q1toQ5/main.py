from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
import numpy as np
import cv2
import glob
import tkinter as tk
from tkinter import filedialog
import os
from PyQt5.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
import random
import torch
from torchsummary import summary
from torchvision import models
from torchvision import transforms
from PIL import Image, ImageEnhance
from UIsetting import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
		# in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        #TODO
        self.ui.pushButton.clicked.connect(self.load_folder)
        self.ui.pushButton_2.clicked.connect(self.Image_L)
        self.ui.pushButton_3.clicked.connect(self.Image_R)
        self.ui.pushButton_4.clicked.connect(self.find_coners)
        self.ui.pushButton_5.clicked.connect(self.find_intrinsic)
        self.ui.pushButton_6.clicked.connect(self.find_extrinsic)
        self.ui.pushButton_7.clicked.connect(self.find_distortion)
        self.ui.pushButton_8.clicked.connect(self.show_result)
        self.ui.pushButton_9.clicked.connect(self.show_words_on_board)
        self.ui.pushButton_10.clicked.connect(self.show_words_vertically)
        self.ui.pushButton_11.clicked.connect(self.stereo_disparity_map)
        self.ui.pushButton_12.clicked.connect(self.load_Image_1)
        self.ui.pushButton_13.clicked.connect(self.load_Image_2)
        self.ui.pushButton_14.clicked.connect(self.keypoints)
        self.ui.pushButton_15.clicked.connect(self.matched_keypoints)
        self.ui.pushButton_16.clicked.connect(self.load_Image)
        self.ui.pushButton_17.clicked.connect(self.show_data_augmentation)
        self.ui.pushButton_18.clicked.connect(self.show_model_structure)
        self.ui.pushButton_19.clicked.connect(self.show_accuracy)
        self.ui.pushButton_20.clicked.connect(self.show_inference)

    def load_folder(self):
        root = tk.Tk()
        root.withdraw()
        self.file_path = filedialog.askdirectory()
        print(self.file_path)
        
    def Image_L(self):
        self.imgL = QFileDialog.getOpenFileName(self)
        
    def Image_R(self):
        self.imgR = QFileDialog.getOpenFileName(self)
        
    
    #1
    def find_coners(self):
        chessboardSize = (11, 8)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0],
                               0:chessboardSize[1]].T.reshape(-1, 2)

        size_of_chessboard_squares_mm = 20
        objp = objp * size_of_chessboard_squares_mm

        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        
        images = glob.glob(os.path.join(self.file_path,"*"))
        
        for fname in images:
            # Because of Chinese path
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(self.gray, chessboardSize, None)

            if ret == True:
                self.objpoints.append(objp)
                corners = cv2.cornerSubPix(
                    self.gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, chessboardSize, corners, ret)
                cv2.namedWindow("Find Coners", 0)
                cv2.resizeWindow("Find Coners", 500, 500)
                cv2.imshow("Find Coners", img)
                cv2.waitKey(500)
                
        cv2.destroyAllWindows()


    def find_intrinsic(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        print("Intrinsic:")
        print(mtx)        

    def find_extrinsic(self):
        msg = int(self.ui.comboBox.currentText()) - 1

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
    
        rmtx = cv2.Rodrigues(rvecs[msg])
        extrinsic = np.c_[rmtx[0], tvecs[msg]]
        print("Extrinsic:")
        print(extrinsic)

    def find_distortion(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        print("Distortion:")
        print(dist)

    def show_result(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        images = glob.glob(os.path.join(self.file_path,"*"))
        for fname in images:
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 0, (w, h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            result = np.hstack((img, dst))
            cv2.namedWindow("Show Result", 0)
            cv2.resizeWindow("Show Result", 1000, 500)
            cv2.imshow("Show Result", result)
            cv2.waitKey(500)
            
        cv2.destroyAllWindows()


    #2
    def show_words_on_board(self):
        chessboardSize = (11, 8)
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0],
                               0:chessboardSize[1]].T.reshape(-1, 2)

        fs = cv2.FileStorage(self.file_path+'/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        words = self.ui.lineEdit.text()
        for i in range(len(words)):
            ch = fs.getNode(words[i]).mat()

            if i == 0:
                # 1d to 3d
                axis_1 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_1)):
                    # 一格3單位
                    axis_1[j] += [7,5,0]
            
            elif i == 1:
                axis_2 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_2)):
                    axis_2[j] += [4,5,0]
                    
            elif i == 2:
                axis_3 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_3)):
                    axis_3[j] += [1,5,0]

            elif i == 3:
                axis_4 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_4)):
                    axis_4[j] += [7,2,0]

            elif i == 4:
                axis_5 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_5)):
                    axis_5[j] += [4,2,0]

            elif i == 5:
                axis_6 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_6)):
                    axis_6[j] += [1,2,0]
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        chess_images = glob.glob(os.path.join(self.file_path,"*.bmp"))
        # Select any index to grab an image from the list
        for i in range(len(chess_images)):
            # Read in the image
            image = cv2.imread(chess_images[i])
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret == True:
                objpoints.append(objp)
                # objp = 8 * 11 objpoints (x, y, z)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
                # corner2 = each object point on 2D image (x, y)
                # gray.shape[::-1] = (2048, 2048)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

                imgpts_1 = None
                imgpts_2 = None
                imgpts_3 = None
                imgpts_4 = None
                imgpts_5 = None
                imgpts_6 = None
                # project 3D points to image plane
                for j in range(len(words)):
                    if j == 0:
                        imgpts_1, jac = cv2.projectPoints(axis_1, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==1:
                        imgpts_2, jac = cv2.projectPoints(axis_2, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==2:
                        imgpts_3, jac = cv2.projectPoints(axis_3, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==3:
                        imgpts_4, jac = cv2.projectPoints(axis_4, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==4:
                        imgpts_5, jac = cv2.projectPoints(axis_5, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==5:
                        imgpts_6, jac = cv2.projectPoints(axis_6, rvecs[i], tvecs[i], mtx, dist)

                def draw_1(image, imgpts_1):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_2(image, imgpts_1 , imgpts_2):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image
                    
                def draw_3(image, imgpts_1 , imgpts_2, imgpts_3):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_4(image, imgpts_1 , imgpts_2, imgpts_3, imgpts_4):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_4), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_4[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_4[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_5(image, imgpts_1 , imgpts_2, imgpts_3, imgpts_4, imgpts_5):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_4), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_4[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_4[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_5), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_5[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_5[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_6(image, imgpts_1 , imgpts_2, imgpts_3, imgpts_4, imgpts_5, imgpts_6):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_4), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_4[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_4[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_5), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_5[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_5[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_6), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_6[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_6[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                if len(words) == 1:
                    img = draw_1(image, imgpts_1)
                elif len(words) == 2:
                    img = draw_2(image, imgpts_1, imgpts_2)
                elif len(words) == 3:
                    img = draw_3(image, imgpts_1, imgpts_2, imgpts_3)
                elif len(words) == 4:
                    img = draw_4(image, imgpts_1, imgpts_2, imgpts_3, imgpts_4)
                elif len(words) == 5:
                    img = draw_5(image, imgpts_1, imgpts_2, imgpts_3, imgpts_4, imgpts_5)
                elif len(words) == 6:
                    img = draw_6(image, imgpts_1, imgpts_2, imgpts_3, imgpts_4, imgpts_5, imgpts_6)

                img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
                cv2.namedWindow('Show Words On Board')
                cv2.imshow('Show Words On Board', img)
                cv2.waitKey(500)      

    def show_words_vertically(self):
        chessboardSize = (11, 8)
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0],
                               0:chessboardSize[1]].T.reshape(-1, 2)
        fs = cv2.FileStorage(self.file_path+'/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        words = self.ui.lineEdit.text()
        
        for i in range(len(words)):
            ch = fs.getNode(words[i]).mat()
            
            if i == 0:
                axis_1 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_1)):
                    axis_1[j] += [7,5,0]

            elif i == 1:
                axis_2 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_2)):
                    axis_2[j] += [4,5,0]
                    
            elif i == 2:
                axis_3 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_3)):
                    axis_3[j] += [1,5,0]

            elif i == 3:
                axis_4 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_4)):
                    axis_4[j] += [7,2,0]

            elif i == 4:
                axis_5 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_5)):
                    axis_5[j] += [4,2,0]

            elif i == 5:
                axis_6 = np.float32(ch).reshape(-1, 3)
                for j in range(len(axis_6)):
                    axis_6[j] += [1,2,0]
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane
        chess_images = glob.glob(os.path.join(self.file_path,"*.bmp"))
        #chess_images = glob.glob(self.file_path)
        # Select any index to grab an image from the list
        for i in range(len(chess_images)):
            # Read in the image
            image = cv2.imread(chess_images[i])
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret == True:
                objpoints.append(objp)
                # objp = 8 * 11 objpoints (x, y, z)
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                imgpoints.append(corners2)
                # corner2 = each object point on 2D image (x, y)
                # gray.shape[::-1] = (2048, 2048)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

                imgpts_1 = None
                imgpts_2 = None
                imgpts_3 = None
                imgpts_4 = None
                imgpts_5 = None
                imgpts_6 = None
                # project 3D points to image plane
                for j in range(len(words)):
                    if j == 0:
                        imgpts_1, jac = cv2.projectPoints(axis_1, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==1:
                        imgpts_2, jac = cv2.projectPoints(axis_2, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==2:
                        imgpts_3, jac = cv2.projectPoints(axis_3, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==3:
                        imgpts_4, jac = cv2.projectPoints(axis_4, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==4:
                        imgpts_5, jac = cv2.projectPoints(axis_5, rvecs[i], tvecs[i], mtx, dist)
                    elif j ==5:
                        imgpts_6, jac = cv2.projectPoints(axis_6, rvecs[i], tvecs[i], mtx, dist)

                def draw_1(image, imgpts_1):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_2(image, imgpts_1 , imgpts_2):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image
                    
                def draw_3(image, imgpts_1 , imgpts_2, imgpts_3):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_4(image, imgpts_1 , imgpts_2, imgpts_3, imgpts_4):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_4), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_4[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_4[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_5(image, imgpts_1 , imgpts_2, imgpts_3, imgpts_4, imgpts_5):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_4), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_4[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_4[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_5), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_5[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_5[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                def draw_6(image, imgpts_1 , imgpts_2, imgpts_3, imgpts_4, imgpts_5, imgpts_6):
                    for j in range(0, len(axis_1), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_1[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_1[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_2), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_2[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_2[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_3), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_3[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_3[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_4), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_4[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_4[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_5), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_5[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_5[j + 1].ravel()) ))), (0, 0, 255), 5)
                    for j in range(0, len(axis_6), 2):
                        image = cv2.line(image, tuple(tuple(map(int, tuple(imgpts_6[j].ravel()) ))), tuple(tuple(map(int, tuple(imgpts_6[j + 1].ravel()) ))), (0, 0, 255), 5)
                    return image

                if len(words) == 1:
                    img = draw_1(image, imgpts_1)
                elif len(words) == 2:
                    img = draw_2(image, imgpts_1, imgpts_2)
                elif len(words) == 3:
                    img = draw_3(image, imgpts_1, imgpts_2, imgpts_3)
                elif len(words) == 4:
                    img = draw_4(image, imgpts_1, imgpts_2, imgpts_3, imgpts_4)
                elif len(words) == 5:
                    img = draw_5(image, imgpts_1, imgpts_2, imgpts_3, imgpts_4, imgpts_5)
                elif len(words) == 6:
                    img = draw_6(image, imgpts_1, imgpts_2, imgpts_3, imgpts_4, imgpts_5, imgpts_6)

                img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
                cv2.namedWindow('Show Words Vertically')
                cv2.imshow('Show Words Vertically', img)
                cv2.waitKey(500)        


    #3
    def stereo_disparity_map(self):
        baseline=342.789
        focal_length=4019.284
        doffs=279.184
        
        def draw_circle(event,x,y,flags,param):            
            if event == cv2.EVENT_LBUTTONDOWN:
                img = cv2.cvtColor(np.copy(disparity),cv2.COLOR_GRAY2BGR)
                img_dot = cv2.cvtColor(np.copy(disparity) ,cv2.COLOR_GRAY2BGR)
                cv2.circle(img_dot,(x,y),10,(255,0,0),-1)
                z=img[y][x][0]
                imgR_dot = cv2.imdecode(np.fromfile(self.imgR[0], dtype=np.uint8), -1)
                imgR_dot = cv2.cvtColor(imgR_dot, cv2.COLOR_RGB2BGR)
                # imgR_dot = cv2.imread(self.imgR[0])
                
                if img[y][x][0] != 0:       
                    cv2.circle(imgR_dot,(x-z,y),25,(0,255,0),-1)
                    print("(",x,", ",y,")",",dis:",z)
                else:
                    print("Failure case")
                
                cv2.namedWindow('imgR_dot',cv2.WINDOW_NORMAL)
                cv2.resizeWindow("imgR_dot", int(imgR_dot.shape[1]/3), int(imgR_dot.shape[0]/3))
                cv2.imshow('imgR_dot', imgR_dot)
                cv2.waitKey(0)
                
        imgL = cv2.imdecode(np.fromfile(self.imgL[0], dtype=np.uint8), -1)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        # imgL = cv2.imread(self.imgL[0],0)
        imgR = cv2.imdecode(np.fromfile(self.imgR[0], dtype=np.uint8), -1)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
        # imgR = cv2.imread(self.imgR[0],0)

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL,imgR)
        disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.namedWindow('disparity',cv2.WINDOW_NORMAL)
        cv2.resizeWindow("disparity", int(disparity.shape[1]/4), int(disparity.shape[0]/4))
        cv2.imshow('disparity', disparity)   

        # imgL = cv2.imread(self.imgL[0])
        imgL = cv2.imdecode(np.fromfile(self.imgL[0], dtype=np.uint8), -1)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('imgL',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imgL', int(disparity.shape[1]/3), int(disparity.shape[0]/3))
        cv2.setMouseCallback('imgL',draw_circle)
        cv2.imshow('imgL',imgL)

        # imgR = cv2.imread(self.imgR[0])
        imgR = cv2.imdecode(np.fromfile(self.imgR[0], dtype=np.uint8), -1)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('imgR_dot',cv2.WINDOW_NORMAL)
        cv2.resizeWindow("imgR_dot", int(imgR.shape[1]/3), int(imgR.shape[0]/3))
        cv2.imshow('imgR_dot', imgR)
        
        cv2.waitKey(0)
    

    #4
    def load_Image_1(self):
        self.img1 = QFileDialog.getOpenFileName(self)
        
    def load_Image_2(self):
        self.img2 = QFileDialog.getOpenFileName(self)
        
    def keypoints(self):
        img = cv2.imdecode(np.fromfile(self.img1[0], dtype=np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 透過轉換函式轉為灰階影像
        color = (0, 255, 0)  # 定義框的顏色
        # SIFT特徵計算
        sift = cv2.xfeatures2d.SIFT_create() # Create a SIFT detector
        keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

        img = cv2.drawKeypoints(grayImg,keypoints_1,img,(0,255,0))
        cv2.namedWindow('keypoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('keypoints', img.shape[1]//3,img.shape[0]//3)
        cv2.imshow('keypoints', img)
        cv2.waitKey(0)#等待按键按下
        cv2.destroyAllWindows()#清除所有窗口
        
    def matched_keypoints(self):
        img1 = cv2.imdecode(np.fromfile(self.img1[0], dtype=np.uint8), -1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.imdecode(np.fromfile(self.img2[0], dtype=np.uint8), -1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # 透過轉換函式轉為灰階影像
        grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # 透過轉換函式轉為灰階影像
        color = (0, 255, 0)  # 定義框的顏色
        # SIFT特徵計算
        sift = cv2.xfeatures2d.SIFT_create() # Create a SIFT detector
        psd_kp1, psd_des1 = sift.detectAndCompute(grayImg1, None)
        psd_kp2, psd_des2 = sift.detectAndCompute(grayImg2, None)
        # Flann特徵匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
            
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(psd_des1, psd_des2, k=2)
        
        matchesMask = [[0,0] for i in range(len(matches))]
        
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(
                           matchesMask = matchesMask,
                           flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        img3 = cv2.drawMatchesKnn(grayImg1, psd_kp1, grayImg2, psd_kp2, matches, None, **draw_params)
        # plt.imshow(img3,),plt.show()
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('match', img3.shape[1]//3,img3.shape[0]//3)
        cv2.imshow('match', img3)
        cv2.waitKey(0)#等待按键按下
        cv2.destroyAllWindows()#清除所有窗口


    #5
    def load_Image(self):
        self.img = QFileDialog.getOpenFileName(self)
        myPixmap = QtGui.QPixmap(self.img[0])
        myScaledPixmap = myPixmap.scaled(self.ui.label_3.size(), Qt.KeepAspectRatio)
        self.ui.label_3.setPixmap(myScaledPixmap)

    def show_data_augmentation(self):
        self.file_path = filedialog.askdirectory()
        image_files = os.listdir(self.file_path)
        image_files = random.sample(image_files, 9)
        images = [Image.open(os.path.join(self.file_path, img)) for img in image_files]

        augmented_images = []
        data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.6),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees = 50,center=(0,0)),
            transforms.RandomResizedCrop(size = (200,200),scale=(0.2, 1.0), ratio=(0.5, 1.1)),
        ])

        for img in images:
            augmented_img = data_augmentation(img)
            augmented_images.append(augmented_img)

        plt.figure(figsize=(12, 12))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[i])
            filename = os.path.splitext(image_files[i])[0]
            plt.title(filename)

        plt.show()
        
    def show_model_structure(self):
        model = models.vgg19_bn()
        summary(model, (3, 32, 32)) 
        
    def show_accuracy(self):
        img = cv2.imread("./acc_and_loss.png")
        plt.figure(figsize=(15,8))
        ax = plt.subplot(1,1,1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.show()

    def show_inference(self):
        # 載入模型
        model_path = 'vgg19_bn_epoch80.pt'
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        # 載入圖片進行推論
        image = Image.open(self.img[0])

        # 定義資料轉換，與模型訓練時的轉換相同
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # 添加 batch 維度

        # 進行推論
        with torch.no_grad():
            outputs = model(image)
            _, predicted = outputs.max(1)

        # 類別名稱對應的字典
        class_names = []
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

        # 取得預測的類別名稱
        predicted_class = predicted.item()
        predicted_class_name = class_names[predicted_class]
        self.ui.label_2.setText(predicted_class_name)

        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        probabilities = probabilities.squeeze(0).numpy()

        plt.bar(class_names, probabilities)
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title('Probability Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())