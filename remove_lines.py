# @Date    : 2018-08-15 
# @Author  : Anup Bera (anupbera@gmail.com)
'''
 [PhoneticsLab] LLC ("COMPANY") CONFIDENTIAL
 Unpublished Copyright (c) 2018-2019 [COMPANY NAME], All Rights Reserved.

 NOTICE:  All information contained herein is, and remains the property of AUTHOR. The intellectual and technical concepts contained
 herein are proprietary to AUTHOR and may be covered by U.S. and Foreign Patents, patents in process, and are protected by trade secret or copyright law.
 Dissemination of this information or reproduction of this material is strictly forbidden unless prior written permission is obtained
 from Author.  Access to the source code contained herein is hereby forbidden to anyone except current PhoneticsLab employees, managers or contractors who have executed 
 Confidentiality and Non-disclosure agreements explicitly covering such access.

 The copyright notice above does not evidence any actual or intended publication or disclosure  of  this source code, which includes  
 information that is confidential and/or proprietary, and is a trade secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC  PERFORMANCE, 
 OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS  SOURCE CODE  WITHOUT  THE EXPRESS WRITTEN CONSENT OF COMPANY IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS  
 TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                
'''


import cv2
from skimage.filters import threshold_li, threshold_sauvola, threshold_local, threshold_otsu, threshold_minimum
from skimage import img_as_ubyte
import numpy as np

def image_skeleton(img, kernel):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    #ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel)
    done = False
    last_zeros = 0
     
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
     
        zeros = size - cv2.countNonZero(img)  
        if zeros>=size or last_zeros == zeros:
            done = True
        last_zeros = zeros
 
    return skel

def get_lines_and_without_lines_image(img, size=[100,100]):
    #img = cv2.imread("Result/bank-0.png")
    #img = cv2.imread(img)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    #img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    #element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)
    #th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    #th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,-2)
    th2 = threshold_li(img)
    th2 = (img > th2)*255
    th2 = img_as_ubyte(th2)
    kernel = np.ones((2,7),np.uint8)
    th_v = cv2.dilate(th2,kernel,iterations =2 )
    kernel = np.ones((2,3),np.uint8)
    th_h = cv2.dilate(th2,kernel,iterations =1 )
    
    horizontal = th2
    vertical = th2
    rows,cols = horizontal.shape
    horizontalsize = size[0]
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal_skel = cv2.erode(th_h, horizontalStructure, (-1, -1))
    horizontal_skel = cv2.dilate(horizontal_skel, horizontalStructure, (-1, -1))
    horizontal_skel = image_skeleton(horizontal_skel, (2,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal_inv = cv2.bitwise_not(horizontal)
    masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
    masked_img_inv = cv2.bitwise_not(masked_img)
    horizontal_mask = masked_img_inv
    #cv2.imwrite("horizontal.jpg", masked_img_inv)
    
    
    verticalsize = size[1]
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical_skel = cv2.erode(th_v, verticalStructure, (-1, -1))
    vertical_skel = cv2.dilate(vertical_skel, verticalStructure, (-1, -1))
    vertical_skel = image_skeleton(vertical_skel, (5,1))
    vertical_skel = cv2.bitwise_not(vertical_skel)
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    vertical_inv = cv2.bitwise_not(vertical)
    masked_img = cv2.bitwise_and(img, img, mask=vertical_inv)
    masked_img_inv = cv2.bitwise_not(masked_img)
    vertical_mask = masked_img_inv
    #cv2.imwrite("vertical.jpg", masked_img_inv)
    
    
    final_without_lines = cv2.bitwise_or(horizontal_mask, vertical_mask)
    final_only_lines = cv2.bitwise_xor(horizontal_mask, vertical_mask)
    cv2.imwrite("final_without_lines.png", final_without_lines)
    cv2.imwrite("final_only_lines.png", final_only_lines)
    cv2.imwrite("final_only_lines_h.png", horizontal_skel)
    cv2.imwrite("final_only_lines_v.png", vertical_skel)
    #final_without_lines = cv2.bitwise_not(final_without_lines)
    #th2 = cv2.adaptiveThreshold(final_without_lines,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    ret,final_without_lines = cv2.threshold(final_without_lines,127,255,0)
    #th2 = threshold_minimum(final_without_lines)
    #th2 = (img > th2)*255
    #th2 = img_as_ubyte(th2)
    #final_without_lines = cv2.bitwise_not(th2)
    
    kernel = np.ones((2,2),np.uint8)
    final_without_lines = cv2.morphologyEx(final_without_lines, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("final_thresholded.png", final_without_lines)

    '''
    th2 = cv2.medianBlur(th2,7)
    th2_1 = cv2.bitwise_not(th2)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    th2 = cv2.erode(th2_1, verticalStructure, (-1, -1), iterations=2)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    th2 = cv2.erode(th2, horizontalStructure, (-1, -1), iterations=1)
    th2 = cv2.bitwise_not(th2)
    th2 = cv2.bitwise_or(th2_1,th2)
    #th2 = cv2.bitwise_xor(th2_1,th2)
    cv2.imwrite('threshold.png', th2)
    '''

    return horizontal_skel, vertical_skel, final_without_lines, th2 

#img = cv2.imread('temp_0.png')
#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#get_lines_and_without_lines_image(img, [40,40])
