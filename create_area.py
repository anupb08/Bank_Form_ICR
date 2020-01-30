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


import re 
import sys
import os
import cv2
import math
#from pylsd.pylsd.lsd import lsd

from hocr_parser import HOCRDocument

class RectCoord:
    def __init__(self, coord, text, parent=None):
        self.coord = coord
        self.parentCoord = parent
        self.isVisited = False
        self.text = text

    def setVisited(self):
        self.isVisited = True

    def isVisited(self):
        return self.isVisited

    def getCoord(self):
        return self.coord

def splitRectangle(area, gap=100):
    splitedRect = []
    text = ' '
    for par in area.paragraphs:
        for line in par.lines:
            lastCoord = (line.words)[0].coordinates
            startCoord = (line.words)[0].coordinates
            text = ((line.words)[0]).ocr_text + ' '
            for word in line.words[1:]:
                coord = word.coordinates
                text = text + word.ocr_text + ' '
                if abs(lastCoord[2] - coord[0]) > gap:
                    rectObj = RectCoord([(startCoord[0], startCoord[1]), (lastCoord[2], lastCoord[3])], text)
                    splitedRect.append(rectObj)
                    startCoord = coord
                    text = ' '
                lastCoord = coord
            rectObj = RectCoord([(startCoord[0], startCoord[1]), (lastCoord[2], lastCoord[3])], text)
            splitedRect.append(rectObj)
            text = ' '

    return splitedRect


def getLayoutInfo(hocrFile):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(dir_path, hocrFile)
    document = HOCRDocument(full_path, is_path=True)
    page = document.pages[0]
    print(page.imageFilename)
    title = page.imageFilename
    PATTERN = re.compile('image\s\"\w+.\w+\"')
    imgfile = PATTERN.search(title)
    
    #print(imgfile.group())
    print(page.nareas)
    areaCoords = []
    for area in page.areas:
        coord = area.coordinates
        (x1,y1,x2,y2) = coord
        #print(area.id)
        if abs(x1-x2) < 20 or abs(y1 - y2) < 10:
            continue
        #if abs(x1-x2) > 1600 or abs(y1 - y2) > 1000:
        #    continue
        #cv2.rectangle(image,(x1,y1),(x2,y2),(200,10,10),2)
        coords = splitRectangle(area, 40)
        #areaCoords.append([(x1,y1),(x2,y2)])
        areaCoords.extend(coords)

    areaCoords.sort(key=lambda x: x.coord[0][1])
    rectCoords = []
    #for areaCoord in areaCoords:
    #    rectCoord = RectCoord(areaCoord)
    #    rectCoords.append(rectCoord)
    area_rect_coords = areaCoords #mergeRect(areaCoords, h_dist=5, w_dist=40)
    
    return area_rect_coords

def mergeRect(rectCoords, h_dist=2, w_dist=2):
    newRectCoord = []

    for i, outerRectCoord1 in enumerate(rectCoords):
        if outerRectCoord1.isVisited == True:
            continue
        outerRectCoord1.isVisited = True
        rectCoord1 = outerRectCoord1.getCoord()
        topRectCoord = rectCoord1[0]
        lastBottomRect = rectCoord1[1]
        for j, innerRectCoord in enumerate(rectCoords):
            break
            if innerRectCoord.isVisited == True:
                continue
            rectCoord2 = innerRectCoord.getCoord()
            if abs(lastBottomRect[1] - rectCoord2[0][1]) > 5: #h_dist:
                #newRectCoord.append([topRectCoord, lastBottomRect])
                continue
            if  rectCoord2[0][1] > lastBottomRect[1] and (abs((rectCoord2[1][0] + rectCoord2[0][0])/2 -  (rectCoord1[1][0] + rectCoord1[0][0])/2) < w_dist or (abs(rectCoord2[0][0] - rectCoord1[0][0]) <5 and rectCoord2[1][0] <= rectCoord1[1][0]) ) and (rectCoord2[0][1] - lastBottomRect[1]) < h_dist:
            #if  rectCoord2[0][1] > lastBottomRect[1] and  (abs(rectCoord2[0][0] - rectCoord1[0][0]) <5 and rectCoord2[1][0] <= rectCoord1[1][0])   and (rectCoord2[0][1] - lastBottomRect[1]) < dist:
                if rectCoord2[0][0] < topRectCoord[0]:
                    temp = (rectCoord2[0][0], topRectCoord[1])
                    topRectCoord = temp
                if rectCoord2[1][0] > lastBottomRect[0]:
                    lastBottomRect = rectCoord2[1]
                else:
                    temp = (lastBottomRect[0], rectCoord2[1][1])
                    lastBottomRect = temp
                innerRectCoord.isVisited = True

        newRectCoord.append([topRectCoord, lastBottomRect])

    return newRectCoord

