# @Date    : 2018-08-15 
# @Author  : Anup Bera (anupbera@gmail.com)

import cv2
import numpy as np
import os
import subprocess
from pylsd.lsd import lsd
import math
import time
import datetime
import json
from create_area import getLayoutInfo
from predict import getPredictionResult
from deskew  import Deskew
from skimage.filters import threshold_sauvola, threshold_li, threshold_local, threshold_minimum
from skimage import img_as_ubyte
from wand.image import Image
#import ghostscript
from remove_lines import get_lines_and_without_lines_image
from read_config_file import * 
from postprocess import *


seg_tolerance = (5, 2, 10)
line_tolerance = (8,3,30)
box_size_min = [60,60]
box_size_max = [100,100]
tolerance2 = [5, 30]
expired_date= 1552030589.0 + 3600*24*60

class EventPoint:
    def __init__(self, point, lineSegment, direction):
        self.point = point
        self.lineSegment = lineSegment
        self.direction = direction


class LinePoint:
    def __init__(self, point, type=0):
        self.XCoord = point[0]
        self.YCoord = point[1]
        self.is_end_point = True
        self.intersection_type = type # 0 -> not intersect, 1 -> cross intersect, 2 -> touch top , 3 -> touch bottom, 4 -> touch left, 5 -> touch right
        self.hSegment = None
        self.vSegment = None

    def isEqual(self, point):
        if self.XCoord == point.XCoord and self.YCoord == point.YCoord:
            return True

        return False

    def setIntersectionType(self, type=0):
        self.intersection_type = type

    def getIntersectionType(self):
        return self.intersection_type

    def getCoord(self):
        return (self.XCoord, self.YCoord)

    def setHorizontalSegment(self, hSegment):
        self.hSegment = hSegment

    def setVerticalSegment(self, vSegment):
        self.vSegment= vSegment

class LineSegment:
    def __init__(self, segPoints=None, linePoints=None):
        self.orientation = None
        self.length = 0
        self.intersectionLines = []
        self.intersectionPoints = []
        self.points = self.setPoints(segPoints, linePoints)
        self.startAndEndPoint = self.setStartAndEndPoint()
        self.pointsCoord = self.setPointsCoord()
        self.currentPointIndex = 0
        self.isSorted = False

    def nextPoint(self, point=None):
        if point is not None:
            index = self.getIndex(point)
            if index is None:
                return None
            self.currentPointIndex = index

        if self.currentPointIndex == len(self.points) -1:
            return None
        next_point = self.points[self.currentPointIndex +1]
        self.currentPointIndex += 1
        return next_point


    def prevPoint(self, point=None):
        if point is not None:
            index = self.getIndex(point)
            if index is None:
                return None
            self.currentPointIndex = index

        if self.currentPointIndex == 0:
            return None
        prev_point = self.points[self.currentPointIndex -1]
        self.currentPointIndex -= 1
        return prev_point

    def getIndex(self, point):
        for i, pt in enumerate(self.points):
            if pt.isEqual(point) is True:
                return i

        return None

    def sort(self):
        self.startAndEndPoint = self.setStartAndEndPoint()
    
    def setPoints(self, segPoints, linePoints):
        if linePoints is None:
            points = []
            for pt in segPoints:
                points.append(LinePoint(pt[0]))
                points.append(LinePoint(pt[1]))
            return points
        else:
            return linePoints


    def getPoints(self):
        return self.points

    def setPointsCoord(self):
        coords = []
        for point in self.points:
            #coords.append((point.XCoord, point.YCoord))
            coords.append(point.getCoord())

        return coords

    def getPointsCoord(self):
        if self.isSorted is False:
            self.sort()
        return self.pointsCoord

    def setIntersectionPoints(self, points):
        self.intersectionPoints = points

    def setIntersectionLines(self, lineSegments):
        self.intersectionLines = lineSegments

    def getIntersectionPoints(self):
        return self.intersectionPoints

    def getIntersectionLines(self):
        return self.intersectionLines

    def getStartAndEndPoints(self):
        return self.startAndEndPoint

    def getStartAndEndPointCoords(self):
        start = self.startAndEndPoint[0].XCoord, self.startAndEndPoint[0].YCoord
        end = self.startAndEndPoint[1].XCoord, self.startAndEndPoint[1].YCoord
        return (start, end)

    def setStartAndEndPoint(self):
        #for pt in linePoints:
        #    points.append(pt[0])
        #    points.append(pt[1])
        points_x = self.points[:]
        points_x.sort(key=lambda tup: tup.XCoord)
        point1_x = points_x[0]
        point2_x = points_x[-1]
        l1 = abs(point1_x.XCoord - point2_x.XCoord)
        points_y= self.points[:]
        points_y.sort(key=lambda tup: tup.YCoord)
        point1_y = points_y[0]
        point2_y = points_y[-1]
        l2 = abs(point1_y.YCoord - point2_y.YCoord)
        if l1 > l2:
            self.orientation = 0
            self.points.sort(key=lambda tup: tup.XCoord)
            self.isSorted = True
            self.length = getLineLength((point1_x.XCoord, point1_x.YCoord) , (point2_x.XCoord, point2_x.YCoord) )
            return (point1_x, point2_x)
        else:
            self.orientation = 1
            self.points.sort(key=lambda tup: tup.YCoord)
            self.isSorted = True
            self.length = getLineLength((point1_y.XCoord, point1_y.YCoord), (point2_y.XCoord, point2_y.YCoord))
            return (point1_y, point2_y)


class Line:
    def __init__(self, points, tolerance=(0,0,0)):
        self.points = points
        self.segments = []
        self.orientation = None
        self.startAndEndPoint = self.setStartAndEndPoint(self.points)
        self.angleWithX = self.setAngleWithX()
        self.isSorted = False
        self.tolerance = tolerance 


    def getStartAndEndPointCoords(self):
        return self.startAndEndPoint
    
    def setStartAndEndPoint(self, linePoints):

        points= []
        for pt in linePoints:
            points.append(pt[0])
            points.append(pt[1])
        points_temp = points
        points_temp.sort(key=lambda tup: tup[0])
        point1_t = points_temp[0]
        point2_t = points_temp[-1]
        l1 = abs(point1_t[0] - point2_t[0])
        points.sort(key=lambda tup: tup[1])
        point1 = points[0]
        point2 = points[-1]
        l2 = abs(point1[1] - point2[1])
        if l1 > l2:
            self.orientation = 0
            return (point1_t, point2_t)
        else:
            self.orientation = 1
            return (point1, point2)

    def getAllSegments(self, fill=False):
        self.splitInSegments(self.points, fill)
        if len(self.segments) > 0:
            return self.segments

    def order(self, point1, point2, axis=0):
        if point1[axis] > point2[axis]:
            temp = point1
            point1 = point2
            point2 = temp
    
        return point1, point2
    
    def splitInSegments(self, points, fill):
        pointsX = []
        pointsY = []
        if self.orientation == 0:
            for point1, point2 in points:
                point1, point2 = self.order(point1, point2, axis=0)
                pointsX.append((point1, point2))
            pointsX.sort(key=lambda tup: tup[0][0])
            self.points = pointsX
            self.isSorted = True
            if fill:
                self.segments = pointsX
                return
            line = []
            point1 = pointsX[0][0]
            point2 = pointsX[0][1]
            lastX =  point2[0]
            line.append((point1, point2))
            for point1, point2 in pointsX[1:]:
                if  (point1[0] -lastX) >self.tolerance[2]:
                    segment = LineSegment(line)
                    self.segments.append(segment)
                    line = []
                if lastX < point2[0]:
                    lastX = point2[0]
                line.append((point1, point2))
            segment = LineSegment(line)
            self.segments.append(segment)
   
        else:
            for point1, point2 in points:
                point1, point2 = self.order(point1, point2, axis=1)
                pointsY.append((point1, point2))
            pointsY.sort(key=lambda tup: tup[0][1])
            self.points = pointsY
            self.isSorted = True
            if fill:
                self.segments = pointsY
                return
            line = []
            point1 = pointsY[0][0]
            point2 = pointsY[0][1]
            lastY =  point2[1]
            line.append((point1, point2))
            for point1, point2 in pointsY[1:]:
                if (point1[1] - lastY)>self.tolerance[2]:
                    segment = LineSegment(line)
                    self.segments.append(segment)
                    line = []
                if lastY < point2[1]:
                    lastY = point2[1]
                line.append((point1, point2))
            segment = LineSegment(line)
            self.segments.append(segment)

    def setAngleWithX(self):
        # not implemented 
        return
    

class FormRectangle:
    def __init__(self, rectangles):
        self.rectangles = rectangles

def getLineLength( pt1, pt2):
    length = math.hypot(pt2[0]-pt1[0], pt2[1] -pt1[1])
    return length
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def angle_between(lineA, lineB):
    
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    if cos_ < -1.0:
        cos_=-1.0
    if cos_>1.0:
        cos_ = 1.0
    # Get angle in radians and then convert to degrees
    #angle = math.acos(dot_prod/magB/magA)
    angle = math.acos(cos_)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 

        return ang_deg



def contains(pts, line2):
    points= pts[:]
    #for pt in pts:
    #    points.append(pt[0])
    #    points.append(pt[1])
    #points.sort(key=lambda tup: tup[0][0])
    nearestLine = points[0]
    lastSmall = 99999
    for line in points:
        small = abs(line[0][0] - line2[0][0])
        if small < lastSmall:
            lastSmall = small
            nearestLine = line
    #line1 = nearestLine
    line1 = pts[-1]
    #line2= Line(pts).startAndEndPoint
    #print(points)

    return isCollinear(line1, line2)

def isCollinear(line1, line2, tolerance=seg_tolerance):
    angle = angle_between(line1,line2)
    p1,p2 = np.asarray(line1)
    p3,p4 = np.asarray(line2)
    if abs(angle) <= tolerance[1] or (angle <= 180 + tolerance[1] and angle >= 180 -tolerance[1]):
        #distance = np.cross((p4 -p3),(p2 -p1))/np.linalg.norm(p4-p3)
        distance = ((p4[1] -p3[1])*p2[0] -(p4[0]-p3[0])*p2[1] + p4[0]*p3[1] - p4[1]*p3[0])/np.linalg.norm(p4-p3)
        #distance = (p4[1] +p3[1]) /2 -p2[1]
        #if line1[0][1] < 50 and line1[1][1] < 50:
        #    print(angle, distance, line1, line2)
        if abs(distance) <= tolerance[0]:
            return True

    return False


def find_intersection( p0, p1, p2, p3 ) :
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]
    denom = float(s10_x * s32_y - s32_x * s10_y)

    if denom == 0 : return None # collinear

    denom_is_positive = denom > 0
    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]
    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == denom_is_positive : return None # no collision

    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive : return None # no collision

    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : return None # no collision
    # collision detected

    t = t_numer / denom
    intersection_point = [int(p0[0] + (t * s10_x)), int(p0[1] + (t * s10_y)) ]

    return intersection_point

'''
def findIntersectionPoints(line1, line2):
    s1 = Segment(line1[0], line1[1])
    s2 = Segment(line2[0], line2[1])
    pt =s1.intersection(s2)
    if len(pt) > 0:
        return (int(pt[0].x), int(pt[0].y))
    else:
        return None
'''

def getIntersectionPointInfo(seg1, seg2, box_tolerance):
    seg1.sort()
    seg2.sort()
    segment1 = seg1.getStartAndEndPoints()
    segment2 = seg2.getStartAndEndPoints()
    
    #######    
    if seg1.orientation == 0:
        adjust = (box_tolerance[0], 0)
    else:
        adjust = (0, box_tolerance[0])

    point11 = tuple(x - y for x, y in zip((segment1[0]).getCoord(), adjust))
    point12 = tuple(x + y for x, y in zip((segment1[1]).getCoord(), adjust))
    if seg2.orientation == 0:
        adjust = (box_tolerance[0], 0)
    else:
        adjust = (0, box_tolerance[0])
    point21 = tuple(x - y for x, y in zip(segment2[0].getCoord(), adjust))
    point22 = tuple(x + y for x, y in zip(segment2[1].getCoord(), adjust))
    #point = findIntersectionPoints((point11,point12),(point21,point22))
    point = find_intersection(point11,point12,point21,point22)
    #######
    point11 = segment1[0].getCoord()
    point12 = segment1[1].getCoord()
    point21 = segment2[0].getCoord()
    point22 = segment2[1].getCoord()
    
    '''
    if (point11[1]  >= point21[1] - box_tolerance[0] or point12[1]  >= point21[1] - box_tolerance[0]) and (point11[1] <= point22[1] + box_tolerance[0] or point12[1] <= point22[1] + box_tolerance[0]):
        if (abs(point21[1] - point11[1]) <= box_tolerance[0] or abs(point21[1] - point12[1]) <= box_tolerance[0]) and point22[1] > point11[1]:
            #point = (point21[0], point21[1]+box_tolerance[0])
            point = point21 #seg2.getStartAndEndPoints()[0].getCoord()
        elif point21[1] < point11[1] and (abs(point22[1] - point11[1]) <= box_tolerance[0] or abs(point22[1] - point12[1]) <= box_tolerance[0]):
            #point = (point21[0], point22[1]-box_tolerance[0])
            point = point22 #seg2.getStartAndEndPoints()[1].getCoord()
        # elif (point11[0] + point12[0])/2 > point21[0] and (point21[1] + point22[1])/2 > point11[1]:
        #     point = (point21[0], point11[1])
        # elif (point11[0] + point12[0])/2 > point22[0] and (point21[1] + point22[1])/2 < point11[1]:
        #     point = (point22[0], point11[1])
        # elif (point11[0] + point12[0])/2 < point21[0] and (point21[1] + point22[1])/2 > point12[1]:
        #     point = (point21[0], point12[1])
        # elif (point11[0] + point12[0])/2 < point22[0] and (point21[1] + point22[1])/2 < point12[1]:
        #     point = (point22[0], point12[1])
        elif abs(point21[0] - point11[0])  <= box_tolerance[0]:
            point = (point21[0], point11[1])
        elif abs(point22[0] - point11[0]) <= box_tolerance[0]:
            point = (point22[0], point11[1])
        elif abs(point22[0] - point12[0]) <= box_tolerance[0]:
            point = (point22[0], point12[1])
        elif abs(point21[0] - point11[0]) <= box_tolerance[0]:
            point = (point21[0], point11[1])
        else:
            point = findIntersectionPoints((point11,point12),(point21,point22))  #((point22[0]+point21[0])/2, (point12[1]+point11[1])/2)
    else:
        point = False
   ''' 

    if point is not None:
        intersection_point = LinePoint(point)
        if seg1.orientation == 0:
            intersection_point.setHorizontalSegment(seg1)
        else:
            intersection_point.setVerticalSegment(seg2)

        if seg1.orientation == 0 and seg2.orientation == 1:
            if point[1] - point21[1] > box_tolerance[1] and point22[1] - point[1]> box_tolerance[1] and point[0] - point11[0] > box_tolerance[1] and point12[0] - point[0]> box_tolerance[1]:
                intersection_point.setIntersectionType(1)
            elif point[1] - point21[1] > box_tolerance[1] and (point[1] - point22[1] <= box_tolerance[0] or point22[1] >= point[1] ):
                intersection_point.setIntersectionType(2)
            elif point22[1] - point[1] > box_tolerance[1] and (abs(point21[1] - point[1]) <= box_tolerance[0] or point21[1] <= point[1]):
                intersection_point.setIntersectionType(3)
            elif point21[0] - point11[0] > box_tolerance[1] and (abs(point21[0] - point12[0]) <= box_tolerance[0] or point21[0] <= point11[0]):
                intersection_point.setIntersectionType(4)
            elif point12[0] - point21[0] > box_tolerance[1] and (abs(point21[0] - point11[0]) <= box_tolerance[0] or point21[0] >= point12[0]):
                intersection_point.setIntersectionType(5)
            else:
                print('Something wrong at ', intersection_point.getCoord(), point21, point22)

            if abs(point21[1] -point[1]) <= box_tolerance[0] and point22[1] >= point[1]:
                start_point_index = seg2.getIndex(segment2[0])
                seg2.getPoints()[start_point_index] = intersection_point
            elif point21[1] < point[1] and (point22[1] - point[1]) <= box_tolerance[0]:
                end_point_index = seg2.getIndex(segment2[1])
                if end_point_index is None:
                    print("wrong")
                seg2.getPoints()[end_point_index] = intersection_point
            else:
                seg2.getPoints().append(intersection_point)

            if abs(point[0] - point11[0]) < box_tolerance[0]  and point12[0] > point[0]:
                start_point_index = seg1.getIndex(segment1[0])
                seg1.getPoints()[start_point_index] = intersection_point
            elif abs(point12[0] - point[0]) < box_tolerance[0] and point11[0]  < point[0]:
                end_point_index = seg1.getIndex(segment1[1])
                seg1.getPoints()[end_point_index] = intersection_point
            else:
                seg1.getPoints().append(intersection_point)
        else:
            seg1.getPoints().append(intersection_point)
            seg2.getPoints().append(intersection_point)
            print("something wrong. Check it")



def condition(pt1, pt2): 
    return False
    if abs(pt1[1] -60) > 5:
    #if abs(pt1[0] -160) > 4 or abs(pt1[1] - 1850) > 4:
    #if abs(pt1[1] -800) > 4:
    #if abs(pt1[1] -1412) > 4:
        return True
    else:
        return False

def get_line_points(gray):
    lines = lsd(gray)
    #lines = detect_lsd(gray, 10)
    list_points = []
    lines = list(lines)
    print('lsd length: ', len(lines))
    lines.sort(key=lambda k :k[1])
    lines.sort(key=lambda k :k[0])
    #for i in range(len(lines)):
    #    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    #    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    for line in lines:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
    
        if condition(pt1,pt2):
            continue
        found = False
        for k, pts in enumerate(list_points):
            if (pt1,pt2) in pts :
                found = True
            elif contains(pts, (pt1,pt2)):
                list_points[k].append((pt1,pt2))
                found = True
        if found:
            continue
        list_points.append([(pt1, pt2)])
    return list_points


def mergeLines(list_points):
    for i, linePoints1 in enumerate(list_points):
        if len(linePoints1)==0:
            continue
        for j, linePoints2 in enumerate(list_points):
            if i>=j or len(linePoints2) ==0:
                continue
            line1 = Line(linePoints1, seg_tolerance)
            line2 = Line(linePoints2, seg_tolerance)
            linePoint1 = line1.getStartAndEndPointCoords()
            linePoint2 = line2.getStartAndEndPointCoords()
            if isCollinear(linePoint1, linePoint2, tolerance=line_tolerance):
                list_points[i].extend(linePoints2)
                list_points[j] = []

    return list_points

def draw_lines(list_points, src, box_tolerance):
    condSegments = []
    eventPoints = []

    k =0
    for pp in list_points:
        if len(pp) == 0:
            continue
        #for p in pp:
        #    cv2.line(src, p[0], p[1], (255,0,0),1)
        k += 1
    print('Before merge total lines: ', k)
    mergeLines(list_points)
    k =0
    for pp in list_points:
        if len(pp) == 0:
            continue
        k += 1
    print('After merge total lines: ', k)

    kk = 0
    for pts1 in list_points:
        if len(pts1) == 0:
            continue
        line = Line(pts1, line_tolerance)
        lineSegments =  line.getAllSegments(fill=False)
        for lineSegment in lineSegments:
            #print(lineSegment.points)
            point1, point2 = lineSegment.getStartAndEndPoints()
            if lineSegment.length > box_tolerance[1]:
                condSegments.append(lineSegment)
                lineSeg = LineSegment(linePoints=[point1, point2])
                if lineSeg.orientation == 0:
                    start = (point1.XCoord - box_tolerance[0], point1.YCoord)
                else:
                    start = (point1.XCoord, point1.YCoord)
                eventPoints.append(EventPoint(start, lineSeg, lineSeg.orientation))
                if lineSeg.orientation == 0:
                    end = (point2.XCoord + box_tolerance[0], point2.YCoord)
                    eventPoints.append(EventPoint(end, lineSeg, -1))
                pnt1, pnt2 = lineSegment.getStartAndEndPointCoords()
                #cv2.line(src, pnt1, pnt2, (0,0,255),1)
                kk += 1
    print('after segment: ', kk)
    return condSegments, eventPoints


def getIntersectionPoints(eventPoints, src, box_tolerance):
    line_set = set()
    #line = []
    eventPoints.sort(key=lambda x: x.point[0])
    for event in eventPoints:
        if event.direction == 0:
            line_set.add(event.lineSegment)
        elif event.direction == -1:
            line_set.remove(event.lineSegment)
        else:
            for seg in line_set:
                getIntersectionPointInfo(seg, event.lineSegment, box_tolerance)
                #if intersectionPoint != False:
                #    seg.getPoints().append(intersectionPoint)
                #    event.lineSegment.getPoints().append(intersectionPoint)


def getConnectedRects(lineSegments, w, h):
    blank_image = np.zeros((w, h, 1), np.uint8)
    #cv2.rectangle(blank_image, (120,120), (1140,1130), (255, 255, 255), 4)
    for line in lineSegments:
        points = line.getStartAndEndPointCoords()
        cv2.line(blank_image, points[0], points[1], (255,255,255), 1)

    ret, labels = cv2.connectedComponents(blank_image)
    print(np.max(labels))

    components = []
    for k in range(np.max(labels) +1):
        components.append([])

    for i in range(w):
        for j in range(h):
            if labels[i][j] == 0:
                continue
            components[labels[i][j]].append((j,i))

    rects = []
    for l in components:
        if len(l) < 50:
            continue
        l.sort(key=lambda x: x[0])
        x1 = l[0][0]
        x2 = l[-1][0]
        l.sort(key=lambda x: x[1])
        y1 = l[0][1]
        y2 = l[-1][1]
        length = getLineLength((x1,y1), (x2, y2))
        if length < 50:
            continue
        rects.append([(x1,y1), (x2, y2)])

    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0


    for rect in rects:
        cv2.rectangle(labeled_img, rect[0], rect[1], (255, 0, 0), 2)

    cv2.imwrite('labeled.png', labeled_img)


    return rects



def equal(l,keys, h):
    for key in keys:
        if abs(key -h) < l and abs(key +h) > l:
            return key

    return 0

def maxLengthIntersectedLine(lineSegments):
    maxSegment = lineSegments[0]
    for lineSegment in lineSegments[1:]:
        if lineSegment.length > maxSegment.length:
            maxSegment = lineSegment

    return maxSegment

def findLargestRectangle(lineSegments):
    largeRectCoords = []
    #lineSegments.points.sort(key=lambda x: x[0][1])    
    for i, lineSegment in enumerate(lineSegments):
        if lineSegment.orientation ==0:
            if len(lineSegment.getIntersectionLines()) == 0 or lineSegment.length < 100:
                continue
            maxIntersectSegment = maxLengthIntersectedLine(lineSegment.getIntersectionLines())
            if maxIntersectSegment.length < 100:
                continue
            startEndPoints = lineSegment.getStartAndEndPointCoords()
            intersectStartEndPoints = maxIntersectSegment.getStartAndEndPointCoords()
            x1 = min(startEndPoints[0][0], intersectStartEndPoints[0][0]) -10
            y1 = min(startEndPoints[0][1], intersectStartEndPoints[0][1]) -10
            x2 = max(startEndPoints[1][0], intersectStartEndPoints[1][0]) +10
            y2 = max(startEndPoints[1][1], intersectStartEndPoints[1][1]) +10
            rectCoord = [(x1,y1),(x2,y2)]

            largeRectCoords.append(rectCoord)
            return largeRectCoords


    return largeRectCoords

def isInside(points, rectCoords):
    for rectCoord in rectCoords:
        if len(points) == 2 and  ((points[0][0] >= rectCoord[0][0] and points[0][1] >= rectCoord[0][1]) and (points[1][0] <= rectCoord[1][0] and points[1][1] <= rectCoord[1][1])):
            return True
            
        
    return False

def isOutside(points, rectCoords):
    penalty = 14
    for rectCoord in rectCoords:
        if (points[1][0] -penalty > rectCoord[0][0] and  points[1][1] -penalty > rectCoord[0][1] and points[1][0]-penalty < rectCoord[1][0] and points[1][1] -penalty< rectCoord[1][1]) :
            return False
        if (points[0][0] +penalty > rectCoord[0][0] and points[0][1] +penalty> rectCoord[0][1] and points[0][0] +penalty < rectCoord[1][0] and points[0][1]+penalty < rectCoord[1][1]):
            return False


    return True

def isEqual(point1, point2):
    if abs(point1.XCoord - point2.XCoord) < tolerance2[0] and abs(point1.YCoord - point2.YCoord) < tolerance2[0]:
        return True
    else:
        return False

def nextPointLieOn(start, line):
    point2 = None
    for pt in line.getPoints():
        point = pt.getCoord()

        if isEqual(start, point) is True:
            #point2 = line.prevPoint(point)
            if point2 is not None and getLineLength(point2, point) > 10:
                #return [point, (point2.XCoord, point2.YCoord)]
                return point
            break
        point2 = point

    return None

def isLieOn(point, line):
    start, end = line.getStartAndEndPointCoords()
    if abs(point[0] - start[0]) < tolerance[0] and start[1] - point[1] <= tolerance2[0] and point[1] - end[1] <= tolerance2[0]:
        return True

    return False

def getNextIntersectionPoint(hLine, hPoint, vPoint):
    next_point = hLine.nextPoint(hPoint)
    while(next_point is not None):
        if next_point.intersection_type == 1 or next_point.intersection_type == 2:
            if next_point.XCoord - hPoint.XCoord > 50:
                return next_point
        next_point = hLine.nextPoint()

    return (hLine.getStartAndEndPoints())[1]

def getRightVLinePoint(vLine, vPoint):
    prev_point = vLine.prevPoint(vPoint)
    while(prev_point is not None):
        if prev_point.intersection_type == 1 or prev_point.intersection_type == 4:
            return prev_point
        prev_point = vLine.prevPoint()

    return (vLine.getStartAndEndPoints())[0]

def getSmallestVariableRectangles(eventPoints, src, rect, box_tolerance, box_max_size):

    eventPoints.sort(key=lambda x: x.point[1])
    rectangles = {}
    for event in eventPoints:
        hL = event.lineSegment
        if event.direction == 1 or event.direction == -1:
            continue
        for point in hL.getPoints():
            if point.intersection_type != 0:
                #cv2.circle(src, (point.XCoord, point.YCoord), 10, (0, 0, 255), 1)
                #cv2.putText(src, (str(point.XCoord) + ',' +str(point.YCoord)), (point.XCoord, point.YCoord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                #cv2.putText(src, (str(point.intersection_type)), (point.XCoord, point.YCoord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                pass

        hL.sort()
        #hL.merge_nearest_points(10)
        hLine = hL.getPoints()
        #if len(hLine) <= 2 and isInside(hLine, connectedRects) is False:
        # continue
        eventPoints2 = eventPoints[:]
        eventPoints2.sort(key=lambda x: x.point[0])
        #rect = (hLine[0].getCoord(), hLine[-1].getCoord())
        for i, hPoint in enumerate(hLine[:-1]):
            # if hLine[i+1].XCoord - hPoint.XCoord < 10:
            #     continue
            for event2 in eventPoints2:
                if event2.direction == 0 or event2.direction == -1:
                    continue
                vL = event2.lineSegment
                vL.sort()
                #vL.merge_nearest_points(10)
                vLine = vL.getPoints()
                for j, vPoint in enumerate(vLine[:-1]):
                    if vLine[j].intersection_type != 0 and isEqual(hLine[i], vLine[j+1]):
                        point = getNextIntersectionPoint(hL, hLine[i], vLine[j])
                        #point1 = linePoint.getCoord()
                        w = point.XCoord - hLine[i].XCoord
                        h = vLine[j+1].YCoord - vLine[j].YCoord
                        if h < box_tolerance[1] or w < box_tolerance[1]:
                            continue
                        #h2 = point2[1] - point1[1]
                        #h = max(h1, h2)
                        #rectangles[rect].append([vLine[j].XCoord, vLine[j].YCoord, w, h])
                        boxes = (vLine[j].getCoord(), point.getCoord())
                        if rect in rectangles:
                            rectangles[rect].append(boxes)
                        else:
                            rectangles[rect] = [boxes]
                        break

    return rectangles


def getVariableBoxes(lineSeg1, lineSeg2, box_min_size, box_max_size):
    rects = []
    points1 = lineSeg1.getPoints()
    points2 = lineSeg2.getPoints()
    if len(points1) <=3:
        return []
    box_dx = points1[2].XCoord - points1[1].XCoord
    box_dy = points2[1].YCoord - points1[1].YCoord
    top_left_point = points1[1].getCoord()
    for i, point in enumerate(points1[2:]):
        #if (point.intersection_type == 1 or point.intersection_type == 3) and (point.XCoord - top_left_point[0]) < box_dx -5 :
        #    continue
        if point.intersection_type == 1 or point.intersection_type == 3:
            #box = (points1[i-1].getCoord(), points2[i].getCoord())
            box = (top_left_point, (point.XCoord, point.YCoord + box_dy))
            rects.append(box)
            top_left_point = point.getCoord()

    return rects



def getBoxes(lineSeg1, lineSeg2, box_min_size, box_max_size):
    rects = []
    points1 = lineSeg1.getPoints()
    points2 = lineSeg2.getPoints()
    if (points1[-1].XCoord - points1[0].XCoord) < box_max_size[0] and abs(points1[-1].XCoord - points1[0].XCoord - points2[-1].XCoord + points2[0].XCoord) < 10:
        rects.append((points1[0].getCoord(), points2[-1].getCoord()))
        return rects


    if box_max_size[0] < points1[1].XCoord - points1[0].XCoord or box_min_size[0] > points1[1].XCoord - points1[0].XCoord:
        box_dx = box_min_size[0] 
    else:
        box_dx = points1[1].XCoord - points1[0].XCoord
    box_dy = points2[0].YCoord - points1[0].YCoord
    #print(lineSeg1.getStartAndEndPointCoords())
    top_left_point = points1[0].getCoord()
    for i, point in enumerate(points1[1:]):
        #print('point8888', point.intersection_type, box_dx, point.XCoord - top_left_point[0])
        if (point.intersection_type == 1 or point.intersection_type == 3) and (point.XCoord - top_left_point[0]) < box_dx -10 :
            continue
        if point.intersection_type == 1 or point.intersection_type == 3:
            if point.XCoord - top_left_point[0] > 400:
                top_left_point = point.getCoord()
                rects = []
                continue
            if point.XCoord - top_left_point[0] > box_max_size[0]:
                split = int((point.XCoord - top_left_point[0] + box_dx/2)/box_dx)
                new_y1 = top_left_point[1]
                for l in range(split):
                    new_x1 = top_left_point[0] + l * box_dx
                    new_x2 = top_left_point[0] + (l+1)* box_dx
                    #box1 = ((new_x1, new_y1), (new_x2, points2[0].YCoord))
                    box1 = ((new_x1, new_y1), (new_x2, new_y1 + box_dy))
                    rects.append(box1)
            else:
                #box = (top_left_point, (point.XCoord, points2[0].YCoord))
                box = (top_left_point, (point.XCoord, top_left_point[1] + box_dy))
                box_dx = (point.XCoord - top_left_point[0])
                #box_dy = points2[i].YCoord - point.YCoord
                rects.append(box)
            top_left_point = point.getCoord()

    return rects



def getSmallestRectangles(eventPoints, src, rect, box_min_size=box_size_min, box_max_size=box_size_max):

    eventPoints.sort(key=lambda x : x.point[1])
    rectangles = {}
    for i, event in enumerate(eventPoints):
        hL = event.lineSegment
        if event.direction == 1 or event.direction == -1:
            continue
        for point in hL.getPoints():
            if point.intersection_type != -1:
                #cv2.circle(src, (point.XCoord, point.YCoord), 10, (0, 0, 255), 1)
                #cv2.putText(src, str( point.XCoord), (point.XCoord, point.YCoord), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)
                #cv2.putText(src, (str(point.intersection_type)), (point.XCoord, point.YCoord+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                pass

        hL.sort()
        for event2 in eventPoints[i+1:]:
            hL2 = event2.lineSegment
            if event2.direction == 1 or event2.direction == -1:
                continue
            if angle_between((hL.getStartAndEndPoints()[0].getCoord(), hL.getStartAndEndPoints()[1].getCoord()),(hL2.getStartAndEndPoints()[0].getCoord(),hL2.getStartAndEndPoints()[1].getCoord())) > 30.0 and angle_between((hL.getStartAndEndPoints()[0].getCoord(), hL.getStartAndEndPoints()[1].getCoord()),(hL2.getStartAndEndPoints()[0].getCoord(),hL2.getStartAndEndPoints()[1].getCoord())) < 160.0:
                continue
            if abs(hL.getStartAndEndPoints()[0].YCoord - hL2.getStartAndEndPoints()[0].YCoord) < box_min_size[1]:
                continue
            if abs(hL.getStartAndEndPoints()[0].YCoord - hL2.getStartAndEndPoints()[0].YCoord) > box_max_size[1]:
                break
            hL2.sort()
            if ((abs(hL.getStartAndEndPoints()[0].XCoord -  hL2.getStartAndEndPoints()[0].XCoord) < box_min_size[0]) or (hL.getStartAndEndPoints()[0].XCoord < hL2.getStartAndEndPoints()[0].XCoord and hL.getStartAndEndPoints()[1].XCoord > hL2.getStartAndEndPoints()[0].XCoord - box_min_size[0]) or (hL.getStartAndEndPoints()[0].XCoord > hL2.getStartAndEndPoints()[0].XCoord and hL.getStartAndEndPoints()[1].XCoord < hL2.getStartAndEndPoints()[0].XCoord + box_min_size[0]) ) and abs(hL.getStartAndEndPoints()[0].YCoord - hL2.getStartAndEndPoints()[0].YCoord) < box_max_size[1]:
            #if abs(hL.getStartAndEndPoints()[0].XCoord -  hL2.getStartAndEndPoints()[0].XCoord) < box_max_size[0] and abs(hL.getStartAndEndPoints()[1].XCoord -  hL2.getStartAndEndPoints()[1].XCoord) < box_max_size[0]:
                boxes = getBoxes(hL, hL2, box_min_size, box_max_size)
                print('XCoord :', hL.getStartAndEndPoints()[0].XCoord)
                if len(boxes) > 0:
                    print('angle: ' , angle_between((hL.getStartAndEndPoints()[0].getCoord(), hL.getStartAndEndPoints()[1].getCoord()),(hL2.getStartAndEndPoints()[0].getCoord(),hL2.getStartAndEndPoints()[1].getCoord())))
                    #rect = ((hL.getStartAndEndPoints()[0].XCoord, hL.getStartAndEndPoints()[0].YCoord), (hL2.getStartAndEndPoints()[0].XCoord, hL2.getStartAndEndPoints()[0].YCoord))
                    #print('intersection@@@@', hL.getStartAndEndPoints()[0].getCoord(), hL2.getStartAndEndPoints()[0].getCoord(), hL.getStartAndEndPoints()[1].getCoord(), hL2.getStartAndEndPoints()[1].getCoord())
                    if rect in rectangles:
                        rectangles[rect].extend(boxes)
                    else:
                        rectangles[rect] = boxes
                    break

    return rectangles

def getCorner(eventPoints, src, rect, box_min_size, box_max_size):
    eventPoints.sort(key=lambda x : x.point[1])
    eventPoints.sort(key=lambda x : x.point[0])
    rectangles = {}
    rectangles[rect] = []
    for i, event in enumerate(eventPoints):
        hL = event.lineSegment
        if event.direction == 1 or event.direction == -1:
            continue
        for point in hL.getPoints():
            if point.intersection_type != -1:
                #cv2.circle(src, (point.XCoord, point.YCoord), 10, (0, 0, 255), 1)
                #cv2.putText(src, str( point.XCoord), (point.XCoord, point.YCoord), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA)
                #cv2.putText(src, (str(point.intersection_type)), (point.XCoord, point.YCoord+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                pass

        hL.sort()
        for event2 in eventPoints:
            hL2 = event2.lineSegment
            if event2.direction == 0: # or event2.direction == -1:
                continue
            if angle_between((hL.getStartAndEndPoints()[0].getCoord(), hL.getStartAndEndPoints()[1].getCoord()),(hL2.getStartAndEndPoints()[0].getCoord(),hL2.getStartAndEndPoints()[1].getCoord())) > 30.0 and angle_between((hL.getStartAndEndPoints()[0].getCoord(), hL.getStartAndEndPoints()[1].getCoord()),(hL2.getStartAndEndPoints()[0].getCoord(),hL2.getStartAndEndPoints()[1].getCoord())) < 165.0:
                continue
            if (hL2.getStartAndEndPointCoords()[1][0] - hL.getStartAndEndPointCoords()[0][0]) > box_min_size[0] and (hL2.getStartAndEndPointCoords()[1][0] - hL.getStartAndEndPointCoords()[0][0]) < box_max_size[0] and hL2.getStartAndEndPointCoords()[1][1] - hL.getStartAndEndPointCoords()[0][1] > box_min_size[1] and hL2.getStartAndEndPointCoords()[1][1] - hL.getStartAndEndPointCoords()[0][1] < box_max_size[1]:
                #if (hL.getStartAndEndPoints()[0].getIntersectionType == 3 or hL.getStartAndEndPoints()[0].getIntersectionType() == 1 or hL.getStartAndEndPoints()[1].getIntersectionType() == 3 or hL.getStartAndEndPoints()[1].getIntersectionType() == 1) and (hL2.getStartAndEndPoints()[0].getIntersectionType() == 2 or hL2.getStartAndEndPoints()[0].getIntersectionType() == 1 or hL2.getStartAndEndPoints()[1].getIntersectionType() == 2 or hL2.getStartAndEndPoints()[1].getIntersectionType() == 1 ):
                if hL.getStartAndEndPoints()[0].getIntersectionType != 2 and hL2.getStartAndEndPoints()[1].getIntersectionType != 3: 

                    box = (hL.getStartAndEndPointCoords()[0], hL2.getStartAndEndPointCoords()[1])
                    rectangles[rect].append(box)
                    break

    return rectangles
            


def removeBorder(img, w, h):
    for j in range(10):
        for i in range(0,w-8):
            if img[i][j] == 0 and img[i+1][j]  == 0 and img[i+2][j] ==0 and img[i+3][j] == 0 and img[i+4][j] == 0:
                img[i][j] = 255
        img[i+1][j] = 255
        img[i+2][j] = 255
        img[i+3][j] = 255
        img[i+4][j] = 255
    for j in [h-1,h-2,h-3,h-4,h-5,h-6,h-7,h-8,h-9,h-10]:
        for i in range(0,w-10):
            if img[i][j] == 0 and img[i+1][j]  == 0 and img[i+2][j] ==0 and img[i+3][j] == 0 and img[i+4][j]==0:
                img[i][j] = 255
        img[i+1][j] = 255
        img[i+2][j] = 255
        img[i+3][j] = 255
        img[i+4][j] = 255

    test1 = False
    for i in range(10):
        for j in range(0,h-8):
            if img[i][j] == 0 and img[i][j+1]  == 0 and img[i][j+2] ==0 and img[i][j+3] == 0 and img[i][j+4]==0:
                img[i][j] = 255
                test1 = True
        #if test1 is not True:
        #    break
        img[i][j+1] = 255
        img[i][j+2] = 255
        img[i][j+3] = 255
        img[i][j+4] = 255
    test = False
    for i in [w-1, w-2,w-3,w-4, w-5,w-6, w-7,w-8,w-9]:
        for j in range(0,h-5):
            if img[i][j] == 0 and img[i][j+1]  == 0 and img[i][j+2] ==0 and img[i][j+3] == 0 and img[i][j+4]==0:
                img[i][j] = 255
                test = True
        #if test is not True:
        #    break
        img[i][j+1] = 255
        img[i][j+2] = 255
        img[i][j+3] = 255
        img[i][j+4] = 255
            



def preprocess_image(imgfile, preprocess_file):
    
    if deskew_enabled() is True:
        Deskew(imgfile, None , preprocess_file, 0.0).deskew()
        src = cv2.imread(preprocess_file, cv2.IMREAD_COLOR)
    else:
        src = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #gray = cv2.fastNlMeansDenoising(gray, 10)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    gray = cv2.bitwise_not(gray)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)
    gray = cv2.bitwise_not(gray)

    h, w,_ = src.shape
    print(h, w)
    cv2.imwrite(preprocess_file, gray)
    
    return src, preprocess_file, gray

def box_extraction(horizontal_roi,vertical_roi, src, box_tolerance, box_min_size, box_max_size, rect, variable=False, with_line=False, corner=False):
    t1 = time.time()
    gray = cv2.GaussianBlur(horizontal_roi, (3,3), 0)
    horizontal_roi = cv2.bitwise_not(gray)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    img = cv2.morphologyEx(horizontal_roi, cv2.MORPH_CLOSE, element)
    horizontal_roi = cv2.bitwise_not(horizontal_roi)
    list_points = get_line_points(horizontal_roi)
    if with_line is False:
        gray = cv2.GaussianBlur(vertical_roi, (3,3), 0)
        vertical_roi = cv2.bitwise_not(gray)
        vertical_roi = cv2.morphologyEx(vertical_roi, cv2.MORPH_CLOSE, element)
        vertical_roi = cv2.bitwise_not(vertical_roi)
        list_points2 = get_line_points(vertical_roi)
        list_points.extend(list_points2)
    t2 = time.time()
    if expired_date - time.time() < 0:
        return []
    print('time taken by get_line_points: %s' % (t2 -t1))
    lineSegments, eventPoints = draw_lines(list_points,src, box_tolerance)
    t3 = time.time()
    print('time taken by draw_lines: %s' % (t3 -t2))
    #lineSegmentPoints = getIntersectionPoints2lineSegments, src)
    getIntersectionPoints(eventPoints, src, box_tolerance)
    t4 = time.time()
    print('time taken by getIntersectionPoints: %s' % (t4 -t3))
    
    if corner is True:
        cv2.imwrite('temp.png', horizontal_roi)
        rectangles = getCorner(eventPoints, src, rect, box_min_size, box_max_size)
    elif variable is  False:
        rectangles = getSmallestRectangles(eventPoints, src, rect, box_min_size, box_max_size)
    else:
        rectangles = getSmallestVariableRectangles(eventPoints, src, rect, box_tolerance, box_max_size)
    #print(rectangles.keys())
    t6 = time.time()
    print('time taken by getSmallestRectangles: %s' % (t6 -t4))
    print('time taken by box_extraction: %s' % (t6 -t1))

    return rectangles


def box_identification(position, coords, box_min_size):
    rectangles = {}
    rectangles[position] = []
    for coord in coords:
        for c in coord:
            c_u = [c[0] - position[0], c[1] - position[1]]
            if c_u[0] < 0:
                c_u[0] = 0
            if c_u[1] < 0:
                c_u[1] = 0
            rect = (tuple(c_u), (c_u[0]+box_min_size[0] +10, c_u[1]+box_min_size[1]+10))
            rectangles[position].append(rect)


    return rectangles


j = 2000
def get_value(imgfile, text_rect, rectangles, tmpsrc, src, model_types, pos=(0,0)):
    tesseract_config = '"%s" stdout -l "eng"  --oem 1 --psm 7 2>/dev/null'
    #command = 'tesseract '
    command = get_tesseract_command()
    #command = 'export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH && /home/anup/Downloads/tesseract-4.0/bin/tesseract --tessdata-dir /home/anup/Downloads/tesseract-4.0/ '
    text = ''
    height,width = tmpsrc.shape
    #print('rect: %s'% text_rect[0][1])
    box_coords = []
    for i, box_rectangles in enumerate(rectangles):
        #x = rect[0][0] 
        #y = rect[0][1] 
        #print(y, text_rect[1] - y)
        #print('matches at %s, %s' % (x,y))
        #box_rectangles = rectangles[rect]
        #if len(box_rectangles) < min_boxes:
        #    break
        if len(model_types) >i:
            model_type = model_types[i]
        else:
            model_type = 'combines'
        g=0
        char_images = []
        char_texts = []
        char_coords = []
        t_coords = []
        for box_rect in box_rectangles:
            #print(box_rect)
            x = box_rect[0][0] -2 
            y = box_rect[0][1] -2
            w = box_rect[1][0] - x +8
            h = box_rect[1][1] - y +8
            x1, y1 = box_rect[0]
            x2, y2 = box_rect[1]
            if h < 30 or w < 30:
                continue
            if h > 100 or w > 2000:
                continue
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if y+h > height:
                h = height - y
            if x+w > width:
                w = width -x
            roi = tmpsrc[y:y+h, x:x+w]
            #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #cv2.rectangle(src, (x1+2,y1+2),(x2-2,y2-2), (200,0,0), 1)


            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
            global j
            uznfile = imgfile[:-3] + 'uzn'
            uzn = open(uznfile, 'w')
            uzn.write('%s %s %s %s' %(x, y, w, h))
            uzn.write(' form_block_' +str(j ))
            uzn.close()
            j += 1
            roi_file = 'roi_'+str(j)+'.png'
            #cv2.imwrite(roi_file, roi)
            #single_image = cv2.imread(roi_file,cv2.COLOR_BGR2GRAY)
            if h <= 100 and w > 100:
                if len(char_images) > 0:
                    stdoutdatas,prob,prob2 = getPredictionResult(char_images, model_type)
                    text = text + ' "'
                    last_x2 = char_coords[0][0][0]
                    for data,coord in zip(stdoutdatas,char_coords):
                        if coord[0][0] - last_x2 > 10:
                            text = text + ';'
                        text = text + data 
                        last_x2 = coord[1][0]
                        if is_print_on_page_enable() is True and data != ' ':
                            cv2.putText(src, data.strip(), (coord[0][0]+pos[0]+5,coord[0][1]+pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,200), 2, cv2.LINE_AA)
                    text = text + '" '
                    char_coords = []
                    char_images = []
                cv2.imwrite(roi_file, roi)
                tesseract_config = '"%s" stdout -l "eng"  --oem 1 --psm 7 2>/dev/null'
                keyText = subprocess.check_output(command + tesseract_config % roi_file, universal_newlines=True, stderr=subprocess.STDOUT, shell=True)
                print(keyText)
                keyText = keyText.split('\n')[0].strip()
                text = text + '   ,,,' +keyText + '   '
                continue

            char_coords.append(box_rect)
            char_images.append(roi)
            t_coords.append([box_rect[0][0]+pos[0],box_rect[0][1]+pos[1]])
            last_x2 = x2
        if len(char_images) > 0:
            stdoutdatas,prob,prob2 = getPredictionResult(char_images, model_type)
            #stdoutdata,prob,prob2 = subprocess.check_output(command + tesseract_config % roi_file, universal_newlines=True, stderr=subprocess.STDOUT, shell=True),1,1
            text = text + ' "'
            last_x2 = char_coords[0][0][0]
            for data,coord in zip(stdoutdatas,char_coords):
                if coord[0][0] - last_x2 > 10:
                    text = text +';'
                text = text + data 
                last_x2 = coord[1][0]
                if is_print_on_page_enable() is True and data != ' ':
                    cv2.putText(src, data.strip(), (coord[0][0]+pos[0] +5,coord[0][1]+pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,200), 2, cv2.LINE_AA)
            text = text + '"'
            print(text,stdoutdatas)
        box_coords.append(t_coords)
        text = text + '\n'
    return text.strip(), box_coords



def write_xml(xml, rect, block_count, key, value):
    txt = '<div class="text_area" id="block_%s" coord="box %s" type="field">'
    coord = ' '.join(map(str, rect))
    block_count += 1
    xml.write(txt % (block_count, coord))
    sub_block_count = 1
    txt = '<span class="form_text" id="block_%s_%s" ocr_value="%s" nlp_value="" score="" type="key"> <![CDATA[  %s ]]> </span>\n'
    ocr_text = key.strip()
    xml.write(txt % (block_count, sub_block_count, ocr_text, ocr_text))
    txt = '<span class="form_text" id="block_%s_%s" ocr_value="%s" nlp_value="" score="" type="value"> <![CDATA[  %s ]]>  </span>\n'
    ocr_text = value.strip()
    sub_block_count = 2
    xml.write(txt % (block_count, sub_block_count, ocr_text, ocr_text))
    xml.write('</div>\n')


def process_field(field, pos):
    return parameters_of_matched_field(field, pos)


def tesseract_process_new(preprocessfile, gray, src, page_no):
    height, width,_ =src.shape
    command = get_tesseract_command()
    #command = 'export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH && /home/anup/Downloads/tesseract-4.0/bin/tesseract --tessdata-dir /home/anup/Downloads/tesseract-4.0/ '
    filename, extension = os.path.splitext(preprocessfile)
    tesseract_config_hocr = ' %s %s --oem 1 configfile'
    print(command +  tesseract_config_hocr % (preprocessfile, filename))
    subprocess.check_output(command +  tesseract_config_hocr % (preprocessfile, filename), shell=True)
    hocrfile = filename + '.hocr'
    
    areaCoords = getLayoutInfo(hocrfile)
    xml = open(filename + '.xml', 'w')
    jsonfile = open(filename + '.json', 'w')
    block_count = 0
    header = '<?xml version="1.0" encoding="UTF-8"?> \n  \
             <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" \n  \
        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"> \n \
    <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en"> \n \
     <head> \n \
      <coord></coord> \n \
     </head> \n \
     <body> \n \
    '
    xml.write(header)
    txt = '<div class="page_layout" id="image" title="%s" coord="box 0 0 %s %s">\n'
    xml.write(txt % (preprocessfile, width, height))
    file_layout = {}
    file_layout['layout'] = []
    data = {}
    coords = [0, 0, width, height]
    file_layout['layout'].append({'filename' : preprocessfile, 'coord' : coords, 'data' : data})


    config_file = 'icici_bank_fields-page' + str(page_no)+ '.csv'
    read_csv(config_file)
    
    idx = 0
    for obj in areaCoords:
        areaCoord = obj.coord
        if abs(areaCoord[0][0]- areaCoord[1][0]) > width/2 and abs(areaCoord[0][1]- areaCoord[1][1]) > height/2:
            continue
        if abs(areaCoord[0][1]- areaCoord[1][1]) > 500:
            continue
        rect = [areaCoord[0][0], areaCoord[0][1], areaCoord[1][0] - areaCoord[0][0], areaCoord[1][1] - areaCoord[0][1]]
        x1 = areaCoord[0][0]
        y1 = areaCoord[0][1]
        x2 = areaCoord[1][0]
        y2 = areaCoord[1][1]
        stdoutdata = obj.text
        #cv2.putText(src, obj.text, obj.coord[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        #print('text: %s' %stdoutdata.encode('utf-8'))
        print('text: %s' %stdoutdata, x1,y1,x2,y2)
        stdoutdata = stdoutdata.strip()
        param_values = process_field(stdoutdata, (x1,y1))
        if param_values is not None:
            keyText = param_values[0]
            #if 'Company Name' != param_values[0]:
            #    continue
            box_max_size = eval(param_values[3])
            box_min_size = eval(param_values[4])
            box_tolerance = eval(param_values[5])
            x = eval(param_values[6])
            y = eval(param_values[7])
            h = int(eval(param_values[8]))
            w = int(eval(param_values[9]))
            post_process_func = param_values[19]
            if param_values[10]  == 'variable':
                box_variable_extraction = True
            else:
                box_variable_extraction = False
            with_line = eval(param_values[11])
            model_type = param_values[12]
            no_lines = 8
            if eval(param_values[13]) is True:
                side_h = int(eval(param_values[14]))
                side_w = int(eval(param_values[15]))
                side_text = eval(param_values[16])
                side_searchable_text = eval(param_values[17])

            print('param_values ', keyText, box_min_size, box_max_size, box_tolerance, h, w, box_variable_extraction,with_line) 

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if y+h > height:
                h = height - y
            if x+w > width:
                w = width -x

            print(x,y, x+h, y+w)
            roi = gray[y:y+h, x:x+w]
            print(roi.shape)

            cv2.imwrite('temp_' + str(idx) + '.png', roi)
            #cv2.imwrite('temp.png', roi)
            idx += 1

            roi_only_lines_h, roi_only_lines_v, roi_without_lines, th_h = get_lines_and_without_lines_image(roi, box_min_size)
            if with_line is True:
                roi_only_lines_h = roi
                roi_only_lines_v = roi
            rectangles = box_extraction(roi_only_lines_h, roi_only_lines_v, src, box_tolerance, box_min_size, box_max_size, ((x,y),(x+w,y+h)), variable=box_variable_extraction, with_line=with_line)
            rectangles_t = {}
            all_rectangles = []
            for key in rectangles.keys():
                boxes = rectangles[key]
                boxes.sort(key=lambda tup: tup[0][1])
                rects =  []
                rects.append(boxes[0])
                for box in boxes[1:]:
                    if abs(box[0][1] - rects[0][0][1]) < box_min_size[1]:
                        rects.append(box)
                    else:
                        rects.sort(key=lambda tup: tup[0][0])
                        all_rectangles.append(rects)
                        rects = []
                        rects.append(box)
                rects.sort(key=lambda tup: tup[0][0])
                all_rectangles.append(rects)


                if len(all_rectangles) != no_lines:
                    print("provided line no and calculated line number is not same")

                #rectangles[key].sort(key=lambda tup: tup[0][1])
            if model_type == 'checks':
                for i, rects in enumerate(all_rectangles):
                    values = []
                    for val in rects:
                        values.append(((val[1][0],val[0][1]),(val[1][0]+side_w, val[1][1])))
                        values.append(val)
                    all_rectangles[i] = values


            if is_box_mark() is True:
                for rects in all_rectangles:
                    #key_t = ((key[0][0] +x, key[0][1] + y), (key[1][0] +x, key[1][1] + y))
                    #values_t = [ ((val[0][0] +x, val[0][1] +y), (val[1][0] +x, val[1][1] +y)) for val in rectangles[key]]
                    values_t = [ ((val[0][0] +x, val[0][1] +y), (val[1][0] +x, val[1][1] +y)) for val in rects]
                    #rectangles_t[key_t] = values_t
                    for box_rect in values_t:
                        xx1, yy1 = box_rect[0]
                        xx2, yy2 = box_rect[1]
                        cv2.rectangle(src, (xx1+2,yy1+2),(xx2-2,yy2-2), (200,0,0), 1)
    
            value, coords = get_value(preprocessfile, None, all_rectangles, roi_without_lines, src, model_type, pos=(x,y))
            print(value)
            if post_process_func != '':
                print(post_process_func)
                vals = eval(post_process_func)
                box_coord = [rect[0], rect[1], x+w, y+h]
                for val in vals:
                    write_xml(xml, box_coord, block_count, val, vals[val])
                continue
    
            value = value.replace(';' , ' ')
            txt = '<div class="text_area" id="block_%s" coord="box %s" type="field">\n'
            box_coord = [rect[0], rect[1], x+w, y+h]
            coord = ' '.join(map(str, box_coord))
            block_count += 1
            xml.write(txt % (block_count, coord))
            data[keyText] = {'coord': box_coord,  'value' : value.strip()}

            sub_block_count = 1
            txt = '<span class="form_text" id="block_%s_%s" coord="box %s" ocr_value="%s" nlp_value="" score="" type="key"><![CDATA[  %s ]]>   </span>\n'
            coord = ' '.join(map(str, rect))
            ocr_text = keyText.strip()
            xml.write(txt % (block_count, sub_block_count, coord, '', ocr_text))
            txt = '<span class="form_text" id="block_%s_%s" coord="box %s" ocr_value="%s" nlp_value="" score="" type="value"><![CDATA[  %s ]]> </span>\n'
            box_coord = [rect[2], rect[1], x+w, y+h]
            coord = ' '.join(map(str, box_coord))
            ocr_text = value.strip()
            sub_block_count = 2
            xml.write(txt % (block_count, sub_block_count, coord, '', ocr_text))
            xml.write('</div>\n')
            
    json.dump(file_layout, jsonfile)
    jsonfile.close()
    xml.write('</div>\n')
    xml.write('</body>\n')
    xml.write('</html>\n')
    xml.close()


def get_y_differance(gray, img_length,  gaps_info):
    roi = gray[0:0+img_length,0:]
    #roi = cv2.equalizeHist(roi)
    cv2.imwrite('gap_check.png', roi)
    command = get_tesseract_command()
    tesseract_config_hocr = ' %s %s --oem 1 configfile'
    subprocess.check_output(command +  tesseract_config_hocr % ('gap_check.png', 'gap_check'), shell=True)
    hocrfile = 'gap_check.hocr'
    
    areaCoords = getLayoutInfo(hocrfile)
    pos = gaps_info[0]
    strings = gaps_info[1]
    for obj in areaCoords:
        for i, string in enumerate(strings):
            if string in obj.text.lower():
                areaCoord = obj.coord
                y_coord = (areaCoord[0][1] + areaCoord[1][1])/2
                x_coord = areaCoord[0][0] 
                return (x_coord - pos[i][0], int(y_coord) - pos[i][1])

    return (0,0)

def pdf2image(pdffile, imgfile='/tmp/tmp.png',outfolder='Result'):
    t1 = time.time()
    image_pdf = Image(filename=pdffile, resolution=400) #take filename
    print("time taken for pdf to image conversion : %s " %(time.time() -t1))
    head, tail = os.path.split(pdffile)
    image_page = image_pdf.convert("png") #png conversion
    page = 1 #init page

    files = ' '
    recognized_text = []
    for img in image_page.sequence: # Every single image in image_page for grayscale conversion in 300 resolution

        img_per_page = Image(image=img)
        #img_per_page.type = 'grayscale'
        #img_per_page.depth = 8
        img_per_page.density = 400
        #img_per_page.threshold = '50%'

        #try:
        #    img_per_page.level(black=0.3, white=1.0, gamma=1.5, channel=None)

        #except AttributeError as e:
        #    print("Update Wand library: %s" % e)

        if page not in get_page_nos():
            page += 1
            continue
        img_per_page.save(filename=imgfile)
        t6 = time.time()
        #config_file = 'icici_bank_fields-page' + str(page)+ '.csv'
        #read_csv(config_file)
        preprocess_file = os.path.join(outfolder, os.path.splitext(os.path.split(pdffile)[1])[0] + '-' + str(page) + '.png')
        src, preprocess_file, gray = preprocess_image(imgfile, preprocess_file)
        #_,roi_only_lines, roi_without_lines = get_lines_and_without_lines_image(gray, [40,40])
        #cv2.imwrite(preprocess_file, roi_without_lines)
        tesseract_process(preprocess_file, gray, src,page)
        t7 = time.time()
        print('time taken by tesseract_process: %s for page %s' % ((t7 -t6), page))
        cv2.imwrite(preprocess_file, src)
        files += preprocess_file + ' '
        page += 1
    print('convert' + files + './Result/output_' + tail)
    outpdf = os.path.join(outfolder, 'output_' + tail)
    stdoutdata = subprocess.check_output('convert' + files + outpdf, universal_newlines=True,
                        stderr=subprocess.STDOUT, shell=True)
    return outpdf 



def read_configuration():
    application_config = 'config'
    read_application_config(application_config)
    global seg_tolerance
    global line_tolerance
    seg_tolerance = get_segment_tolerance()
    line_tolerance = get_line_tolerance()


def process_bank_form(imgfile, page):
    src, preprocess_file, gray = preprocess_image(imgfile, '/tmp/temp1.png')
    #_,roi_only_lines, roi_without_lines = get_lines_and_without_lines_image(gray, [40,40])
    #imgfile= 'without_lines1.png'
    #cv2.imwrite(imgfile, roi_without_lines)
    tesseract_process(imgfile, gray, src, page)
    cv2.imwrite(os.path.join('test_'+os.path.splitext(os.path.split(imgfile)[1])[0] +'.png'), src)
    return 'test_'+os.path.splitext(os.path.split(imgfile)[1])[0] +'.png'



