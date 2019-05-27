#!/usr/bin/env python

import os
import numpy as np
import cv2
import sys, getopt
import math
import yaml
import codecs

#Read points from the landmark file
def readPointsFromLandMark(file):
    fs = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
    fn = fs.getNode("Point2f")
    print(fn);
    img_prop = fs.getNode("ImageSize")
    cols = img_prop.at(0).real()
    rows = img_prop.at(1).real()
    m = fn.mat()
    points = []
    for i in range(m.shape[1]):
        points.append((int(m[0,i,0]), int(m[0,i,1])))
    points.append((0, 0))
    points.append((int((cols-1)/2), 0))
    points.append((int(cols-1), 0))
    points.append((int(cols-1), int((rows-1)/2)))
    points.append((int(cols-1), int(rows-1)))
    points.append((int((cols-1)/2), int(rows-1)))
    points.append((0, int(rows-1)))
    points.append((0, int((rows-1)/2)))
    return points

# Read points from text file
def readPoints(path) :
    points = [];
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    return points

# Apply affine transform calculated using srcTri and dstTri to src
def applyAffineTransform(src, srcTri, dstTri, size) :

    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    mask = np.zeros((r[3], r[2], img1.shape[2]), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    #imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    imgRect = (1.0 - 0) * warpImage1 + 0 * warpImage2
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

    # Create mask to remove redundant area
    mask1 = np.zeros_like(warpImage1)
    g_warpImage1 = cv2.cvtColor(warpImage1, cv2.COLOR_BGR2GRAY)
    g_warpImage2 = cv2.cvtColor(warpImage2, cv2.COLOR_BGR2GRAY)
    for i in range(0, 3) :
        mask1[:, :, i] = (g_warpImage1[:, :] >0) & (g_warpImage2[:, :] >0)
        mask1[:, :, i] = np.where(mask1[:, :, i], 1, 0)
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * mask1

def morphImage(filename1, filename2, alpha) :
    img1 = cv2.imread(filename1+"/head3d.jpg");
    img2 = cv2.imread(filename2+"/head3d.jpg");

    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # points1 = readPoints(filename1 + '.txt')
    # points2 = readPoints(filename2 + '.txt')
    print(filename1+'/facelandmarks.yml');
    points1 = readPointsFromLandMark(filename1+'/facelandmarks.yml')
    points2 = readPointsFromLandMark(filename2+'/facelandmarks.yml')

    points = [];

    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))


    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

    # Read triangles from tri.txt
    with open("tri.txt") as file :
        for line in file :
            x,y,z = line.split()

            x = int(x)-1
            y = int(y)-1
            z = int(z)-1

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]

            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    cv2.imwrite("result_"+str(alpha)+".jpg", np.uint8(imgMorph))

if __name__ == '__main__' :

    argv = sys.argv[1:]
    #inputfile = 'houi_1.jpg'
    inputfile = 'sasi.jpg'
    outputfile = 'houi_1.jpg'
    # with open('Alan1/facelandmarks.yml', 'r') as f:
    #     read_data = yaml.load(f)
    #     print (read_data["Point2f"])
    # fs = cv2.FileStorage("Alan1/facelandmarks.yml", cv2.FILE_STORAGE_READ)
    # fn = fs.getNode("Point2f")
    # img_prop = fs.getNode("ImageSize")
    # print(fn.mat().shape)
    # m = fn.mat()
    # cols = img_prop.at(0).real()
    # rows = img_prop.at(1).real()
    # with codecs.open('facelandmarks.yml', 'r', encoding='utf8') as f:
    #  yml_dict = yaml.safe_load(f)
    # print(yml_dict)
    num = 2
    try:
       opts, args = getopt.getopt(argv,"hi:o:n:",["ifile=","ofile=", "num="])
    except getopt.GetoptError:
       print('test.py -i <inputfile> -o <outputfile> -n <number>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('test.py -i <inputfile> -o <outputfile> -n <number>')
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg
       elif opt in ("-o", "--ofile"):
          outputfile = arg
       elif opt in ("-n", "--num"):
          num = int(arg)

    base = float(1/num)
    for i in range(1, num) :
        alp = '%.1f' % (base*i)
        print(alp);
        morphImage(inputfile, outputfile, float(1.0))
