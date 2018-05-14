#!/usr/bin/env python

### I AM USING OPENCV 3.0 ###
### SEE THE FILE 'question1-output.jpg' FOR A SCREEN CAP OF THE OUTPUT ###

import cv2
import sys
import math
import numpy
from numpy import matrix

def compute_projection_matrix(obj, img):
    A = []
    for i in range(0,10):
        row1 = numpy.array([obj[i][0], obj[i][1], obj[i][2], 1, 0, 0, 0, 0, -img[i][0][0]*obj[i][0], -img[i][0][0]*obj[i][1], -img[i][0][0]*obj[i][2], -img[i][0][0]])
        A.append(row1)
        row2 = numpy.array([0, 0, 0, 0, obj[i][0], obj[i][1], obj[i][2], 1, -img[i][0][1]*obj[i][0], -img[i][0][1]*obj[i][1], -img[i][0][1]*obj[i][2], -img[i][0][1]])
        A.append(row2)
    A = numpy.asarray(A)
    prod = numpy.dot(A.T, A)
    ret,eigenvalues,eigenvectors = cv2.eigen(prod,True)
    decompose_projection_matrix(eigenvectors)

def decompose_projection_matrix(eVecMat):
    m = eVecMat[11]
    m = m.reshape((3,4))
    sig = 0.00002654350455
    
    q1 = numpy.array([m[0][0], m[0][1], m[0][2]], numpy.float32)
    q2 = numpy.array([m[1][0], m[1][1], m[1][2]], numpy.float32)
    q3 = numpy.array([m[2][0], m[2][1], m[2][2]], numpy.float32)
    q4 = numpy.array([m[0][3], m[1][3], m[2][3]], numpy.float32)
    
    o_x = numpy.dot(q1,q3)
    o_y = numpy.dot(q2,q3)
    f_x = math.sqrt(numpy.dot(q1,q1) - o_x*o_x)
    f_y = math.sqrt(numpy.dot(q2,q2) - o_y*o_y)
    
    rVec = []
    r1 = numpy.array([-(o_x*m[2][0] - m[0][0])/f_x, -(o_x*m[2][1] - m[0][1])/f_x, -(o_x*m[2][2] - m[0][2])/f_x])
    rVec.append(r1)
    r2 = numpy.array([-(o_y*m[2][0] - m[1][0])/f_y, -(o_y*m[2][1] - m[1][1])/f_y, -(o_y*m[2][2] - m[1][2])/f_y])
    rVec.append(r2)
    r3 = numpy.array([-m[2][0]/sig, -m[2][1]/sig, -m[2][2]/sig])
    rVec.append(r3)
    rVec = numpy.asarray(rVec)
    print'Computed Rotation:'
    print rVec

    t_z = m[2][3]
    t_x = (o_x*t_z - m[0][3])/f_x
    t_y = (o_y*t_z - m[1][3])/f_y
    tVec = numpy.array([t_x, t_y, t_z])
    print'Computed Translation:'
    print tvec

    camMat = numpy.array([[-f_x/sig, 0, 0], [0, -f_y/sig, 0], [0, 0, 1]], numpy.float32)
    print'Computed Camera Matrix:'
    print camMat

R = numpy.array([[0.902701, 0.051530, 0.427171],
                 [0.182987, 0.852568, -0.489535],
                 [-0.389418, 0.520070, 0.760184]],
                numpy.float32)

rvec = cv2.Rodrigues(R)[0]
print 'Initial Rotation'
print R

cameraMatrix = numpy.array([[-1100.000000, 0.000000, 0.000000],
                            [0.000000, -2200.000000, 0.000000],
                            [0.000000, 0.000000, 1.000000]],numpy.float32)

print 'Initial Camera Matrix'
print cameraMatrix

tvec = numpy.array([12,16,21], numpy.float32)

print 'Initial Translation'
print tvec

objectPoints = numpy.array([[0.1251, 56.3585, 19.3304],
                            [80.8741, 58.5009, 47.9873],
                            [35.0291, 89.5962, 82.2840],
                            [74.6605, 17.4108, 85.8943],
                            [71.0501, 51.3535, 30.3995],
                            [1.4985, 9.1403, 36.4452],
                            [14.7313, 16.5899, 98.8525],
                            [44.5692, 11.9083, 0.4669],
                            [0.8911, 37.7880, 53.1663],
                            [57.1184, 60.1764, 60.7166]], numpy.float32) 

"""print 'Initial ObjectPoints'
print objectPoints"""

imagepoints,jac = cv2.projectPoints(objectPoints, rvec , tvec, cameraMatrix, None)
"""print 'Image Points'
print imagepoints"""

compute_projection_matrix(objectPoints, imagepoints)





