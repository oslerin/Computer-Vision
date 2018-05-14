import numpy as np
import cv2

### I AM USING OPENCV 3.0 ###

#function to write on image
def write_on_image(image, text):
    fontFace = cv2.FONT_HERSHEY_DUPLEX;
    fontScale = 2.0
    thickness = 3
    textOrg = (10, 130)
    cv2.putText(image, text, textOrg, fontFace, fontScale, thickness, 8);
    return image

#function to determine the speed listed on the sign
#the assumption made here is that the contour with the largest area is the one to be compared with the .bng files
def determine_speed(image):
    largest_contour = big_contours[0]
    image_contour = image_original.copy()
    cv2.drawContours(image_contour, [largest_contour], 0, 255, 4)
    #find the leftmost, topmost, rightmost and bottommost points of this contour (corners of the rectangle)
    lm = tuple(largest_contour[largest_contour[:,:,0].argmin()][0])
    rm = tuple(largest_contour[largest_contour[:,:,0].argmax()][0])
    tm = tuple(largest_contour[largest_contour[:,:,1].argmin()][0])
    bm = tuple(largest_contour[largest_contour[:,:,1].argmax()][0])
    #determine if the rectangle in the image is angled up or down
    #this is neccesary to determine the order of the corners in the first parameter for getPerspectiveTransform
    if rm[1] < lm[1]:
        #if the y-value of the rightmost point is less than the y-value of the leftmost point it is angled down
        #this means the corners are topleft = topmost, topright = rightmost, bottomleft = leftmost and bottomright = bottommost
        p1 = np.float32([tm, rm, lm, bm])
    if rm[1] > lm[1]:
        #if the y-value of the rightmost point is greater than the y-value of the leftmost point it is angled up
        #this means the corners are topleft = leftmost, topright = topmost, bottomleft = bottommost and bottomright = rightmost
        p1 = np.float32([lm, tm, bm, rm])
    p2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    pers = cv2.getPerspectiveTransform(p1,p2)
    warp = cv2.warpPerspective(image,pers,(300,300))
    cv2.imshow('warped',warp)
    ##template match
    temp40 = cv2.imread('speed_40.bmp')
    temp80 = cv2.imread('speed_80.bmp')
    match1 = cv2.matchTemplate(warp, temp40, cv2.TM_CCOEFF_NORMED)
    match2 = cv2.matchTemplate(warp, temp80, cv2.TM_CCOEFF_NORMED)
    mi1,ma1,mil1,mal1 = cv2.minMaxLoc(match1)
    mi2,ma2,mil2,mal2 = cv2.minMaxLoc(match2)
    #the minimum value (mil and mi2) for each matchTemplate gives how accurate the match is
    acc_thres = -0.1 #determined by trial and error
    if mi1 < acc_thres and mi2 < acc_thres: #indicates case where the sign is neither 40 nor 80
        write_on_image(image_original, "Unknown")
    elif abs(mi1) > abs(mi2): #indicates case where 40 has a greater match
        write_on_image(image_original, "40 sign")
    elif abs(mi1) < abs(mi2): #indicates case where 80 has a greater match
        write_on_image(image_original, "80 sign")

#load a colour image and convert it to greyscale
image_original = cv2.imread('speedsign13.jpg')
image = image_original
image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
image = cv2.blur(image, (3,3))
cv2.imshow('original image',image_original)

#find and display canny edges
canny_threshold = 120;
image = cv2.Canny(image, canny_threshold, canny_threshold*2, apertureSize=3)
cv2.imshow('edge image', image)

#find all contours and save the 10 biggest
_, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
big_contours = sorted(contours, key=cv2.contourArea, reverse=True)[: 10]

#draw each of these contours, one after the other
for c in big_contours:
    image_contour = image_original.copy()
    cv2.drawContours(image_contour, [c], 0, 255, 4)
    cv2.imshow('contour', image_contour)
    area = cv2.contourArea(c)
    print'Area', area
    cv2.waitKey(0)

#determine whether its a speed sign or a stop sign
#create a list to store the number of verticies then determine the shape from the number of vertices
shape = None
vertex_list = []
for c in big_contours:
    approx = cv2.approxPolyDP(c,0.008*cv2.arcLength(c, True), True)
    vertex = len(approx)
    print vertex
    vertex_list.append(vertex)
for i in big_contours:
    for j in vertex_list:
        if j == 4:
            shape = True #rectangle = speed sign
        if j == 8:
            shape = False #octagon = stop sign

#determining speed
if shape == True: #rectangle so try to determine the speed
    determine_speed(image_original)
if shape == False: #stop sign so just print stop sign
    write_on_image(image_original, "Stop Sign")

cv2.imwrite('speedsign13-result.jpg',image_original)
cv2.waitKey(0)

cv2.destroyAllWindows()
