import cv2 as cv
import numpy as np

# I first imported my image, turned it to grayscale, then resized it
img = cv.imread("img1.jpg")

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.resize(img, (500 , 700))
gray_img = cv.resize(gray_img, (500 , 700))

# I added a blurred image to make it easier to identify contours
blurred_img = cv.GaussianBlur(gray_img, (5,5), 1)

# I identified edges of image
edges_img = cv.Canny(blurred_img, 180, 180)

# I found  the contours
contours, _ = cv.findContours(edges_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
contour_img = cv.drawContours(contour_img, contours, -1, (255,255,0), 5)

# I resized my blurred, edges, and contoured image
blurred_img = cv.resize(img, (500 , 700))
edges_img = cv.resize(edges_img, (500 , 700))
contour_img = cv.resize(contour_img, (500 , 700))

corners_img = img.copy()
maxArea = 0 # this will store my largest contour area
biggest = [] # this will store the biggest 4 corner contour

for i in contours:
    area = cv.contourArea(i)
    if area <= 500 :
        continue # skips to next contour

    # I found perimeter and approximated it
    peri = cv.arcLength(i, True)
    edges = cv.approxPolyDP(i, 0.02*peri, True)

    # if my shape has 4 corners and is bigger than any previous shape, it will be stored in the values i defined above
    if area > maxArea and len(edges) == 4 :
        biggest = edges
        maxArea = area

    # I drew a circle on each corner
    if len(biggest) != 0:
        cv.drawContours(corners_img, biggest, -1, (255, 0, 255), 25)


import cv2 as cv
import numpy as np

# to reorder my points, i have to image the picture's contour lines to be on an x-y axis,
# x is positive to the right and y is positive downwards. So to find values of these coordinates
# the ones with the highest difference between the y and x values (aka the top right and bottom left corners)
# will have the value x-y, while the rest will have x+y
def order(points):
    # points now are ordered in 4 rows two column format
    points = points.reshape((4, 2))
    # empty array for later
    ordered = np.zeros((4, 2), dtype=np.float32)

    sum = points.sum(axis=1) # x+y
    difference = np.diff(points, axis=1) # x-y

    # these define which corner is which as I explained above
    ordered[0] = points[np.argmin(sum)]
    ordered[1] = points[np.argmin(difference)]
    ordered[2] = points[np.argmax(sum)]
    ordered[3] = points[np.argmax(difference)]

    return ordered # this will return the array with my ordered corners

height, width = img.shape[:2]

original = order(biggest)
# this code just basically stretches out my corners so they now fit with the coordinates of my original image
new = np.array([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
], dtype=np.float32)

matrix = cv.getPerspectiveTransform(original, new)
warped = cv.warpPerspective(img, matrix, (width, height))


gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 1)

recolored = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11, 2)


# I displayed all my different versions
cv.imshow("document", img)
cv.imshow("gray document", gray_img)
cv.imshow("blurred", blurred_img)
cv.imshow("edges", edges_img)
cv.imshow("contour", contour_img)
cv.imshow("corners", corners_img)
cv.imshow("warped", warped)
cv.imshow("recolored", recolored)

cv.waitKey(0)