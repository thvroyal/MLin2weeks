import cv2
import imutils

img = cv2.imread("shapes_and_colors.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the threshold image
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# (x,y,w,h) -> (x,y)
contours = imutils.grab_contours(contours)

# loop over contours
for cnt in contours:
    if cv2.contourArea(cnt) > 60:
        M = cv2.moments(cnt)
        # center point : (cX, cY)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # draw the contour and center of the shape on the image
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 3, (255, 255, 255), -1)
        cv2.putText(img, 'center', (cX - 5, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Image", img)
        cv2.waitKey()
