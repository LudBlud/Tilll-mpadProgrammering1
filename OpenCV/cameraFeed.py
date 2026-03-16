########### Webbkamera-feed i OpenCV ##############
#
# Author: Ludvig Overland
# Date 11/3-26
# Description: Tar ett videofeed genom mobilkamera och 
#              Camo och displayar det i ett fönster med
#              hjälp av OpenCV

import cv2

cv2.namedWindow("CameraFeed") #Öppnar ett window och kallar det CameraFeed
vc = cv2.VideoCapture(1) #Bestämmer var videofeeden ska komma från

if vc.isOpened(): #Kollar så att den öppnades rätt
    rval, frame = vc.read() #Rval betyder om det förra är sant(tror jag), gör detta. Sedan setter man frame till kamerans output.
else:
    rval = False
    
while rval == True: #kör programmet tills output från kameran bryts
    cv2.imshow("CameraFeed", frame) #Visar frame (video) i fönstret "CameraFeed"
    rval, frame = vc.read() #Hämtar nästa frame från webbkameran
    key = cv2.waitKey(20)
    if key == 27: #Om tangenten som trycks är ESC stängs programmet av
        break
    
cv2. destroyWindow("CameraFeed")
vc.release()