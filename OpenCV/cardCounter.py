######### Cardrecogniser med Open CV ###########
#
# Author: Ludvig Overland
# Date: 11/3/26
# Description: Tar videon från CameraFeed.py och greyscalar
#              och thresholdar för känna igen kanten på 
#              ett kort inom videon och räkna hur många
#              det är.

#Importera nödvändiga bibliotek
import cv2
import numpy as np
import time



