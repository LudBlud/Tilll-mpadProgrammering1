######## CardIdentify #######
#
# Author: Ludvig Overland
# Date: 19/3-26
# Description: Tar den thresholdade bilden, hittar contours runt
#              de vita delarna och identifierar hur många spelkort
#              som finns i bilden.

import cv2
import numpy as np

BKG_THRESH = 70

# Minsta och största area för att räknas som ett kort
CARD_MIN_AREA = 10000
CARD_MAX_AREA = 200000

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh


def find_cards(thresh_image):
    #Hittar contours och filtrerar ut de som ser ut som kort
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if CARD_MIN_AREA < area < CARD_MAX_AREA:
            # Kolla att formen är ungefär rektangulär
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) >= 4:
                card_contours.append(cnt)

    return card_contours


# Skapa fönster
cv2.namedWindow("CameraFeed")
cv2.namedWindow("ProcessedFeed")

vc = cv2.VideoCapture(1)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    processed_frame = preprocess_image(frame)

    # Hitta kortens contours
    card_contours = find_cards(processed_frame)
    antal_kort = len(card_contours)

    # Rita contours på den processerade bilden (konvertera till BGR så text syns i färg)
    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(display_frame, card_contours, -1, (255, 0, 0), 2)

    # Skriv antal kort längst ner på bilden
    h, w = display_frame.shape[:2]
    text = f"Mangd Kort: {antal_kort}"
    cv2.putText(display_frame, text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("CameraFeed", frame)
    cv2.imshow("ProcessedFeed", display_frame)

    rval, frame = vc.read()
    key = cv2.waitKey(20)

    if key == 27:
        break

cv2.destroyAllWindows()
vc.release()