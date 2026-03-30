############## CardRecognise ##############
# 
# Author: Ludvig Overland
# Date: 27/3-26
# Description: Tar de inringade korten från cardIdentify.py
#              och hittar vilken rank och suit kortet har.
#              Projektet tar mycket inspiration från Evan Juras liknande projekt,
#              tanken är att man ska göra det på samma sätt:
# Method:
#   1. Hitta korten genom processen från cardIdentify.py
#   2. Croppa ut varenda kort och ta sedan en skärmbild på övre vänstra hörnet, där rank och suit finns
#   3. Dela den skärmdumpen på hälften för att isolera suit och rank för sig
#   4. Jämför med bildbiblioteket jag snodde från Evan Juras
#   5. Den med minst överlappande pixlar antas vara rätt.
#
#   Se denna videon för en bättre förklaring: https://www.youtube.com/watch?v=m-QPjO-2IkA

import cv2
import numpy as np
import os

CARD_MIN_AREA = 10000
CARD_MAX_AREA = 200000
BKG_THRESH = 60

# ── Bildbibliotek ──────────────────────────────────────────────────────────────
RANK_NAMES = ["Ace","Two","Three","Four","Five","Six","Seven",
              "Eight","Nine","Ten","Jack","Queen","King"]
SUIT_NAMES = ["Clubs","Diamonds","Hearts","Spades"]

RANK_SWEDISH = {
    "Ace": "Ess", "Two": "Två", "Three": "Tre", "Four": "Fyra",
    "Five": "Fem", "Six": "Sex", "Seven": "Sju", "Eight": "Åtta",
    "Nine": "Nio", "Ten": "Tio", "Jack": "Knekt", "Queen": "Dam", "King": "Kung"
}
SUIT_SWEDISH = {
    "Clubs": "Klöver", "Diamonds": "Ruter", "Hearts": "Hjärter", "Spades": "Spader"
}

# ── Nya funktioner ─────────────────────────────────────────────────────────────

def load_reference_images(image_folder="img"):
    """Laddar referensbilder och tröskelvärdar dem (ingen invertering)."""
    ranks = {}
    suits = {}
    for name in RANK_NAMES:
        path = os.path.join(image_folder, f"{name}.jpg")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (70, 125))
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            ranks[name] = img
    for name in SUIT_NAMES:
        path = os.path.join(image_folder, f"{name}.jpg")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (70, 100))
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            suits[name] = img
    return ranks, suits


def match_template(roi, reference_dict, target_size):
    """
    Applicerar ROI som mask på referensbilden.
    Färre vita pixlar = bättre matchning.
    """
    roi_resized = cv2.resize(roi, target_size)
    _, roi_thresh = cv2.threshold(roi_resized, 127, 255, cv2.THRESH_BINARY)

    best_name = "Okänd"
    best_score = float("inf")
    for name, ref in reference_dict.items():
        masked = cv2.bitwise_and(roi_thresh, ref)
        score = np.sum(masked)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def get_corner_roi(warped_card):
    """Plockar ut övre vänstra hörnet och delar i rank (övre) och suit (nedre)."""
    corner = warped_card[0:185, 0:70]
    rank_roi = corner[0:125, 0:70]
    suit_roi = corner[85:185, 0:70]
    return rank_roi, suit_roi


def warp_card(frame, contour):
    """Perspektivkorrigerar ett kort så det blir rektangulärt (bird's-eye view)."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
    else:
        x, y, w, h = cv2.boundingRect(contour)
        pts = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.float32([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ])

    dst = np.float32([[0, 0], [350, 0], [350, 500], [0, 500]])
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame, M, (350, 500))
    return warped


def identify_card(frame_gray, contour, rank_refs, suit_refs):
    """Identifierar ett korts rank och suit. Returnerar svensk sträng."""
    warped = warp_card(frame_gray, contour)
    rank_roi, suit_roi = get_corner_roi(warped)

    rank_name = match_template(rank_roi, rank_refs, (70, 125))
    suit_name = match_template(suit_roi, suit_refs, (70, 100))

    rank_sv = RANK_SWEDISH.get(rank_name, rank_name)
    suit_sv = SUIT_SWEDISH.get(suit_name, suit_name)
    return f"{suit_sv} {rank_sv}"


def get_contour_center(contour):
    """Returnerar mittpunkten för en contour."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def draw_card_labels(frame, card_contours, card_labels, antal_kort):
    """
    Ritar kortidentiteter som text ovanpå korten samt
    antal kort i nedre vänstra hörnet – på den oprocessade bilden.
    """
    overlay = frame.copy()

    for contour, label in zip(card_contours, card_labels):
        cx, cy = get_contour_center(contour)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(overlay,
                      (cx - tw // 2 - 4, cy - th - 4),
                      (cx + tw // 2 + 4, cy + baseline + 4),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(overlay, label,
                    (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    h, w = overlay.shape[:2]
    count_text = f"Mangd Kort: {antal_kort}"
    (tw, th), baseline = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(overlay,
                  (6, h - th - baseline - 20),
                  (tw + 14, h - 10),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(overlay, count_text,
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return overlay


# ── Befintlig kod (oförändrad) ─────────────────────────────────────────────────

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh


def find_cards(thresh_image):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if CARD_MIN_AREA < area < CARD_MAX_AREA:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) >= 4:
                card_contours.append(cnt)

    return card_contours


# ── Huvudloop ──────────────────────────────────────────────────────────────────

rank_refs, suit_refs = load_reference_images(".")

cv2.namedWindow("CameraFeed")
cv2.namedWindow("ProcessedFeed")

vc = cv2.VideoCapture(1)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    processed_frame = preprocess_image(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    card_contours = find_cards(processed_frame)
    antal_kort = len(card_contours)

    # Identifiera varje kort
    card_labels = []
    for cnt in card_contours:
        label = identify_card(frame_gray, cnt, rank_refs, suit_refs)
        card_labels.append(label)

    # ── Processerat fönster ──
    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(display_frame, card_contours, -1, (255, 0, 0), 2)
    h, w = display_frame.shape[:2]
    text = f"Mangd Kort: {antal_kort}"
    cv2.putText(display_frame, text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # ── Oprocesserat fönster med labels och kortantal ──
    annotated_frame = draw_card_labels(frame, card_contours, card_labels, antal_kort)

    cv2.imshow("CameraFeed", annotated_frame)
    cv2.imshow("ProcessedFeed", display_frame)

    rval, frame = vc.read()
    key = cv2.waitKey(20)

    if key == 27:
        break

cv2.destroyAllWindows()
vc.release()