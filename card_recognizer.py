"""
Kortspels-igenkanning via Camo-kamera
======================================
Kopplar Python till din telefons kamera via Camo och kanner igen spelkort i realtid.

Installation:
    pip install opencv-python numpy

Krav:
    - Camo-appen installerad på telefon
    - Camo-drivrutinen installerad pa datorn
    - Starta Camo på telefonen och datorn innan du kör skriptet
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path


# ──────────────────────────────────────────────
#  KONFIGURATION
# ──────────────────────────────────────────────

CARD_DATA_FILE = "kort_data.json"
CAPTURE_DIR    = "inlarda_kort"   
CAMERA_INDEX   = 1


# ──────────────────────────────────────────────
#  HJALPFUNKTIONER
# ──────────────────────────────────────────────

def oppna_kamera(idx: int = 1) -> cv2.VideoCapture:
    """
    Oppnar Camo-kameran utan explicit backend (CAP_DSHOW ger svart bild).
    """
    print("Ansluter till Camo (index 1)...")
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Varmer upp kameran (2-3 sekunder)...")
    for _ in range(60):
        cap.read()
    print("Kamera redo!")
    return cap


def text_overlay(bild, meddelande: str, pos: tuple,
                 skala: float = 0.7, farg: tuple = (255, 255, 255), tjocklek: int = 2) -> None:
    ersatt = (meddelande
              .replace("\u00e5", "a").replace("\u00c5", "A")
              .replace("\u00e4", "a").replace("\u00c4", "A")
              .replace("\u00f6", "o").replace("\u00d6", "O")
              .replace("\u2013", "-").replace("\u2014", "-"))
    cv2.putText(bild, ersatt, pos, cv2.FONT_HERSHEY_SIMPLEX, skala, farg, tjocklek)


def hitta_camo_kamera() -> int:
    print("Soker efter tillgangliga kameror...")
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  Kamera {idx}: {w}x{h} px")
    vald = input(f"\nAnge kameraindex (standard {CAMERA_INDEX}): ").strip()
    return int(vald) if vald.isdigit() else CAMERA_INDEX


def extrahera_kortbild(frame) -> np.ndarray:
    gra = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    suddig = cv2.GaussianBlur(gra, (5, 5), 0)
    kanter = cv2.Canny(suddig, 50, 150)
    konturer, _ = cv2.findContours(kanter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if konturer:
        storst = max(konturer, key=cv2.contourArea)
        yta = cv2.contourArea(storst)
        if yta > frame.shape[0] * frame.shape[1] * 0.10:
            x, y, w, h = cv2.boundingRect(storst)
            return frame[y:y+h, x:x+w]
    return frame


def normalisera(bild, storlek: tuple = (200, 300)) -> np.ndarray:
    return cv2.resize(bild, storlek)


def berakna_likhet(bild1, bild2) -> float:
    b1 = cv2.cvtColor(normalisera(bild1), cv2.COLOR_BGR2HSV)
    b2 = cv2.cvtColor(normalisera(bild2), cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([b1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([b2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return max(0.0, cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)) * 100


# ──────────────────────────────────────────────
#  INLARNING
# ──────────────────────────────────────────────

def spara_kort(kortnamn: str, bild) -> None:
  
    Path(CAPTURE_DIR).mkdir(exist_ok=True)

    saker_namn = (kortnamn
                  .replace(" ", "_")
                  .replace("/", "-")
                  .replace("\u00e5", "a").replace("\u00c5", "A")
                  .replace("\u00e4", "a").replace("\u00c4", "A")
                  .replace("\u00f6", "o").replace("\u00d6", "O"))

    filnamn = f"{CAPTURE_DIR}/{saker_namn}.jpg"

    lyckades = cv2.imwrite(filnamn, bild)
    if not lyckades:
        print(f"  FEL: Kunde inte spara bilden till {filnamn}")
        print(f"  Kontrollera att mappen '{CAPTURE_DIR}' ar skrivbar.")
        return

    data: dict = {}
    if os.path.exists(CARD_DATA_FILE):
        with open(CARD_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[kortnamn] = filnamn
    with open(CARD_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  OK Kort sparat: '{kortnamn}' -> {filnamn}")


def inlarningslage(kamera_idx: int) -> None:
    print("\n===================================")
    print("  INLARNINGSLAGE")
    print("  MELLANSLAG = ta bild  |  ESC = avsluta")
    print("===================================\n")

    cap = oppna_kamera(kamera_idx)
    if not cap.isOpened():
        print("Kunde inte oppna kameran.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        visning = frame.copy()
        text_overlay(visning, "INLARNING: MELLANSLAG=ta bild  ESC=avsluta",
                     (10, 30), skala=0.6, farg=(0, 255, 0))

        h, w = frame.shape[:2]
        mx, my = w // 5, h // 6
        cv2.rectangle(visning, (mx, my), (w - mx, h - my), (0, 255, 0), 2)

        cv2.imshow("Inlarning av kort", visning)
        tangent = cv2.waitKey(1) & 0xFF

        if tangent == 27:
            break
        elif tangent == 32:
            kortbild = extrahera_kortbild(frame)
            kortnamn = input("Ange kortets namn (t.ex. 'Hjarter Ess'): ").strip()
            if kortnamn:
                spara_kort(kortnamn, kortbild)
            else:
                print("  Inget namn - bilden sparades inte.")

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
#  IGENKANNING
# ──────────────────────────────────────────────

def ladda_kortdata() -> dict:
    if not os.path.exists(CARD_DATA_FILE):
        print("Ingen kortdata hittad. Kor inlarningslaget forst (alternativ 1).")
        return {}
    with open(CARD_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    kort = {}
    for namn, sokväg in data.items():
        bild = cv2.imread(sokväg)
        if bild is not None:
            kort[namn] = bild
        else:
            print(f"  Varning: kunde inte lasa {sokväg}")
    print(f"Laddade {len(kort)} kort fran disk.")
    return kort


def kanna_igen_kort(aktuell_bild, kortbibliotek: dict, troskel: float = 60.0) -> tuple:
    bast_namn  = "Okant kort"
    bast_poang = 0.0
    for namn, mall in kortbibliotek.items():
        poang = berakna_likhet(aktuell_bild, mall)
        if poang > bast_poang:
            bast_poang = poang
            bast_namn  = namn
    if bast_poang < troskel:
        return "Okant kort", bast_poang
    return bast_namn, bast_poang


def igenkanningslage(kamera_idx: int) -> None:
    kortbibliotek = ladda_kortdata()
    if not kortbibliotek:
        return

    print("\n===================================")
    print("  IGENKANNINGSLAGE  -  ESC = avsluta")
    print("===================================\n")

    cap = oppna_kamera(kamera_idx)
    if not cap.isOpened():
        print("Kunde inte oppna kameran.")
        return

    bildraknare = 0
    aktuellt_resultat = ("Soker...", 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        bildraknare += 1
        if bildraknare % 15 == 0:
            kortbild = extrahera_kortbild(frame)
            aktuellt_resultat = kanna_igen_kort(kortbild, kortbibliotek)

        namn, poang = aktuellt_resultat
        visning = frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(visning, (0, h - 80), (w, h), (0, 0, 0), -1)
        farg = (0, 255, 0) if poang >= 60 else (0, 165, 255)
        text_overlay(visning, f"Kort: {namn}", (10, h - 50), skala=0.8, farg=farg)
        text_overlay(visning, f"Likhet: {poang:.1f}%",
                     (10, h - 15), skala=0.6, farg=(200, 200, 200))

        mx, my = w // 5, h // 6
        cv2.rectangle(visning, (mx, my), (w - mx, h - my), (0, 255, 255), 2)

        cv2.imshow("Kortigenkanning", visning)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
#  HUVUDMENY
# ──────────────────────────────────────────────

def main() -> None:
    print("╔════════════════════════════════════╗")
    print("║   KORTSPELS-IGENKANNING MED CAMO   ║")
    print("╠════════════════════════════════════╣")
    print("║  1. Lar in nya kort                ║")
    print("║  2. Starta igenkanning             ║")
    print("║  3. Hitta Camo-kamera automatiskt  ║")
    print("║  4. Avsluta                        ║")
    print("╚════════════════════════════════════╝")

    kamera_idx = CAMERA_INDEX

    while True:
        val = input("\nValj alternativ (1-4): ").strip()
        if val == "1":
            inlarningslage(kamera_idx)
        elif val == "2":
            igenkanningslage(kamera_idx)
        elif val == "3":
            kamera_idx = hitta_camo_kamera()
            print(f"Kamera {kamera_idx} ar nu vald.")
        elif val == "4":
            print("Avslutar. Hej da!")
            break
        else:
            print("Ogiltigt val - ange 1, 2, 3 eller 4.")


if __name__ == "__main__":
    main()