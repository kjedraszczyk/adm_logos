from pathlib import Path
import cv2
import numpy as np

def ladowanie_herbow(base_dir="herby"):
    base = Path(base_dir)
    data = {}

    for liga in base.iterdir():
        nazwaligi = liga.name
        data[nazwaligi] = []
        for image in liga.glob("*.png"):
            img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            data[nazwaligi].append((image, img))

    return data

def maska(img):
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        return mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def kontury(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0 or area == 0:
        circularity = 0.0
    else:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0.0

    return circularity, aspect_ratio

def kolowosc(logos_by_league):
    stats = {}

    for league, items in logos_by_league.items():
        circularities = []

        for path, img in items:
            mask = maska(img)
            circ, _ = kontury(mask)
            if circ is not None:
                circularities.append(circ)

        if circularities:
            stats[league] = {
                "mean_circularity": float(np.mean(circularities)),
                "median_circularity": float(np.median(circularities)),
                "n": len(circularities)
            }
        else:
            stats[league] = {"mean_circularity": None, "n": 0}

    return stats



if __name__ == "__main__":
    logos_by_league = ladowanie_herbow()
    circ_stats = kolowosc(logos_by_league)
    print("Kolistość herbów po ligach:")
    for league, data in circ_stats.items():
        print(f"{league}: średnia={data['mean_circularity']:.3f}, n={data['n']}")

if 0 == 1:
    print("test1")