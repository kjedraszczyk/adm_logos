from pathlib import Path
import cv2
import numpy as np

# =========================
# 1. Wczytywanie danych
# =========================

def load_league_logos(base_dir="logos"):
    """
    base_dir/
        eng/
        esp/
        fra/
        ger/
        ita/
    Zwraca: dict liga -> lista (ścieżka, obraz)
    """
    base_path = Path(base_dir)
    data = {}

    for league_dir in base_path.iterdir():
        if league_dir.is_dir():
            league_name = league_dir.name
            data[league_name] = []
            for img_path in league_dir.glob("*.png"):
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                data[league_name].append((img_path, img))

    return data


# =========================
# 2. Maska logotypu
# =========================

def make_binary_mask(img):
    """
    Tworzy binarną maskę logotypu.
    Jeśli jest kanał alfa, używa go. W przeciwnym razie – Otsu na szarości.
    """
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        return mask

    # jeśli brak kanału alfa – progowanie Otsu na szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# =========================
# 3. Kolistość + wielkość
# =========================

def shape_features_from_mask(mask):
    """
    Zwraca: (kolistość, aspect_ratio, pole)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0 or area == 0:
        circularity = 0.0
    else:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0.0

    size_pixels = int(np.count_nonzero(mask))

    return float(circularity), float(aspect_ratio), float(size_pixels)


# =========================
# 4. HUE / kolor + „tekstura kolorystyczna”
# =========================

def hue_features(img, mask=None, bins=16):
    """
    Zwraca:
        - znormalizowany histogram HUE (bins,)
        - entropię histogramu jako miarę złożoności kolorystycznej
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]

    if mask is None:
        mask = np.ones_like(h, dtype=np.uint8) * 255

    hist = cv2.calcHist([h], [0], mask, [bins], [0, 180])
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s

    eps = 1e-8
    entropy = -np.sum(hist * np.log(hist + eps))

    return hist, float(entropy)


def dominant_hue(hist):
    """
    Zwraca indeks binu o największej wartości – uproszczony dominujący kolor.
    """
    if hist is None or len(hist) == 0:
        return None
    return int(np.argmax(hist))


# =========================
# 5. „Ilość napisów” – gęstość krawędzi
# =========================

def text_like_edge_density(img, mask=None):
    """
    Prosty wskaźnik ilości napisów: gęstość krawędzi Canny w obszarze logo.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    edge_pixels = np.count_nonzero(edges)
    mask_pixels = np.count_nonzero(mask) if mask is not None else edges.size

    if mask_pixels == 0:
        return 0.0

    density = edge_pixels / mask_pixels
    return float(density)


# =========================
# 6. Ekstrakcja cech dla jednego herbu
# =========================

def extract_features_for_logo(img):
    """
    Zwraca wektor cech dla jednego herbu:
    [kolistość, aspect_ratio, wielkość, entropia_HUE, gęstość_krawędzi]
    + można osobno trzymać cały histogram HUE, jeśli chcesz
    """
    mask = make_binary_mask(img)

    circularity, aspect_ratio, size_pixels = shape_features_from_mask(mask)
    if circularity is None:
        # coś poszło nie tak z konturami
        return None, None

    hue_hist, hue_entropy = hue_features(img, mask)
    edge_density = text_like_edge_density(img, mask)

    feature_vector = np.array([
        circularity,
        aspect_ratio,
        size_pixels,
        hue_entropy,
        edge_density
    ], dtype=np.float32)

    return feature_vector, hue_hist


# =========================
# 7. Standaryzacja cech
# =========================

def standardize_features(feature_matrix):
    """
    feature_matrix: (N x D)
    Zwraca: (X_std, means, stds)
    """
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)

    stds_safe = np.where(stds == 0, 1.0, stds)
    X_std = (feature_matrix - means) / stds_safe

    return X_std, means, stds_safe


# =========================
# 8. Główna pętla: ekstrakcja i statystyki ligowe
# =========================

def compute_league_features(logos_by_league):
    """
    Iteruje po wszystkich herbach, wyciąga cechy i buduje:
    - X: macierz cech (N x D)
    - y: lista etykiet lig (N)
    - hue_hists: lista histogramów HUE (N)
    """
    X = []
    y = []
    hue_hists = []

    for league, items in logos_by_league.items():
        for path, img in items:
            features, h_hist = extract_features_for_logo(img)
            if features is None:
                continue
            X.append(features)
            y.append(league)
            hue_hists.append(h_hist)

    X = np.vstack(X) if len(X) > 0 else np.empty((0, 5), dtype=np.float32)
    return X, y, hue_hists


def aggregate_league_stats(X_std, y):
    """
    Liczy średnią z cech (na z‑score) osobno dla każdej ligi.
    Zwraca: dict liga -> dict nazwa_cejh -> średnia
    """
    X_std = np.asarray(X_std)
    y = np.asarray(y)

    leagues = np.unique(y)
    stats = {}

    feature_names = [
        "circularity_z",
        "aspect_ratio_z",
        "size_z",
        "hue_entropy_z",
        "edge_density_z"
    ]

    for league in leagues:
        idx = np.where(y == league)[0]
        if len(idx) == 0:
            continue
        mean_vec = X_std[idx].mean(axis=0)
        stats[league] = {
            name: float(val) for name, val in zip(feature_names, mean_vec)
        }

    return stats


import matplotlib.pyplot as plt
import numpy as np


def plot_circularity_by_league(X, y, save_path=None):
    """
    Tworzy profesjonalny boxplot kolistości z pełnym opisem.
    """
    # Zbierz dane per liga
    leagues = sorted(set(y))
    data = []
    for league in leagues:
        vals = [X[i, 0] for i in range(len(y)) if y[i] == league]
        data.append(vals)

    # Kolory dla lig
    colors = {
        'eng': '#1f77b4',  # Premier League - niebieski
        'esp': '#ff7f0e',  # La Liga - pomarańczowy
        'fra': '#2ca02c',  # Ligue 1 - zielony
        'ger': '#d62728',  # Bundesliga - czerwony
        'ita': '#9467bd'  # Serie A - fioletowy
    }

    plt.figure(figsize=(12, 8))
    box_plot = plt.boxplot(data, tick_labels=leagues, patch_artist=True, notch=False, showmeans=True)

    # Kolorowanie skrzynek
    for patch, league in zip(box_plot['boxes'], leagues):
        patch.set_facecolor(colors.get(league, '#9edae5'))
        patch.set_alpha(0.7)

    # Linie mediany na czerwono
    for median in box_plot['medians']:
        median.set(color='red', linewidth=3)

    # Średnie na czarno
    for mean in box_plot['means']:
        mean.set(marker='D', markerfacecolor='black', markersize=8)

    plt.ylabel('Kolistość herbu\n(0 = kwadrat, 1 = idealne koło)', fontsize=14, fontweight='bold')
    plt.xlabel('Liga', fontsize=14, fontweight='bold')
    plt.title('ROZKŁAD KSZTAŁTÓW HERBÓW W LIGACH TOP5\n\n'
              'WYJAŚNIENIE:\n'
              'SKRZYNKA = 50% herbów (od 25% do 75% rozkładu)\n'
              'LINIA = MEDIANA (połowa herbów ma mniejszą/większą kolistość)\n'
              'DIAMENT = ŚREDNIA\n'
              'WĄSY = zakres 95% herbów\n'
              'KROPKA = outliery (skrajne herby)\n\n'
              'INTERPRETACJA:\n'
              'Wysoka mediana + wąska skrzynka = JEDNOLITE, OKRĄGŁE herby\n'
              'Niska mediana + szeroka skrzynka = RÓŻNORODNE kształty',
              fontsize=16, fontweight='bold', pad=20)

    # Grid + styl
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Legenda
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, label='Mediana'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                   markersize=8, label='Średnia'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label='50% herbów')
    ]
    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Wykres zapisany: {save_path}")

    plt.show()
    plt.close()

# =========================
# 9. Uruchomienie
# =========================

if __name__ == "__main__":
    BASE_DIR = "herby"  # dostosuj, jeśli masz inną nazwę

    logos_by_league = load_league_logos(BASE_DIR)
    print("Znalezione ligi:", list(logos_by_league.keys()))

    # Ekstrakcja cech dla wszystkich herbów
    X, y, hue_hists = compute_league_features(logos_by_league)
    print("Liczba herbów:", len(y))
    print("Macierz cech X shape:", X.shape)


    if len(y) > 0:
        # Wizualizacja kolistości
        plot_circularity_by_league(X, y, "kolistosc_ligi.png")

        print("WYNIKI WIZUALIZACJI:")
        print("Sprawdź plik 'kolistosc_ligi.png' lub okno wykresu!")

        # =========================
        # Standaryzacja cech
        X_std, means, stds = standardize_features(X)

        # Statystyki ligowe na z‑score
        league_stats = aggregate_league_stats(X_std, y)

        print("\nŚrednie z‑score cech po ligach:")
        for league, stats in league_stats.items():
            print(f"\nLiga: {league}")
            for name, val in stats.items():
                print(f"  {name}: {val:.3f}")
    else:
        print("Brak poprawnie wczytanych herbów.")