import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATENBANK_DIR = os.path.join(SCRIPT_DIR, "face_database")  # face_database/person1/img1.jpg ...
GRID_GROESSE  = 8     # Bild in 8x8 Grids aufteilen
BINS          = 256   # 256 Bins pro Histogramm -> 64 * 256 = 16384 Features
BILD_GROESSE  = 256   # Einheitliche Bildgröße
TESTBILD_PFAD = os.path.join(SCRIPT_DIR, "testbild.jpg")


# region LBP – Local Binary Pattern (eigene Implementierung, kein OpenCV)
def berechne_lbp(bild):
    hoehe, breite = bild.shape
    lbp_bild = np.zeros((hoehe, breite), dtype=np.uint8)

    # Randpixel werden ignoriert, daher range(1, ...-1)
    for i in range(1, hoehe - 1):
        for j in range(1, breite - 1):
            zentrum = bild[i, j]

            # 8 Nachbarn im Uhrzeigersinn vergleichen – Bit = 1 wenn Nachbar >= Zentrum
            code = 0
            code |= (bild[i - 1, j - 1] >= zentrum) << 7  # oben-links
            code |= (bild[i - 1, j    ] >= zentrum) << 6  # oben
            code |= (bild[i - 1, j + 1] >= zentrum) << 5  # oben-rechts
            code |= (bild[i    , j + 1] >= zentrum) << 4  # rechts
            code |= (bild[i + 1, j + 1] >= zentrum) << 3  # unten-rechts
            code |= (bild[i + 1, j    ] >= zentrum) << 2  # unten
            code |= (bild[i + 1, j - 1] >= zentrum) << 1  # unten-links
            code |= (bild[i    , j - 1] >= zentrum) << 0  # links

            # Ergebnis ist ein Wert zwischen 0 und 255
            lbp_bild[i, j] = code

    return lbp_bild
# endregion


# region LBPH – Local Binary Pattern Histogram
def berechne_lbph(lbp_bild, grid_groesse=GRID_GROESSE, bins=BINS):
    hoehe, breite = lbp_bild.shape
    zell_hoehe    = hoehe // grid_groesse
    zell_breite   = breite // grid_groesse

    histogramme = []

    # Bild in 8x8 = 64 Zellen aufteilen
    for zeile in range(grid_groesse):
        for spalte in range(grid_groesse):
            y_start = zeile  * zell_hoehe
            y_ende  = y_start + zell_hoehe
            x_start = spalte * zell_breite
            x_ende  = x_start + zell_breite

            zelle = lbp_bild[y_start:y_ende, x_start:x_ende]

            # Pro Zelle ein Histogramm mit 256 Bins berechnen (Wertebereich 0-255)
            hist, _ = np.histogram(zelle.ravel(), bins=bins, range=(0, 255))

            # Histogramm normalisieren, damit die Summe = 1
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)

            histogramme.append(hist)

    # Alle 64 Histogramme zu einem Feature-Vektor konkatenieren: 64 * 256 = 16384 Features
    return np.concatenate(histogramme)
# endregion


# Chi-Quadrat-Distanz: empfohlene Metrik für LBPH-Vergleiche
def chi_quadrat_distanz(vektor1, vektor2):
    zaehler = (vektor1 - vektor2) ** 2
    nenner  = vektor1 + vektor2
    maske   = nenner != 0  # Division durch Null vermeiden
    return float(np.sum(zaehler[maske] / nenner[maske]))


# Euklidische Distanz als alternative Metrik
def euklidische_distanz(vektor1, vektor2):
    return float(np.sqrt(np.sum((vektor1 - vektor2) ** 2)))


# Testbild einlesen, LBPH berechnen, mit allen gespeicherten Features vergleichen
def erkenne_person(testbild_pfad, X_features, y_labels, label_map, metrik="chi"):
    # Graustufen-Konvertierung
    bild = cv2.imread(testbild_pfad, cv2.IMREAD_GRAYSCALE)
    if bild is None:
        print(f"  [FEHLER] Testbild nicht gefunden: {testbild_pfad}")
        return None, None, None

    bild = cv2.resize(bild, (BILD_GROESSE, BILD_GROESSE))

    # LBP und LBPH des Testbilds berechnen
    lbp_bild    = berechne_lbp(bild)
    test_vektor = berechne_lbph(lbp_bild)

    # Mit allen gespeicherten Features vergleichen
    beste_distanz = float("inf")
    bestes_label  = -1

    for idx in range(len(X_features)):
        if metrik == "chi":
            distanz = chi_quadrat_distanz(test_vektor, X_features[idx])
        else:
            distanz = euklidische_distanz(test_vektor, X_features[idx])

        if distanz < beste_distanz:
            beste_distanz = distanz
            bestes_label  = int(y_labels[idx])

    # Beste Übereinstimmung ausgeben
    bester_name = label_map[bestes_label]
    return bestes_label, bester_name, beste_distanz


# Accuracy: Anteil korrekt erkannter Personen
def berechne_accuracy(wahre_labels, vorhergesagte_labels):
    korrekt = np.sum(wahre_labels == vorhergesagte_labels)
    return korrekt / len(wahre_labels)


# Confusion Matrix: Zeile = wahres Label, Spalte = vorhergesagtes Label
def berechne_confusion_matrix(wahre_labels, vorhergesagte_labels, anzahl_klassen):
    cm = np.zeros((anzahl_klassen, anzahl_klassen), dtype=int)
    for wahr, vorhergesagt in zip(wahre_labels, vorhergesagte_labels):
        cm[wahr][vorhergesagt] += 1
    return cm


def visualisiere_confusion_matrix(cm, label_map, accuracy):
    n              = len(label_map)
    personen_namen = [label_map[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 12))
    bild_plot = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(bild_plot, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(personen_namen, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(personen_namen, fontsize=8)

    # Zahlenwerte in jede Zelle schreiben
    schwellwert = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            farbe = "white" if cm[i, j] > schwellwert else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", color=farbe, fontsize=7)

    ax.set_xlabel("Vorhergesagtes Label", fontsize=12)
    ax.set_ylabel("Wahres Label", fontsize=12)
    ax.set_title(f"Confusion Matrix  -  Accuracy: {accuracy * 100:.2f}%", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print("=" * 60)
    print("  Gesichtserkennung mit LBPH")
    print("=" * 60)

    # --- Schritt 1: Bilder aus face_database laden, Features extrahieren ---
    print("\n[Schritt 1] Lade Bilder und berechne LBPH-Features ...\n")

    X_features   = []  # alle Feature-Vektoren
    y_labels     = []  # numerische Labels (0, 1, 2, ...)
    label_map    = {}  # {0: "person1", 1: "person2", ..., 29: "person30"}
    person_label = 0

    # Ordnerstruktur: face_database/person1/img1.jpg, face_database/person2/img1.jpg, ...
    for person_name in sorted(os.listdir(DATENBANK_DIR), key=lambda x: int(x.replace("person", ""))):
        person_pfad = os.path.join(DATENBANK_DIR, person_name)
        if not os.path.isdir(person_pfad):
            continue

        bilder_geladen = 0
        for bild_datei in sorted(os.listdir(person_pfad)):
            bild_pfad = os.path.join(person_pfad, bild_datei)

            # Graustufen-Konvertierung
            bild = cv2.imread(bild_pfad, cv2.IMREAD_GRAYSCALE)
            if bild is None:
                continue

            bild = cv2.resize(bild, (BILD_GROESSE, BILD_GROESSE))

            # LBP berechnen
            lbp_bild = berechne_lbp(bild)

            # LBPH berechnen
            feature_vektor = berechne_lbph(lbp_bild)

            # Feature-Vektor erzeugen
            X_features.append(feature_vektor)
            y_labels.append(person_label)
            label_map[person_label] = person_name
            bilder_geladen += 1

            print(f"  {person_name}: Feature-Länge = {len(feature_vektor)}")

        if bilder_geladen > 0:
            person_label += 1
        else:
            print(f"  [WARNUNG] {person_name}: Keine Bilder geladen – Ordner wird übersprungen.")

    X_features = np.array(X_features)  # Form: (Anzahl Bilder, 16384)
    y_labels   = np.array(y_labels)

    print(f"\n  X_features Shape : {X_features.shape}")
    print(f"  y_labels   Shape : {y_labels.shape}")
    print(f"  label_map  Größe : {len(label_map)} Einträge")

    # --- Schritt 2: Als NumPy-Dateien speichern ---
    print("\n[Schritt 2] Speichere NumPy-Dateien ...\n")

    # X_features.npy: alle Feature-Vektoren, Form (Anzahl Bilder, Feature-Länge)
    np.save(os.path.join(SCRIPT_DIR, "X_features.npy"), X_features)

    # y_labels.npy: numerische Labels, z.B. [0, 0, 0, 1, 1, 1, ...]
    np.save(os.path.join(SCRIPT_DIR, "y_labels.npy"), y_labels)

    # label_map.npy: Dictionary {0: "person1", 1: "person2", ..., 29: "person30"}
    np.save(os.path.join(SCRIPT_DIR, "label_map.npy"), label_map)

    print(f"  X_features.npy  gespeichert  ->  Shape   : {X_features.shape}")
    print(f"  y_labels.npy    gespeichert  ->  Shape   : {y_labels.shape}")
    print(f"  label_map.npy   gespeichert  ->  Einträge: {len(label_map)}")

    # --- Schritt 3: Gespeicherte Dateien laden und verifizieren ---
    print("\n[Schritt 3] Lade gespeicherte NumPy-Dateien ...\n")

    X_geladen   = np.load(os.path.join(SCRIPT_DIR, "X_features.npy"))
    y_geladen   = np.load(os.path.join(SCRIPT_DIR, "y_labels.npy"))
    map_geladen = np.load(
        os.path.join(SCRIPT_DIR, "label_map.npy"), allow_pickle=True
    ).item()

    print(f"  X_features geladen : {X_geladen.shape}")
    print(f"  y_labels   geladen : {y_geladen.shape}")
    print(f"  label_map  geladen : {map_geladen}")

    # --- Schritt 4: Evaluation mit Confusion Matrix und Accuracy ---
    print("\n[Schritt 4] Evaluation: Confusion Matrix und Accuracy ...\n")

    wahre_labels         = []
    vorhergesagte_labels = []

    for i in range(len(X_geladen)):
        test_vektor  = X_geladen[i]
        wahres_label = int(y_geladen[i])

        beste_distanz = float("inf")
        bestes_label  = -1

        # Chi-Quadrat-Distanz zu allen gespeicherten Feature-Vektoren berechnen
        for j in range(len(X_geladen)):
            distanz = chi_quadrat_distanz(test_vektor, X_geladen[j])
            if distanz < beste_distanz:
                beste_distanz = distanz
                bestes_label  = int(y_geladen[j])

        wahre_labels.append(wahres_label)
        vorhergesagte_labels.append(bestes_label)

    wahre_labels         = np.array(wahre_labels)
    vorhergesagte_labels = np.array(vorhergesagte_labels)

    accuracy = berechne_accuracy(wahre_labels, vorhergesagte_labels)
    cm       = berechne_confusion_matrix(wahre_labels, vorhergesagte_labels, int(max(wahre_labels)) + 1)

    print(f"  Accuracy       : {accuracy * 100:.2f}%")
    print(f"  Confusion Matrix ({cm.shape[0]}x{cm.shape[1]}):")
    print(f"  Hauptdiagonale : {np.diag(cm).tolist()}")

    # --- Schritt 5: Confusion Matrix visualisieren ---
    print("\n[Schritt 5] Zeige Confusion Matrix ...\n")
    visualisiere_confusion_matrix(cm, map_geladen, accuracy)

    # --- Schritt 6: Live-Demo mit Testbild ---
    print("\n[Schritt 6] Live-Demo: Erkenne Person aus Testbild ...\n")

    if not os.path.exists(TESTBILD_PFAD):
        print(f"  [HINWEIS] Kein Testbild gefunden: {TESTBILD_PFAD}")
        print(f"  Lege 'testbild.png' in den Projektordner und starte neu.")
    else:
        label, name, distanz = erkenne_person(
            TESTBILD_PFAD, X_geladen, y_geladen, map_geladen, metrik="chi"
        )

        if name is not None:
            print(f"  Testbild            : {TESTBILD_PFAD}")
            print(f"  Erkannte Person     : {name}  (Label: {label})")
            print(f"  Chi-Quadrat-Distanz : {distanz:.6f}")

            # Testbild und erkanntes Datenbankbild nebeneinander anzeigen
            testbild_anzeige  = cv2.imread(TESTBILD_PFAD, cv2.IMREAD_GRAYSCALE)
            datenbank_bild    = cv2.imread(
                os.path.join(DATENBANK_DIR, name, "img1.jpg"), cv2.IMREAD_GRAYSCALE
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

            ax1.imshow(testbild_anzeige, cmap="gray")
            ax1.set_title("Testbild")
            ax1.axis("off")

            ax2.imshow(datenbank_bild, cmap="gray")
            ax2.set_title(f"Beste Übereinstimmung:\n{name}  (Distanz: {distanz:.4f})")
            ax2.axis("off")

            plt.tight_layout()
            plt.show()

    print("\n" + "=" * 60)
    print("  Fertig!")
    print("=" * 60)
