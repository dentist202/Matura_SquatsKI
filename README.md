# Squats mit KI – Anleitung

Die App bewertet Kniebeugen mit einer leeren Langhantel von der Seite.
Sie läuft nur unter **Windows** (der Ton kommt von `winsound`).

## 1. Python installieren
Python **3.11** (getestet mit 3.11.9).

```
py -3.11 --version
```

## 2. Bibliotheken installieren
Im Ordner dieser Datei:

```
py -3.11 -m pip install -r requirements.txt
```

Getestete Versionen: mediapipe 0.10.35, opencv-python 4.13.0.92, numpy 2.4.4, pandas 3.0.5, matplotlib 3.10.9.

## 3. Programm starten

```
py -3.11 main.py
```

Wichtig: Das Programm im Ordner dieser Datei starten. Die Datei `pose_landmarker_full.task` (Modell von MediaPipe) und der Ordner `video` müssen daneben liegen.

## 4. Video oder Webcam
Standardmässig läuft die App mit einem Video. In `main.py` (Zeilen 96 und 97) wählt man die Quelle:

```
cap = cv2.VideoCapture("video/gym2_sp.mp4")          # Video
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)           # Webcam
```

Für die Webcam die erste Zeile mit `#` ausschalten und bei der zweiten das `#` entfernen. Zusätzlich die Zeitangabe (timestamp) wie im Kommentar am Anfang von `main.py` beschrieben umstellen.
Falls die Webcam beim ersten Bild noch kein Bild liefert, endet das Programm sofort. Dann einfach noch einmal starten.

## 5. Videos im Ordner `video`
| Datei | Inhalt |
|---|---|
| richtig.mp4 | richtige Kniebeugen |
| gym2_sp.mp4 | Kniebeugen im Fitnessstudio |
| fehler_zu_wenig_tief.mp4 | Fehler: zu wenig tief («Go deeper») |
| fehler_oberkoerper_vorne.mp4 | Fehler: Oberkörper zu weit vorne («Chest up») |
| fehler_fersen.mp4 | Fehler: Fersen heben ab («Keep heels down») |
| fehler_huefte_zuerst.mp4 | Fehler: Hüfte kommt zuerst hoch («Hips too fast») |
| fehler_gewicht_zehen.mp4 | Fehler: Gewicht auf den Zehen («Weight on toes») |

## 6. Ablauf
1. Startbildschirm: Regeln lesen und auf START klicken.
2. In den grünen Rahmen stellen.
3. Die rechte Seite zur Kamera drehen und 3 s stehen bleiben.
4. Kniebeugen machen. Nach jeder Kniebeuge zeigt das Panel eine Farbe und einen Hinweis, bei Rot ertönt ein Ton.
5. Den Rahmen länger als 2 s verlassen: Die Resultate erscheinen.

Die Taste `q` beendet das Programm. Die Werte jedes Bildes werden in `results.csv` gespeichert.

## 7. test.py
Hilfsprogramm für die Auswertung der gespeicherten CSV-Dateien (Diagramme, Slider, Skelett). Es gehört nicht zur App.
