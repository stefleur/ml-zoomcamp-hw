# ML Zoomcamp 2026 — Hausaufgabe 6

Lösungen für [Homework 6](homework.md) mit dem [Kursdatensatz](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv). Zielvariable: `fuel_efficiency_mpg`.

| Frage | Antwort | Messwert |
| --- | --- | --- |
| 1 | `model_year` | erstes Split-Merkmal des Baums |
| 2 | **1,837** | RMSE = 1,837054 (`scikit-learn` 1.7.2) |
| 3 | **100** | RMSE = 1,768136; 150 Bäume: 1,768804 |
| 4 | **10** | mittlerer RMSE = 1,755405 |
| 5 | `vehicle_weight` | Importance = 0,203879 unter den vier Optionen |
| 6 | **0,1** | RMSE = 1,724810; `eta=0,3`: 1,826458 |

Die Daten werden wie vorgegeben zuerst in 80 % Training und 20 % Test und dann in 75 % Training und 25 % Validierung innerhalb der 80 % aufgeteilt. Fehlende Werte werden durch 0 ersetzt. Die Merkmale werden mit `DictVectorizer(sparse=True)` kodiert. Der Testsatz bleibt unberührt.

Für Frage 3 sind die RMSE-Werte bei 10, 50, 100 und 150 Bäumen **1,837054**, **1,771567**, **1,768136** und **1,768804**. Damit ist 100 auch nach Rundung auf drei Dezimalstellen die beste Wahl.

Für Frage 4 sind die Mittelwerte über diese vier Baumzahlen bei maximaler Tiefe 10, 15, 20 und 25 entsprechend **1,755405**, **1,783953**, **1,788242** und **1,786317**.

## Reproduzieren

Mit Python 3.11 bis 3.13:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python solve_homework.py
```

Die `scikit-learn`-Version ist für die numerischen Ergebnisse wichtig: Version 1.9.1 ergab für Frage 2 auf denselben Daten 1,833220 und für Frage 3 eine andere Reihenfolge der beiden besten Werte. Die Antworten oben stammen aus Version 1.7.2. Die XGBoost-Werte wurden mit Version 3.4.1 berechnet.
