# DABI-2 – Data Science Group Projects

Gruppenprojekte aus dem 3. Semester des Studiengangs Data Science an der [Hochschule Karlsruhe (HKA)](https://www.h-ka.de/), Sommersemester 2026: Zeitreihenanalyse und Prognose, Machine Learning für die Vorhersage von Hotelstornierungen sowie Umwelt- und Geodatenanalyse.

*English version below.*

---

## Inhalt

- [Überblick](#überblick)
- [Projekte](#projekte)
  - [Datenanalyse 2 – Zeitreihenanalyse und Prognose](#datenanalyse-2--zeitreihenanalyse-und-prognose)
  - [Maschinelles Lernen 1 – Vorhersage von Hotelstornierungen](#maschinelles-lernen-1--vorhersage-von-hotelstornierungen)
  - [Umweltmonitoring – Klima-, CO₂- und Geodaten](#umweltmonitoring--klima--co₂--und-geodaten)
- [Tech Stack](#tech-stack)
- [Projektstruktur](#projektstruktur)
- [Ausführen](#ausführen)
- [Hintergrund](#hintergrund)
- [Autoren](#autoren)

## Überblick

Das Repository enthält die Gruppenabgaben aus drei Modulen. Jede Abgabe besteht aus einem dokumentierten Jupyter Notebook (bzw. einem QGIS-Projekt), den verwendeten Daten und, sofern vorhanden, der Aufgabenstellung. Die Notebooks erklären Vorgehen, Entscheidungen und Ergebnisse in Markdown-Zellen; die Interpretationen sind farblich hervorgehoben.

Schwerpunkte:

- **Zeitreihenanalyse:** Zerlegung in Trend, Saisonalität und Residuum (manuell, STL, MSTL), ACF/PACF, lineare Regression und ARIMA.
- **Machine Learning:** vollständiger Workflow von der explorativen Analyse über Feature Engineering und scikit-learn-Pipelines bis zur Hyperparameter-Optimierung und kostensensitiven Schwellwertwahl mit einem gemischt-ganzzahligen Optimierungsmodell (Gurobi).
- **Umweltdaten:** Auswertung von DWD-Klimadaten und der CO₂-Messreihe vom Mauna Loa, einfache Prognoseverfahren und eine selbst digitalisierte Karte in QGIS.

## Projekte

### Datenanalyse & Business Intelligence 2 – Zeitreihenanalyse und Prognose

Ordner: [`Datenanalyse 2/`](Datenanalyse%202/)

| Abgabe | Daten | Inhalt |
|---|---|---|
| [Abgabe 1](Datenanalyse%202/Abgabe%201/analyse.ipynb) | Stromverbrauch eines Energieversorgers in Halbstunden-Intervallen (~80.000 Messungen), getrennt nach Standard- und zeitvariablem Tarif | Datenbereinigung, Zerlegung in Trend, Tages- und Jahressaisonalität (manuell und mit STL), ACF/PACF, lineares Prognosemodell für einen Horizont von 6 Stunden, Varianzzerlegung und Vergleich des Verbrauchsverhaltens zwischen den Tarifen |
| [Abgabe 2](Datenanalyse%202/Abgabe%202/analyse.ipynb) | FRED-Zeitreihe [IPG2211A2N](https://fred.stlouisfed.org/series/IPG2211A2N) (Industrieproduktion Strom- und Gasversorgung, monatlich seit 1939) | Erkennen multiplikativer Saisonalität, Log-Transformation, Bereinigung von Trend und Saisonalität, Prüfung auf Stationarität und Normalverteilung, Bestimmung der ARIMA-Ordnung über ACF/PACF und Ein-Schritt-Prognose |
| [Abgabe 3](Datenanalyse%202/Abgabe%203/analyse.ipynb) | Datensatz aus Abgabe 1 | Vertiefung von Abgabe 1: Ursache der negativen Autokorrelation der Residuen bei Lag 12–15, MSTL mit Tages- und Wochenperiode, verbesserte manuelle Zerlegung (rollierender Trend, Wochen- und Jahressaisonalität) und Analyse der Streuung der Prognosewerte |

**Ergebnisse:**

- Trend und Periodizität erklären rund 97 % der Varianz des Stromverbrauchs; auf das Residuum entfallen etwa 2 %.
- Durch die verbesserte Zerlegung in Abgabe 3 steigt das R² der 6-Stunden-Prognose des Residuums von 0,79 auf 0,93.
- Haushalte mit zeitvariablem Tarif zeigen eine stärker ausgeprägte Tagesperiodizität, was auf ein bewussteres zeitliches Verbrauchsverhalten hindeutet.
- Das ARIMA(1,0,2)-Modell erreicht auf der Originalskala ein R² von 0,998 bei der Ein-Schritt-Prognose.

### Maschinelles Lernen 1 – Vorhersage von Hotelstornierungen

Ordner: [`Maschinelles Lernen 1/`](Maschinelles%20Lernen%201/) · Notebook: [`project_submission.ipynb`](Maschinelles%20Lernen%201/project_submission.ipynb) · [Präsentation (PDF)](Maschinelles%20Lernen%201/Pr%C3%A4sentation.pdf)

Semesterprojekt im Modul *Data Mining & Grundlagen Maschinelles Lernen 1*: Ein Klassifikationsmodell sagt bei Buchungseingang voraus, ob eine Buchung storniert wird. Datengrundlage sind 95.512 Buchungen zweier portugiesischer Hotels aus den Jahren 2015 bis 2017.

- **Explorative Analyse:** Stornierungsraten nach Hoteltyp, Vorlaufzeit und Anzahlungsart, Verteilungen und Ausreißer, Korrelationsanalyse.
- **Vorverarbeitung:** `ColumnTransformer` und `Pipeline` mit Imputation, Kappung von Ausreißern am 99-%-Quantil, Skalierung und One-Hot-Encoding.
- **Feature Engineering:** Risikogruppen für Herkunftsländer und Agenturen auf Basis ihrer Stornierungsraten, Anzahl der Feiertage im Aufenthaltszeitraum (Verknüpfung mit einer Feiertagstabelle), geglättete Stornierungsrate früherer Buchungen, kurzfristige Buchung und Zimmeränderung.
- **Modelle:** Logistische Regression, Entscheidungsbaum, Random Forest, Gradient Boosting und Stacking (Entscheidungsbaum, Random Forest und Gradient Boosting mit logistischer Regression als Meta-Modell), jeweils auf allen Daten und getrennt nach Hoteltyp trainiert. Hyperparameter werden per `GridSearchCV` mit Kreuzvalidierung optimiert; Feature und Permutation Importances dienen der Feature-Auswahl.
- **Kostenoptimierung:** Falsch-positive Vorhersagen (Umbuchung des Gastes, 250 €) und falsch-negative (entgangener Umsatz) verursachen unterschiedliche Kosten. Der Klassifikationsschwellwert wird deshalb über ein gemischt-ganzzahliges lineares Programm mit Gurobi so gewählt, dass die Gesamtkosten minimal sind.

**Ergebnis:** Der nach Hoteltyp getrennt trainierte Random Forest erzielt auf den Validierungsdaten die höchste Accuracy (87,95 %) und bei optimiertem Schwellwert die geringsten Gesamtkosten. Mit ihm wurden die Vorhersagen für die 23.878 Testbuchungen erstellt ([`data/vorhersage.csv`](Maschinelles%20Lernen%201/data/vorhersage.csv)).

| Modell | Accuracy | Precision | Recall | False Positive Rate |
|---|---|---|---|---|
| Random Forest (getrennt) | 87,95 % | 86,03 % | 80,36 % | 7,62 % |
| Stacking (getrennt) | 87,81 % | 85,52 % | 80,59 % | 7,97 % |
| Gradient Boosting (getrennt) | 87,02 % | 84,16 % | 79,82 % | 8,77 % |
| Entscheidungsbaum | 85,82 % | 83,21 % | 77,11 % | 9,09 % |
| Logistische Regression (getrennt) | 83,16 % | 81,05 % | 70,89 % | 9,68 % |

*Werte auf dem Validierungsdatensatz, je Modelltyp die bessere Variante.*

### Umweltmonitoring – Klima-, CO₂- und Geodaten

Ordner: [`Umweltmonitoring/`](Umweltmonitoring/)

| Arbeitsblatt | Daten | Inhalt |
|---|---|---|
| [Arbeitsblatt 1](Umweltmonitoring/Arbeitsblatt%201/) | DWD-Messdaten der Station Freiburg (Temperatur und Niederschlag) | Monatliche Mittel-, Höchst- und Tiefsttemperaturen und Niederschläge, Ermittlung der heißesten Sommer, kältesten Winter und niederschlagsreichsten Zeiträume, Abweichungen vom Referenzzeitraum 1980–2000 sowie Einordnung der deutschen Klimaschutzziele. Ergebnisse in [`Abgabe.pdf`](Umweltmonitoring/Arbeitsblatt%201/Abgabe.pdf). |
| [Arbeitsblatt 3](Umweltmonitoring/Arbeitsblatt%203/analyse.ipynb) | NOAA-CO₂-Messungen (Flask Samples) vom Mauna Loa, 1969–2024; DWD-Niederschlag Freiburg | Behandlung fehlender Werte, Kennzahlen pro Jahrzehnt, jährliche Wachstumsraten (CAGR) und Test auf exponentielles Wachstum; Zerlegung und Autokorrelation der Niederschlagsreihe sowie Vergleich von Mittelwert-, naiver, saisonal naiver Prognose und exponentieller Glättung für Horizonte von 1 bis 12 Monaten (MAE, MAPE) |
| [Arbeitsblatt 4](Umweltmonitoring/Arbeitsblatt%204/) | OpenStreetMap als Hintergrundkarte | Eigene Geodatenbank (GeoPackage) in QGIS mit Punkt-, Linien- und Flächen-Layern: digitalisierte Karte der Parkmöglichkeiten am Universitätsklinikum Heidelberg, attribuiert und topologisch bereinigt. Exportierte Karte als [PDF](Umweltmonitoring/Arbeitsblatt%204/Universit%C3%A4tsklinikum%20Heidelberg%2C%20Parkm%C3%B6glichkeiten.pdf). |

## Tech Stack

| Bereich | Werkzeuge |
|---|---|
| Sprache und Umgebung | Python 3, Jupyter Notebook |
| Datenverarbeitung | pandas, NumPy, SciPy |
| Zeitreihen | statsmodels (STL, MSTL, ARIMA, Holt-Winters, ACF/PACF) |
| Machine Learning | scikit-learn (Pipelines, ColumnTransformer, GridSearchCV, Ensemble-Methoden) |
| Optimierung | Gurobi (`gurobipy`) |
| Visualisierung | Matplotlib, Seaborn |
| Geodaten | QGIS, GeoPackage |

## Projektstruktur

```
.
├── Datenanalyse 2/
│   ├── Abgabe 1/              # Stromverbrauch: Zerlegung, ACF/PACF, lineare Prognose, Tarifvergleich
│   ├── Abgabe 2/              # FRED-Zeitreihe: Log-Transformation und ARIMA
│   └── Abgabe 3/              # Vertiefung von Abgabe 1: MSTL, verbesserte Zerlegung
├── Maschinelles Lernen 1/
│   ├── data/                  # Trainings-, Test- und Feiertagsdaten, abgegebene Vorhersagen
│   ├── project_submission.ipynb
│   └── Präsentation.pdf/.pptx
└── Umweltmonitoring/
    ├── Arbeitsblatt 1/        # DWD-Klimadaten Freiburg
    ├── Arbeitsblatt 3/        # CO₂ Mauna Loa, Niederschlagsprognose
    └── Arbeitsblatt 4/        # QGIS-Projekt und GeoPackage
```

## Ausführen

Die Notebooks lesen ihre Daten über relative Pfade und werden daher aus ihrem jeweiligen Ordner gestartet.

```bash
pip install jupyter pandas numpy scipy statsmodels scikit-learn matplotlib seaborn gurobipy
```

```bash
jupyter notebook
```

Hinweis: Die Schwellwertoptimierung im ML-Projekt verwendet Gurobi. Für die Größe des Modells reicht die mit `gurobipy` ausgelieferte eingeschränkte Lizenz nicht aus; benötigt wird eine (z. B. akademische) Vollversion. Das QGIS-Projekt in Arbeitsblatt 4 lässt sich mit QGIS ab Version 3.22 öffnen.

## Hintergrund

Die Projekte sind im Sommersemester 2026 als Gruppenarbeiten in den Modulen *Datenanalyse & Business Intelligence 2*, *Data Mining & Grundlagen Maschinelles Lernen 1* und *Umweltmonitoring* an der Hochschule Karlsruhe entstanden. Die Aufgabenstellungen stammen von den jeweiligen Lehrenden und liegen den Abgaben zur Einordnung bei.

## Autoren

- Patrick Heß
- Sven van Tongeren
- Sebastian Straßner
- Daniel Daub

---

<br>

# English version

Group projects from the 3rd semester of the Data Science program at [Hochschule Karlsruhe (HKA)](https://www.h-ka.de/), summer semester 2026: time-series analysis and forecasting, machine learning for predicting hotel booking cancellations, and environmental and geospatial data analysis.

## Contents

- [Overview](#overview)
- [Projects](#projects)
  - [Data Analysis 2 – Time-Series Analysis and Forecasting](#data-analysis-2--time-series-analysis-and-forecasting)
  - [Machine Learning 1 – Predicting Hotel Cancellations](#machine-learning-1--predicting-hotel-cancellations)
  - [Environmental Monitoring – Climate, CO₂ and Geospatial Data](#environmental-monitoring--climate-co₂-and-geospatial-data)
- [Tech Stack](#tech-stack-1)
- [Project Structure](#project-structure)
- [Running the Notebooks](#running-the-notebooks)
- [Background](#background)
- [Authors](#authors)

## Overview

This repository contains the group submissions from three modules. Each submission consists of a documented Jupyter notebook (or a QGIS project), the data it uses and, where available, the assignment. The notebooks explain the approach, decisions and results in Markdown cells, with interpretations highlighted in color. The notebooks and the assignments are written in German.

Focus areas:

- **Time-series analysis:** decomposition into trend, seasonality and residual (manual, STL, MSTL), ACF/PACF, linear regression and ARIMA.
- **Machine learning:** the full workflow from exploratory analysis through feature engineering and scikit-learn pipelines to hyperparameter tuning and a cost-sensitive choice of the decision threshold using a mixed-integer optimization model (Gurobi).
- **Environmental data:** analysis of German Weather Service (DWD) climate data and the Mauna Loa CO₂ record, baseline forecasting methods, and a hand-digitized map in QGIS.

## Projects

### Data Analysis  & Business Intelligence 2 – Time-Series Analysis and Forecasting

Folder: [`Datenanalyse 2/`](Datenanalyse%202/)

| Submission | Data | Content |
|---|---|---|
| [Abgabe 1](Datenanalyse%202/Abgabe%201/analyse.ipynb) | Electricity consumption of a utility's customers in half-hour intervals (~80,000 readings), split by standard and time-of-use tariff | Data cleaning, decomposition into trend, daily and yearly seasonality (manual and STL), ACF/PACF, a linear model forecasting 6 hours ahead, variance decomposition, and a comparison of consumption behavior between the tariffs |
| [Abgabe 2](Datenanalyse%202/Abgabe%202/analyse.ipynb) | FRED series [IPG2211A2N](https://fred.stlouisfed.org/series/IPG2211A2N) (industrial production of electric and gas utilities, monthly since 1939) | Detecting multiplicative seasonality, log transformation, removing trend and seasonality, checks for stationarity and normality, choosing the ARIMA order from ACF/PACF, and one-step-ahead forecasting |
| [Abgabe 3](Datenanalyse%202/Abgabe%203/analyse.ipynb) | Dataset from Abgabe 1 | Follow-up to Abgabe 1: the cause of the negative residual autocorrelation at lags 12–15, MSTL with daily and weekly periods, an improved manual decomposition (rolling trend, weekly and yearly seasonality), and an analysis of the spread of the forecasts |

**Results:**

- Trend and periodicity explain about 97 % of the variance in electricity consumption; the residual accounts for about 2 %.
- The improved decomposition in Abgabe 3 raises the R² of the 6-hour residual forecast from 0.79 to 0.93.
- Households on the time-of-use tariff show a more pronounced daily periodicity, which suggests they pay more attention to when they use electricity.
- The ARIMA(1,0,2) model reaches an R² of 0.998 on the original scale for one-step-ahead forecasts.

### Machine Learning 1 – Predicting Hotel Cancellations

Folder: [`Maschinelles Lernen 1/`](Maschinelles%20Lernen%201/) · Notebook: [`project_submission.ipynb`](Maschinelles%20Lernen%201/project_submission.ipynb) · [Presentation (PDF)](Maschinelles%20Lernen%201/Pr%C3%A4sentation.pdf)

Semester project for the module *Data Mining & Fundamentals of Machine Learning 1*: a classifier predicts at booking time whether a booking will be cancelled. The data covers 95,512 bookings at two Portuguese hotels from 2015 to 2017.

- **Exploratory analysis:** cancellation rates by hotel type, lead time and deposit type, distributions and outliers, correlation analysis.
- **Preprocessing:** `ColumnTransformer` and `Pipeline` with imputation, outlier capping at the 99th percentile, scaling and one-hot encoding.
- **Feature engineering:** risk groups for countries of origin and travel agencies based on their cancellation rates, number of public holidays during the stay (joined from a holiday table), a smoothed cancellation rate of the guest's previous bookings, a short-notice booking flag, and a room-change flag.
- **Models:** logistic regression, decision tree, random forest, gradient boosting and stacking (decision tree, random forest and gradient boosting with logistic regression as meta-model), each trained on all data and separately per hotel type. Hyperparameters are tuned with `GridSearchCV` and cross-validation; feature and permutation importances guide feature selection.
- **Cost optimization:** false positives (relocating a guest, €250) and false negatives (lost revenue) have different costs. The decision threshold is therefore chosen with a mixed-integer linear program in Gurobi that minimizes total cost.

**Result:** the random forest trained separately per hotel type achieves the highest validation accuracy (87.95 %) and, with an optimized threshold, the lowest total cost. It produced the predictions for the 23,878 test bookings ([`data/vorhersage.csv`](Maschinelles%20Lernen%201/data/vorhersage.csv)).

| Model | Accuracy | Precision | Recall | False Positive Rate |
|---|---|---|---|---|
| Random Forest (per hotel) | 87.95 % | 86.03 % | 80.36 % | 7.62 % |
| Stacking (per hotel) | 87.81 % | 85.52 % | 80.59 % | 7.97 % |
| Gradient Boosting (per hotel) | 87.02 % | 84.16 % | 79.82 % | 8.77 % |
| Decision Tree | 85.82 % | 83.21 % | 77.11 % | 9.09 % |
| Logistic Regression (per hotel) | 83.16 % | 81.05 % | 70.89 % | 9.68 % |

*Validation set scores; for each model type, the better of the two variants.*

### Environmental Monitoring – Climate, CO₂ and Geospatial Data

Folder: [`Umweltmonitoring/`](Umweltmonitoring/)

| Worksheet | Data | Content |
|---|---|---|
| [Arbeitsblatt 1](Umweltmonitoring/Arbeitsblatt%201/) | DWD measurements for the Freiburg station (temperature and precipitation) | Monthly mean, maximum and minimum temperatures and precipitation, the hottest summers, coldest winters and wettest periods, deviations from the 1980–2000 reference period, and a discussion of Germany's climate targets. Results in [`Abgabe.pdf`](Umweltmonitoring/Arbeitsblatt%201/Abgabe.pdf). |
| [Arbeitsblatt 3](Umweltmonitoring/Arbeitsblatt%203/analyse.ipynb) | NOAA CO₂ flask samples from Mauna Loa, 1969–2024; DWD precipitation for Freiburg | Handling missing values, statistics per decade, compound annual growth rates and a test for exponential growth; decomposition and autocorrelation of the precipitation series, and a comparison of mean, naive, seasonal naive and exponential smoothing forecasts for horizons of 1 to 12 months (MAE, MAPE) |
| [Arbeitsblatt 4](Umweltmonitoring/Arbeitsblatt%204/) | OpenStreetMap as base map | A custom geodatabase (GeoPackage) in QGIS with point, line and polygon layers: a hand-digitized map of parking options at Heidelberg University Hospital, with attributes and clean topology. Exported map as [PDF](Umweltmonitoring/Arbeitsblatt%204/Universit%C3%A4tsklinikum%20Heidelberg%2C%20Parkm%C3%B6glichkeiten.pdf). |

## Tech Stack

| Area | Tools |
|---|---|
| Language and environment | Python 3, Jupyter Notebook |
| Data processing | pandas, NumPy, SciPy |
| Time series | statsmodels (STL, MSTL, ARIMA, Holt-Winters, ACF/PACF) |
| Machine learning | scikit-learn (pipelines, ColumnTransformer, GridSearchCV, ensemble methods) |
| Optimization | Gurobi (`gurobipy`) |
| Visualization | Matplotlib, Seaborn |
| Geospatial | QGIS, GeoPackage |

## Project Structure

```
.
├── Datenanalyse 2/
│   ├── Abgabe 1/              # Electricity consumption: decomposition, ACF/PACF, linear forecast, tariff comparison
│   ├── Abgabe 2/              # FRED series: log transformation and ARIMA
│   └── Abgabe 3/              # Follow-up to Abgabe 1: MSTL, improved decomposition
├── Maschinelles Lernen 1/
│   ├── data/                  # Training, test and holiday data, submitted predictions
│   ├── project_submission.ipynb
│   └── Präsentation.pdf/.pptx
└── Umweltmonitoring/
    ├── Arbeitsblatt 1/        # DWD climate data for Freiburg
    ├── Arbeitsblatt 3/        # Mauna Loa CO₂, precipitation forecasting
    └── Arbeitsblatt 4/        # QGIS project and GeoPackage
```

## Running the Notebooks

The notebooks load their data through relative paths, so start them from their own folder.

```bash
pip install jupyter pandas numpy scipy statsmodels scikit-learn matplotlib seaborn gurobipy
```

```bash
jupyter notebook
```

Note: the threshold optimization in the machine learning project uses Gurobi. The size-limited license bundled with `gurobipy` is not enough for this model; a full (for example academic) license is required. The QGIS project in Arbeitsblatt 4 opens in QGIS 3.22 or later.

## Background

These projects were completed as group work in the summer semester 2026 in the modules *Datenanalyse & Business Intelligence 2* (Data Analysis 2), *Data Mining & Grundlagen Maschinelles Lernen 1* (Data Mining & Fundamentals of Machine Learning 1) and *Umweltmonitoring* (Environmental Monitoring) at Hochschule Karlsruhe. The assignments were written by the respective lecturers and are included with the submissions for context.

## Authors

- Patrick Heß
- Sven van Tongeren
- Sebastian Straßner
- Daniel Daub
