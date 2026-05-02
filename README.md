# EZZAYRA — Détection précoce d'anomalies sur oliveraies

Système de détection d'anomalies non-supervisé pour les oliveraies tunisiennes : NDVI Sentinel-2 vs météo locale + LST MODIS, avec dashboard cartographique Leaflet et API de diagnostic.

## Architecture

Modèle deux-étages :

- **Tier 1 (per-parcelle, cache)** — pour chacune des 49 parcelles EZZAYRA, un modèle Ridge personnel entraîné sur 5 ans d'historique. Réponses instantanées pour le dashboard.
- **Tier 2 (global, fallback)** — un modèle Ridge poolé sur les 49 parcelles (avec la variable `system` en feature). Sert à diagnostiquer toute oliveraie inconnue que le jury fournit.

Pipeline de prédiction : `phenology(date) + weather(rain, temp, GDD) + LST_anomaly` → NDVI attendu → résiduel → seuils par quantiles mensuels → vert/orange/rouge.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Un fichier de service-account Earth Engine est attendu (par défaut dans `~/Downloads/anomalydetection-*.json`). Surchargeable via `EE_SERVICE_ACCOUNT_KEY` et `EE_PROJECT_ID`.

## Préparation des données

```bash
# 1. Vérifier la connexion GEE
python test_ee.py

# 2. Charger les 49 parcelles dans SQLite
python -m scripts.load_parcels

# 3. Entraîner tous les modèles (5 ans d'historique chacun, ~25-40 min)
python -m scripts.refresh_all
```

Le batch `refresh_all` est résumable : il saute les parcelles déjà entraînées sauf si `--force` est passé.

## Lancer le serveur

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

- Dashboard : http://127.0.0.1:8765/
- Docs API automatiques : http://127.0.0.1:8765/docs

## Endpoints

| Méthode | URL | Description |
|---|---|---|
| GET | `/` | Dashboard cartographique Leaflet |
| GET | `/api/health` | Statut système (GEE, modèles, parcelles) |
| GET | `/api/parcels` | Liste des parcelles + statut cache |
| GET | `/api/parcels/{id}` | Détail d'une parcelle |
| GET | `/api/diagnostic/cached/{id}` | Diagnostic en cache (instantané) |
| **POST** | **`/api/diagnostic-anomalie`** | **Diagnostic live (endpoint jury)** |

## Endpoint principal — exemple

```bash
curl -X POST http://127.0.0.1:8765/api/diagnostic-anomalie \
  -H "Content-Type: application/json" \
  -d '{
    "oliveraie": {
      "id": "O_2026_307",
      "polygone": {
        "type": "Polygon",
        "coordinates": [[[9.20,35.20],[9.22,35.20],[9.22,35.22],[9.20,35.22],[9.20,35.20]]]
      },
      "systeme": "intensif"
    },
    "date": "2024-09-30"
  }'
```

Réponse :

```json
{
  "statut": "orange",
  "anomaly_score": 1.84,
  "ndvi_observe": [0.42, 0.45, 0.41, 0.37, 0.34],
  "ndvi_attendu": [0.48, 0.50, 0.52, 0.51, 0.50],
  "explication": "NDVI 12% sous l'attendu sur les 3 dernières semaines (pluie 90j déficitaire). Cause probable : déficit pluviométrique cumulé. À surveiller (orange).",
  "recommandation": "Inspection visuelle dans 48h et vérification de l'irrigation."
}
```

Le format d'entrée accepté est flexible : `polygone` (GeoJSON, format brief), `polygon`, `geometry`, ou `coordinates: [{lat,lng}]` (format EZZAYRA).

## Démo live — dessiner une nouvelle oliveraie sur la carte

Le dashboard inclut un bouton **"Tester une nouvelle oliveraie"** qui ouvre un outil de dessin Leaflet. Le polygone tracé est envoyé en POST à l'API, le résultat est affiché en popup et dans le panneau de détail, et la nouvelle parcelle apparaît sur la carte colorée selon son statut.

## Tests

```bash
python -m pytest tests/ -v
```

## Démo de stress synthétique

Pour montrer au jury que le système détecte un vrai stress même quand aucune parcelle réelle n'est en alerte aujourd'hui :

```bash
python -m scripts.inject_synthetic_stress --drop 0.30
```

## Postman

Collection prête dans `postman/EZZAYRA_anomaly.postman_collection.json` avec 6 cas de test.

## Pièges traités

| Piège du brief | Mitigation |
|---|---|
| Phénologie ignorée | Features sin/cos du jour-de-l'année |
| Bruit pixel | Buffer négatif 10m + moyenne sur la parcelle |
| Faux positifs après taille | Seuils `q66/q90` calculés par mois (les mois de taille s'élargissent automatiquement) |
| Différence E/I/HI | Modèle per-parcelle pour Tier 1 + feature `system` dans le modèle global |

## Structure du projet

```
.
├── app/
│   ├── main.py            FastAPI (5 endpoints)
│   ├── normalize.py       parser polygones (GeoJSON | EZZAYRA latlng)
│   ├── gee_client.py      Sentinel-2 NDVI + MODIS LST
│   ├── weather.py         Open-Meteo Archive + GDD/rain rolling
│   ├── features.py        ingénierie features (phénologie, météo, LST)
│   ├── train.py           Ridge per-parcelle + global pooled
│   ├── diagnose.py        routeur Tier 1/Tier 2, scoring, statut
│   ├── explain.py         explication FR + recommandation
│   ├── geocode.py         centroïde → gouvernorat tunisien
│   ├── db.py              SQLite (parcelles + statuts)
│   └── static/            dashboard Leaflet
├── scripts/
│   ├── load_parcels.py    EZZAYRA JSON → SQLite
│   ├── refresh_all.py     batch nightly (5y fetch + train + diagnose all)
│   ├── seed_dev.py        smoke test 1 parcelle
│   ├── train_one.py       pipeline complet sur 1 parcelle
│   └── inject_synthetic_stress.py  démo stress reproductible
├── tests/                 pytest
├── data/                  EZZAYRA JSON + SQLite
├── models/                modèles entraînés (.joblib)
├── history/               cache 5y NDVI+météo+LST par parcelle (.parquet)
└── postman/               collection de tests
```
