#!/bin/sh
set -e

echo "[startup] SEED_ON_START=${SEED_ON_START:-true}"
echo "[startup] TRAIN_ON_START=${TRAIN_ON_START:-false}"

if [ "${SEED_ON_START:-true}" = "true" ]; then
  echo "[startup] Seeding parcels (idempotent)..."
  # Run the seeding script; ignore errors so service still starts
  python -m scripts.load_parcels || echo "[startup] load_parcels failed or already seeded"
fi

if [ "${TRAIN_ON_START:-false}" = "true" ]; then
  if [ ! -f "/app/models/_global.joblib" ]; then
    echo "[startup] Training models (this can take several minutes)..."
    # If training fails, do not crash the container; log and proceed so cached APIs still work
    python -m scripts.refresh_all || echo "[startup] refresh_all failed; live diagnostics will return 503 until models exist"
  else
    echo "[startup] Global model already present; skipping training"
  fi
fi

echo "[startup] Launching API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
