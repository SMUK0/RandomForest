# ml/train_random_forest_citas.py
# Entrena un Random Forest con búsqueda de hiperparámetros y calibración
# Guarda el modelo (calibrado) en ml_artifacts/random_forest_citas_v1.joblib
# No usa if _name_ == "_main_": se ejecuta directo al correr el archivo.

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Detectar versión de scikit-learn de forma robusta (para metadata)
try:
    import sklearn  # noqa: F401
    SKLEARN_VERSION = getattr(sklearn, "_version_", "unknown")
except Exception:
    SKLEARN_VERSION = "unknown"

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# ===========================
# 1) Datos (sintéticos de ejemplo)
#    ⚠ Cuando tengas históricos reales, reemplaza este bloque por carga desde CSV/BD
#    y arma X,y con las MISMAS columnas de FEATURES.
# ===========================
rng = np.random.default_rng(13)
N = 5000  # tamaño del dataset simulado

data = pd.DataFrame({
    "dia_semana": rng.integers(0, 7, N),
    "hora": rng.integers(8, 21, N),
    "prioridad_numeric": rng.choice([5, 4, 3, 2, 1, 0], size=N,
                                    p=[0.10, 0.15, 0.25, 0.25, 0.15, 0.10]),
    "edad": rng.integers(14, 80, N),
    "dias_desde_ultima_sesion": rng.integers(0, 60, N),
    "match_disponibilidad": rng.integers(0, 2, N),
    "ocupacion_slot_psico": rng.integers(0, 2, N),
    "paciente_prefiere_tarde": rng.integers(0, 2, N),
})

# Etiquetas con probabilidad "realista"
logit = (
    1.2 * data["match_disponibilidad"]
    + 0.6 * (data["prioridad_numeric"] / 5)
    + 0.4 * (data["hora"].between(14, 18)).astype(int)
    - 0.8 * data["ocupacion_slot_psico"]
    + 0.3 * data["paciente_prefiere_tarde"]
)
prob = 1 / (1 + np.exp(-logit))
data["label"] = (rng.random(N) < prob).astype(int)

FEATURES = [
    "dia_semana","hora","prioridad_numeric","edad",
    "dias_desde_ultima_sesion","match_disponibilidad",
    "ocupacion_slot_psico","paciente_prefiere_tarde"
]

X = data[FEATURES]
y = data["label"]

# Split para evaluación final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===========================
# 2) Búsqueda de hiperparámetros (RandomizedSearchCV)
# ===========================
base_rf = RandomForestClassifier(
    n_estimators=600,          # más árboles mejora estabilidad
    bootstrap=True,
    oob_score=True,           # estima performance con muestras OOB
    n_jobs=-1,
    random_state=42
)

param_dist = {
    "max_depth": [None, 8, 12, 16, 24, 32],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(
    base_rf,
    param_distributions=param_dist,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rs.fit(X_train, y_train)

best_rf = rs.best_estimator_

# ===========================
# 3) Calibración de probabilidades (mejores "scores")
# ===========================
cal_rf = CalibratedClassifierCV(best_rf, method="sigmoid", cv=3)
cal_rf.fit(X_train, y_train)

# ===========================
# 4) Métricas en test
# ===========================
y_prob = cal_rf.predict_proba(X_test)[:, 1]
y_pred = cal_rf.predict(X_test)

metrics = {
    "best_cv_auc": float(rs.best_score_),
    "test_auc": float(roc_auc_score(y_test, y_prob)),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "oob_score": float(getattr(best_rf, "oob_score_", np.nan)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "best_params": rs.best_params_,
}
print("Metrics:", metrics)
print(f"scikit-learn: {SKLEARN_VERSION}")

# ===========================
# 5) Guardar artefactos (mismo nombre que usas en predict)
# ===========================
ARTIFACT_DIR = Path("ml_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

version = "v1"  # mantenemos v1 para que tu predict lo lea sin cambiar flags
model_path = ARTIFACT_DIR / f"random_forest_citas_{version}.joblib"
meta_path = ARTIFACT_DIR / f"random_forest_citas_{version}.json"

joblib.dump({
    "model": cal_rf,                 # <--- guarda el CALIBRADO
    "features": FEATURES,
    "sklearn_version": SKLEARN_VERSION,
    "trained_at": datetime.utcnow().isoformat(),
    "version": version,
    "best_params": rs.best_params_,
}, model_path)

with meta_path.open("w", encoding="utf-8") as f:
    json.dump({
        "metrics": metrics,
        "features": FEATURES,
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "sklearn_version": SKLEARN_VERSION,
        "best_params": rs.best_params_,
    }, f, ensure_ascii=False, indent=2)

print(f"Saved: {model_path}")
print(f"Saved: {meta_path}")