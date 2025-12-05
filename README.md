# Standalone Random Forest — Citas con disponibilidad por CSV

Este paquete **no depende de Django**. Entrena y predice slots de citas **leyendo la disponibilidad** de **paciente** y **psicólogo** desde archivos CSV.

## CSVs de disponibilidad (plantillas incluidas)
- `data/disponibilidad_paciente.csv`
- `data/disponibilidad_psicologo.csv`

**Formato (cabeceras obligatorias):**
```
dia_semana,hora_inicio,hora_fin
0,10:00,18:00   # Lunes 10-18
1,10:00,18:00   # Martes 10-18
...
```

- `dia_semana`: 0=Lunes ... 6=Domingo
- `hora_inicio` / `hora_fin`: en HH:MM (24h). `hora_fin` es exclusivo (ej.: 18:00 implica que 17:00-18:00 es el último bloque).
- Puedes repetir filas para varios tramos en un mismo día.

## Pasos rápidos (Windows / PowerShell)

### Python 3.11 (recomendado)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-311.txt
```

### Entrenamiento
```powershell
python ml/train_random_forest_citas.py
```

### Predicción (leyendo CSVs)
```powershell
python ml/predict_slots.py --weeks 2 --top_k 10 --prioridad alta ^
  --pac_csv data/disponibilidad_paciente.csv ^
  --psi_csv data/disponibilidad_psicologo.csv ^
  --csv salidas/top_slots.csv --calendar
```

- `--csv`: exporta resultados a CSV (opcional)
- `--calendar`: imprime calendario ASCII marcando TOP slots (opcional)
