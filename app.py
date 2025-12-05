# app.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import joblib
import pandas as pd
from datetime import datetime, timedelta, time
from sqlalchemy.orm import Session

# Configuración de la base de datos (usando los datos que pasaste)
DATABASE_URL = "postgresql://postgres:@127.0.0.1:5432/consultorio"  # String de conexión con PostgreSQL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Definir tu modelo de paciente
class Paciente(Base):
    __tablename__ = "pacientes"
    
    id_paciente = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    apellido = Column(String, index=True)
    prioridad_clinica = Column(String, index=True)  # Puede ser "alta", "media", "baja", etc.
    edad = Column(Integer)

# Crear la app FastAPI
app = FastAPI()

# Modelo de entrada para el body de la API
class Disponibilidad(BaseModel):
    dia_semana: int
    hora_inicio: str
    hora_fin: str

class PrediccionRequest(BaseModel):
    semanas: int
    top_k: int
    prioridades: str
    edad: int
    dias_desde_ultima: int
    paciente_csv: list[Disponibilidad]
    psi_csv: list[Disponibilidad]
    prefiere_tarde: bool

# Cargar el modelo preentrenado
pack = joblib.load("ml_artifacts/random_forest_citas_v1.joblib")
model = pack["model"]

# Definir las constantes de inicio y fin de hora
HORA_INICIO = 9  # La hora de inicio de las citas
HORA_FIN = 18  # La hora final de las citas

# Definir las constantes reemplazadas
PRIORIDAD_NUM = {
    "muy_urgente": 5,
    "alta": 4,
    "media": 3,
    "regular": 2,
    "baja": 1,
    "muy_baja": 0,
}

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

FEATURE_ORDER = [
    "dia_semana", "hora", "prioridad_numeric", "edad", "dias_desde_ultima_sesion",
    "match_disponibilidad", "ocupacion_slot_psico", "paciente_prefiere_tarde"
]

def parse_hhmm(s: str):
    h, m = s.split(":")
    return time(int(h), int(m))

def read_disponibilidad(disponibilidad_list):
    # Convertir los datos del paciente/psicólogo a un DataFrame
    data = []
    for item in disponibilidad_list:
        data.append({
            "dia_semana": item.dia_semana,
            "hora_inicio_t": parse_hhmm(item.hora_inicio),
            "hora_fin_t": parse_hhmm(item.hora_fin),
        })
    return pd.DataFrame(data)

def disponible_en(df_disp, dt: datetime) -> int:
    dow = dt.weekday()
    start = time(dt.hour, 0)
    end = time(dt.hour + 1, 0)
    subset = df_disp[df_disp["dia_semana"] == dow]
    for _, row in subset.iterrows():
        if (row["hora_inicio_t"] <= start) and (row["hora_fin_t"] >= end):
            return 1
    return 0

def generar_slots(semanas=2, base_dt=None):
    base = base_dt or datetime.now().replace(minute=0, second=0, microsecond=0)
    end = base + timedelta(weeks=semanas)
    cur = base
    while cur < end:
        if HORA_INICIO <= cur.hour < HORA_FIN:
            yield cur
            cur += timedelta(hours=1)
        else:
            next_day = (cur + timedelta(days=1)).replace(hour=HORA_INICIO, minute=0, second=0, microsecond=0)
            cur = next_day

def scorear_prioridad(model, prioridad_key, df_pac, df_psi, semanas, edad, dias_desde_ultima, prefiere_tarde):
    filas = []
    for dt in generar_slots(semanas=semanas):
        if not (disponible_en(df_pac, dt) and disponible_en(df_psi, dt)):
            continue
        feats = {
            "dia_semana": dt.weekday(),
            "hora": dt.hour,
            "prioridad_numeric": PRIORIDAD_NUM[prioridad_key],
            "edad": edad,
            "dias_desde_ultima_sesion": dias_desde_ultima,
            "match_disponibilidad": 1,
            "ocupacion_slot_psico": 0,
            "paciente_prefiere_tarde": int(prefiere_tarde),
        }
        X = pd.DataFrame([feats], columns=FEATURE_ORDER)
        score = float(model.predict_proba(X)[0, 1])
        filas.append({
            "prioridad": prioridad_key,
            "fecha": pd.Timestamp(dt),
            "dia": DIAS[feats["dia_semana"]],
            "hora": feats["hora"],
            "score": score
        })
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["fecha_str"] = df["fecha"].dt.strftime("%Y-%m-%d")
    df["hora_str"] = df["hora"].map(lambda h: f"{h:02d}:00")
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

# Obtener pacientes de la base de datos
def get_pacientes(db):
    return db.query(Paciente).filter(Paciente.prioridad_clinica.in_(['alta', 'media', 'baja'])).all()

# Función para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict_slots/")  # Endpoint para hacer la predicción de los slots
async def predict_slots(request: PrediccionRequest, db: Session = Depends(get_db)):
    try:
        # Leer las disponibilidades del paciente y psicólogo
        df_pac = read_disponibilidad(request.paciente_csv)
        df_psi = read_disponibilidad(request.psi_csv)

        # Obtener pacientes desde la base de datos
        pacientes = get_pacientes(db)
        
        # Procesar pacientes y hacer la predicción (simplificado)
        pacientes_data = [{"id_paciente": paciente.id_paciente, "nombre": paciente.nombre, "apellido": paciente.apellido} for paciente in pacientes]

        # Modo prioridad múltiple
        prios = [p.strip() for p in request.prioridades.split(",")]

        dfs = []
        for p in prios:
            dfp = scorear_prioridad(model, p, df_pac, df_psi, request.semanas, request.edad, request.dias_desde_ultima, request.prefiere_tarde)
            if dfp.empty:
                continue
            top = dfp.head(request.top_k).copy()
            dfs.append(top)

        if not dfs:
            raise HTTPException(status_code=404, detail="No hubo resultados que cumplan disponibilidad de paciente y psicólogo.")

        # Combinar todos los resultados y devolverlos
        combined = pd.concat(dfs, ignore_index=True)

        return {"slots": combined.to_dict(orient="records")}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail="Error al acceder a la base de datos")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
