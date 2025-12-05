import argparse
from datetime import datetime, timedelta
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import LabelEncoder  # Para la codificación de 'dia_semana'

# --- Configuración básica ---
HORA_INICIO = 8  # Hora de inicio de jornada
HORA_FIN = 21    # Hora de fin de jornada

PRIORIDAD_NUM = {
    "muy_urgente": 5,
    "alta": 4,
    "media": 3,
    "regular": 2,
    "baja": 1,
    "muy_baja": 0,
}

FEATURE_ORDER = [
    "dia_semana", "hora", "prioridad_numeric", "edad",
    "dias_desde_ultima_sesion", "match_disponibilidad",
    "ocupacion_slot_psico", "paciente_prefiere_tarde"
]

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# --- Conexión a la Base de Datos ---
def get_db_connection(db_url):
    """
    Establece una conexión a la base de datos usando SQLAlchemy.
    """
    print(f"Conectando a la base de datos con URL: {db_url}")
    try:
        engine = create_engine(db_url)
        connection = engine.connect()
        Session = sessionmaker(bind=engine)
        session = Session()
        print("Conexión establecida exitosamente.")  # Verificación de conexión
        return session, engine
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        raise

# --- Leer datos de pacientes desde la base de datos ---
def read_pacientes_db(session):
    """
    Lee los datos de los pacientes desde la base de datos (tabla `pacientes`).
    """
    query = """
    SELECT id_paciente, nombre, apellido, fecha_nacimiento, edad, trastorno_salud_mental, prioridad_clinica
    FROM pacientes
    WHERE prioridad_clinica IN ('muy_urgente', 'alta', 'media', 'regular', 'baja', 'muy_baja')
    """
    df_pacientes = pd.read_sql(query, session.bind)
    
    # Crear la columna 'dia_semana' a partir de la fecha de nacimiento
    df_pacientes['fecha_nacimiento'] = pd.to_datetime(df_pacientes['fecha_nacimiento'])
    df_pacientes['dia_semana'] = df_pacientes['fecha_nacimiento'].dt.dayofweek
    df_pacientes['dia_semana'] = df_pacientes['dia_semana'].map({i: DIAS[i] for i in range(7)})

    # Asignar la columna 'hora' con un valor por defecto (ajustar según tu lógica)
    df_pacientes['hora'] = HORA_INICIO  # Asignando un valor por defecto a 'hora'

    # Convertir la columna 'dia_semana' a valores numéricos
    label_encoder = LabelEncoder()
    df_pacientes['dia_semana'] = label_encoder.fit_transform(df_pacientes['dia_semana'])

    return df_pacientes

# --- Función para obtener el score usando el modelo ---
def scorear_prioridad(model, prioridad, df_pacientes, weeks, edad, dias_desde_ultima, prefiere_tarde):
    """
    Calcula los scores para las citas según la prioridad.
    """
    # Filtramos los pacientes según la prioridad
    df_pacientes_prioridad = df_pacientes[df_pacientes['prioridad_clinica'] == prioridad]

    if df_pacientes_prioridad.empty:
        print(f"No hay pacientes con la prioridad '{prioridad}'.")
        return pd.DataFrame()

    # Creamos las nuevas características para el modelo
    df_pacientes_prioridad.loc[:, 'prioridad_numeric'] = df_pacientes_prioridad['prioridad_clinica'].map(PRIORIDAD_NUM)
    df_pacientes_prioridad.loc[:, 'edad'] = edad
    df_pacientes_prioridad.loc[:, 'dias_desde_ultima_sesion'] = dias_desde_ultima
    df_pacientes_prioridad.loc[:, 'match_disponibilidad'] = 1  # Esto debe ser calculado si es necesario
    df_pacientes_prioridad.loc[:, 'ocupacion_slot_psico'] = 1  # Esto debe ser calculado si es necesario
    df_pacientes_prioridad.loc[:, 'paciente_prefiere_tarde'] = prefiere_tarde

    # Usamos solo las columnas necesarias
    features = df_pacientes_prioridad[FEATURE_ORDER]

    # Predecir usando el modelo RandomForest
    df_pacientes_prioridad['score'] = model.predict_proba(features)[:, 1]

    return df_pacientes_prioridad[['prioridad_clinica', 'fecha_nacimiento', 'edad', 'score', 'id_paciente', 'prioridad_clinica']]

# --- Guardar citas en base de datos ---
def guardar_citas_en_bd(df_citas, session, engine):
    """
    Guarda las citas generadas en la base de datos.
    """
    try:
        # Comienza una nueva transacción
        with engine.begin():
            for _, row in df_citas.iterrows():
                # Insertar cada fila de citas en la base de datos
                sql = """
                INSERT INTO citas_sugeridas (prioridad, fecha, hora, score, id_paciente, origen, estado, created_at, fecha_str, hora_str, orden)
                VALUES (:prioridad, :fecha, :hora, :score, :id_paciente, :origen, :estado, :created_at, :fecha_str, :hora_str, :orden)
                """
                session.execute(sql, {
                    "prioridad": row["prioridad_clinica"],
                    "fecha": row["fecha_nacimiento"],  # Aquí puedes ajustar si tienes la fecha de la cita
                    "hora": row["hora"],  # Añadir hora si es necesario
                    "score": row["score"],
                    "id_paciente": row["id_paciente"],
                    "origen": 'generado',  # Puedes cambiarlo según sea necesario
                    "estado": 'pendiente',  # Puedes cambiarlo según sea necesario
                    "created_at": datetime.now(),  # Establecer la fecha de creación
                    "fecha_str": str(row["fecha_nacimiento"]),  # Ajusta esto si tienes otro formato de fecha
                    "hora_str": str(row["hora"]),  # Ajusta esto si tienes otro formato de hora
                    "orden": 1  # Ajusta esto según sea necesario
                })
            session.commit()
        print(f"{len(df_citas)} citas guardadas en la base de datos.")
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error al guardar las citas en la base de datos: {e}")
    finally:
        session.close()

# --- Función principal ---
def main():
    ap = argparse.ArgumentParser(description="Generador de slots con Random Forest (solo CLI)")
    ap.add_argument("--artifact", default="ml_artifacts/random_forest_citas_v1.joblib")
    ap.add_argument("--weeks", type=int, default=2)
    ap.add_argument("--edad", type=int, default=30)
    ap.add_argument("--dias_desde_ultima", type=int, default=14)
    ap.add_argument("--prioridades", type=str, default=None, help="lista separada por comas: muy_urgente,alta,...")
    ap.add_argument("--prefiere_tarde", action="store_true")
    ap.add_argument("--db_url", required=True, help="URL de conexión a la base de datos")
    ap.add_argument("--output_file", required=True, help="Nombre del archivo de salida")
    ap.add_argument("--file_type", choices=["csv"], default="csv", help="Tipo de archivo de salida")
    args = ap.parse_args()

    # Conectar a la base de datos
    session, engine = get_db_connection(args.db_url)

    # Obtener los datos de los pacientes desde la base de datos
    df_pacientes = read_pacientes_db(session)

    # Cargar el modelo de Random Forest
    pack = joblib.load(args.artifact)
    model = pack["model"] if isinstance(pack, dict) and "model" in pack else pack

    # Proceso para las prioridades
    prios = [p.strip() for p in args.prioridades.split(",") if p.strip()]
    if not prios:
        print("No se especificaron prioridades válidas.")
        return

    for p in prios:
        dfp = scorear_prioridad(model, p, df_pacientes, args.weeks, args.edad, args.dias_desde_ultima, args.prefiere_tarde)

        if dfp.empty:
            print(f"No hubo resultados para la prioridad: {p}")
        else:
            # Guardar citas generadas en la base de datos
            guardar_citas_en_bd(dfp, session, engine)

            # Si el archivo CSV es requerido
            if args.file_type == "csv":
                dfp.to_csv(args.output_file, index=False)
                print(f"Archivo CSV generado: {args.output_file}")

if __name__ == "__main__":
    main()
