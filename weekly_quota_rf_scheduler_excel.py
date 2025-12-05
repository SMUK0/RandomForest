import argparse
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path

# --- Funciones auxiliares ---

def generar_slots_semana(inicio_semana, horas=(8, 21)):
    """Genera todos los posibles slots horarios para una semana."""
    dias = range(7)
    horas_rango = range(horas[0], horas[1])
    slots = []
    for d in dias:
        for h in horas_rango:
            inicio = f"{h:02d}:00"
            fin = f"{h+1:02d}:00"
            slots.append((d, inicio, fin))
    return slots

def cargar_disponibilidades(pac_csv, psi_csv):
    """Carga la disponibilidad de pacientes y psicólogos."""
    pacientes = pd.read_csv(pac_csv)
    psicologos = pd.read_csv(psi_csv)
    return pacientes, psicologos

def crear_dataset_prediccion(pacientes, psicologos, slots):
    """Crea combinaciones posibles (paciente, psicólogo, horario)."""
    combinaciones = []
    for _, pac in pacientes.iterrows():
        for _, psi in psicologos.iterrows():
            for (dia, inicio, fin) in slots:
                if pac["dia_semana"] == dia and psi["dia_semana"] == dia:
                    if (pac["hora_inicio"] <= inicio < pac["hora_fin"]) and (psi["hora_inicio"] <= inicio < psi["hora_fin"]):
                        combinaciones.append({
                            "paciente": pac["nombre"],
                            "psicologo": psi["nombre"],
                            "dia_semana": dia,
                            "hora_inicio": inicio,
                            "hora_fin": fin,
                            "prioridad": pac.get("prioridad", "media")
                        })
    return pd.DataFrame(combinaciones)

def seleccionar_top_citas(predicciones, top_k_per):
    """Selecciona las mejores citas sin solapamiento."""
    predicciones = predicciones.sort_values(by="probabilidad", ascending=False)
    seleccionadas = []
    ocupados = set()

    for _, row in predicciones.iterrows():
        clave = (row["dia_semana"], row["hora_inicio"])
        if clave not in ocupados:
            seleccionadas.append(row)
            ocupados.add(clave)
        if len(seleccionadas) >= top_k_per:
            break

    return pd.DataFrame(seleccionadas)

def mostrar_calendario(citas, inicio_semana):
    """Imprime el calendario semanal."""
    print("\n=== CALENDARIO GENERADO ===\n")
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    horas = [f"{h:02d}:00" for h in range(8, 21)]
    calendario = pd.DataFrame(index=horas, columns=dias).fillna("")

    for _, row in citas.iterrows():
        dia = dias[row["dia_semana"]]
        hora = row["hora_inicio"]
        valor = f"{row['paciente']}-{row['psicologo']} ({row['prioridad']})"
        calendario.loc[hora, dia] = valor

    print(f"SEMANA: {inicio_semana.date()} - {(inicio_semana + timedelta(days=6)).date()}")
    print(calendario.to_string())

# --- Script principal ---

def main():
    parser = argparse.ArgumentParser(description="Generador de horario semanal con Random Forest")
    parser.add_argument("--artifact", required=True, help="Ruta al modelo entrenado .joblib")
    parser.add_argument("--pac_csv", required=True, help="CSV de disponibilidad de pacientes")
    parser.add_argument("--psi_csv", required=True, help="CSV de disponibilidad de psicólogos")
    parser.add_argument("--weeks", type=int, default=1, help="Número de semanas a generar")
    parser.add_argument("--top_k_per", type=int, default=20, help="Cantidad máxima de citas por semana")
    parser.add_argument("--csv", default="salidas/horario_combinado.csv", help="Ruta de salida CSV")
    parser.add_argument("--calendar", action="store_true", help="Mostrar calendario en consola")
    args = parser.parse_args()

    # Cargar modelo
    model = joblib.load(args.artifact)

    # Generar slots base
    inicio_semana = datetime.now()
    slots = generar_slots_semana(inicio_semana)

    # Cargar datos
    pacientes, psicologos = cargar_disponibilidades(args.pac_csv, args.psi_csv)
    df_pred = crear_dataset_prediccion(pacientes, psicologos, slots)

    if df_pred.empty:
        print("⚠️ No hay combinaciones válidas entre pacientes y psicólogos.")
        return

    # Simular predicciones (para tu modelo real, ajusta columnas)
    X = pd.get_dummies(df_pred[["dia_semana"]], drop_first=True)
    df_pred["probabilidad"] = model.predict_proba(X)[:, 1]

    # Seleccionar top citas sin solapamiento
    top_citas = seleccionar_top_citas(df_pred, args.top_k_per)

    # Guardar en CSV
    Path("salidas").mkdir(exist_ok=True)
    top_citas.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print(f"✅ Horario guardado en: {args.csv} ({len(top_citas)} filas)")

    # Mostrar calendario
    if args.calendar:
        mostrar_calendario(top_citas, inicio_semana)

if __name__ == "__main__":
    main()
