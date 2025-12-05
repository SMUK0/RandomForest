# schedule_cli.py
from _future_ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sqlalchemy import create_engine, text, bindparam

# ----------------- Configuración -----------------
HORA_INICIO = 8     # jornada del psicólogo (ajústalo si quieres)
HORA_FIN    = 21
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# Prioridades que entiende el modelo
PRIORIDAD_NUM: Dict[str, int] = {
    "muy_urgente": 5,
    "alta": 4,
    "media": 3,
    "regular": 2,
    "baja": 1,
    "muy_baja": 0,
}

# Mapeo desde tu BD (Paciente.prioridad_clinica = alto/medio/bajo)
MAP_DB_TO_RF: Dict[str, str] = {
    "alto": "alta",
    "medio": "media",
    "bajo": "baja",
}

FEATURE_ORDER = [
    "dia_semana", "hora", "prioridad_numeric", "edad",
    "dias_desde_ultima_sesion", "match_disponibilidad",
    "ocupacion_slot_psico", "paciente_prefiere_tarde"
]

# ----------------- Helpers -----------------
def round_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def generar_slots(semanas: int = 2, base_dt: Optional[datetime] = None):
    base = round_to_hour(base_dt or datetime.now())
    end = base + timedelta(weeks=semanas)
    cur = base
    while cur < end:
        if HORA_INICIO <= cur.hour < HORA_FIN:
            yield cur
            cur += timedelta(hours=1)
        else:
            cur = (cur + timedelta(days=1)).replace(
                hour=HORA_INICIO, minute=0, second=0, microsecond=0
            )

def ascii_calendar(rows: pd.DataFrame, weeks: int = 2, titulo: str = ""):
    if rows.empty:
        print(f"\n[Calendario {titulo}] Sin resultados.")
        return
    start: pd.Timestamp = rows["fecha"].min()
    start = start - pd.Timedelta(days=int(start.weekday() % 7))

    print(f"\n=== Calendario {titulo} ===")
    for w in range(weeks):
        semana_ini = start + pd.Timedelta(weeks=w)
        semana_fin = semana_ini + pd.Timedelta(days=6)
        print(f"\nSEMANA {w+1}: {semana_ini.date()} — {semana_fin.date()}")
        header = "Hora  | " + " | ".join([
            f"{(semana_ini + pd.Timedelta(days=d)).strftime('%d-%m')} {DIAS[d]:>3}"
            for d in range(7)
        ])
        print(header)
        print("-" * len(header))
        for h in range(HORA_INICIO, HORA_FIN):
            line = f"{h:02d}:00 | "
            cols = []
            for d in range(7):
                day = (semana_ini + pd.Timedelta(days=d)).date()
                hit = rows[(rows["fecha"].dt.date == day) & (rows["hora"] == h)]
                if not hit.empty:
                    s = f"{hit.iloc[0]['alias']}-{hit.iloc[0]['prio'][0].upper()}"
                else:
                    s = ""
                cols.append(f"{s:^10}")
            print(line + " | ".join(cols))

def within(rango: Dict, t: time) -> bool:
    return rango["hora_inicio"] <= t < rango["hora_fin"]

def paciente_prefiere_tarde(disps: List[Dict]) -> int:
    """Heurística: si ≥60% de rangos empiezan después de las 14:00."""
    if not disps:
        return 0
    tarde = sum(1 for r in disps if r["hora_inicio"].hour >= 14)
    return int(tarde / max(1, len(disps)) >= 0.6)

def dias_desde_ultima(fecha_ultima: Optional[datetime]) -> int:
    if not fecha_ultima:
        return 90
    return max(0, (datetime.now() - fecha_ultima).days)

def alias_paciente(nombre: str, apellido: str, idp: int) -> str:
    base = f"{nombre[:1]}{apellido[:1]}".upper()
    return f"{base}{str(idp)[-2:]}"

# ----------------- Capa datos (SQLAlchemy) -----------------
@dataclass
class PacienteRow:
    id: int
    nombre: str
    apellido: str
    edad: int
    prioridad_text: str
    prio_rf: str
    prefiere_tarde: int
    ultima_sesion: Optional[datetime]
    disponibilidades: List[Dict]

def fetch_pacientes(engine, psicologo_id: int) -> List[PacienteRow]:
    """Pacientes del psicólogo + última sesión completada + disponibilidades activas."""
    with engine.begin() as con:
        q = con.execute(text("""
            SELECT p.id_paciente, p.nombre, p.apellido, p.edad, p.prioridad_clinica,
                   MAX(CASE WHEN c.estado='completada' THEN c.fecha_hora END) AS ultima_sesion
            FROM pacientes p
            LEFT JOIN citas c ON c.id_paciente = p.id_paciente
            WHERE p.id_psicologo = :psi
            GROUP BY p.id_paciente, p.nombre, p.apellido, p.edad, p.prioridad_clinica
        """), {"psi": psicologo_id})

        base: Dict[int, Dict] = {
            r.id_paciente: dict(
                id=r.id_paciente,
                nombre=r.nombre,
                apellido=r.apellido,
                edad=(r.edad if r.edad is not None else 30),
                prioridad=(r.prioridad_clinica or "medio"),
                ultima=r.ultima_sesion,
            )
            for r in q.mappings()
        }

        if not base:
            return []

        ids = list(base.keys())
        q2 = text("""
            SELECT d.id_paciente, d.dia_semana, d.hora_inicio, d.hora_fin
            FROM disponibilidades d
            WHERE d.activo = TRUE AND d.id_paciente IN :ids
        """).bindparams(bindparam("ids", expanding=True))

        disp_map: Dict[int, List[Dict]] = {k: [] for k in ids}
        for r in con.execute(q2, {"ids": ids}).mappings():
            disp_map[r.id_paciente].append({
                "dia_semana": int(r.dia_semana),
                "hora_inicio": r.hora_inicio,
                "hora_fin": r.hora_fin,
            })

        out: List[PacienteRow] = []
        for pid, b in base.items():
            prio_rf = MAP_DB_TO_RF.get((b["prioridad"] or "").lower(), "media")
            out.append(PacienteRow(
                id=pid,
                nombre=b["nombre"],
                apellido=b["apellido"],
                edad=b["edad"],
                prioridad_text=b["prioridad"],
                prio_rf=prio_rf,
                prefiere_tarde=paciente_prefiere_tarde(disp_map.get(pid, [])),
                ultima_sesion=b["ultima"],
                disponibilidades=disp_map.get(pid, []),
            ))
        return out

def fetch_citas_ocupadas(engine, psicologo_id: int, start: datetime, end: datetime) -> set[Tuple]:
    """Retorna {(date, hour)} ya ocupados por el psicólogo en el rango."""
    with engine.begin() as con:
        q = con.execute(text("""
            SELECT fecha_hora
            FROM citas
            WHERE id_psicologo = :psi
              AND fecha_hora >= :ini AND fecha_hora < :fin
              AND estado != 'cancelada'
        """), {"psi": psicologo_id, "ini": start, "fin": end})
        busy: set[Tuple] = set()
        for (fh,) in q.fetchall():
            fh = round_to_hour(fh)
            busy.add((fh.date(), fh.hour))
        return busy

# ----------------- Scoring y selección -----------------
def generar_candidatos(model, pacientes: List[PacienteRow], busy: set[Tuple], weeks: int) -> pd.DataFrame:
    filas: List[Dict] = []
    horizon_start = round_to_hour(datetime.now())
    for p in pacientes:
        for dt in generar_slots(semanas=weeks, base_dt=horizon_start):
            t = time(dt.hour, 0)
            dow = dt.weekday()

            # disponibilidad del paciente
            if not any((r["dia_semana"] == dow) and within(r, t) for r in p.disponibilidades):
                continue

            # ocupación del psicólogo
            if (dt.date(), dt.hour) in busy:
                continue

            feats = {
                "dia_semana": dow,
                "hora": dt.hour,
                "prioridad_numeric": PRIORIDAD_NUM[p.prio_rf],
                "edad": p.edad,
                "dias_desde_ultima_sesion": dias_desde_ultima(p.ultima_sesion),
                "match_disponibilidad": 1,
                "ocupacion_slot_psico": 0,  # ya filtrado arriba
                "paciente_prefiere_tarde": p.prefiere_tarde,
            }
            X = pd.DataFrame([feats], columns=FEATURE_ORDER)
            score = float(model.predict_proba(X)[0, 1])

            filas.append({
                "paciente_id": p.id,
                "paciente": f"{p.nombre} {p.apellido}",
                "alias": alias_paciente(p.nombre, p.apellido, p.id),
                "prio": p.prio_rf,
                "prio_w": PRIORIDAD_NUM[p.prio_rf],
                "fecha": pd.Timestamp(dt),
                "dia": DIAS[dow],
                "hora": dt.hour,
                "score": score,
            })

    df = pd.DataFrame(filas)
    if df.empty:
        return df
    # ordenar por prioridad numérica y score
    return df.sort_values(["prio_w", "score"], ascending=[False, False]).reset_index(drop=True)

def empacar_horario(df: pd.DataFrame, max_por_semana: int = 40, max_por_dia: int = 8) -> pd.DataFrame:
    """
    Greedy: ordenado por prioridad/score; evita 2 citas del mismo paciente el mismo día y evita colisiones.
    """
    if df.empty:
        return df

    df = df.copy()
    df["slot_key"] = list(zip(df["fecha"].dt.date, df["hora"]))
    df["pd_key"] = list(zip(df["paciente_id"], df["fecha"].dt.date))

    usados, usados_pd = set(), set()
    sel_rows: List[Dict] = []

    for _, r in df.iterrows():
        if r["slot_key"] in usados:
            continue
        if r["pd_key"] in usados_pd:
            continue
        # límite por día total (de la agenda del psicólogo)
        if sum(1 for sk in usados if sk[0] == r["slot_key"][0]) >= max_por_dia:
            continue

        sel_rows.append(r)
        usados.add(r["slot_key"])
        usados_pd.add(r["pd_key"])

        if len(sel_rows) >= max_por_semana:
            break

    out = pd.DataFrame(sel_rows).drop(columns=["slot_key", "pd_key"])
    if out.empty:
        return out
    out["fecha_str"] = out["fecha"].dt.strftime("%Y-%m-%d")
    out["hora_str"] = out["hora"].map(lambda h: f"{h:02d}:00")
    return out.sort_values(["fecha", "hora"]).reset_index(drop=True)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="Generar horario de citas (CLI) con Random Forest")
    ap.add_argument("--db-url", default="postgresql://postgres:@127.0.0.1:5432/consultorio")
    ap.add_argument("--psicologo-id", type=int, required=True)
    ap.add_argument("--artifact", default="ml_artifacts/random_forest_citas_v1.joblib")
    ap.add_argument("--weeks", type=int, default=2)
    ap.add_argument("--max-semanal", type=int, default=40, help="máximo de citas a proponer")
    ap.add_argument("--max-por-dia", type=int, default=8, help="tope por día")
    ap.add_argument("--csv", type=str, default="", help="ruta para guardar CSV (opcional)")
    ap.add_argument("--calendar", action="store_true", help="mostrar calendario en consola")
    args = ap.parse_args()

    pack = joblib.load(args.artifact)
    model = pack["model"]

    engine = create_engine(args.db_url)

    pacientes = fetch_pacientes(engine, args.psicologo_id)
    if not pacientes:
        print("⚠ No hay pacientes asignados o sin disponibilidad activa.")
        return

    horizon_start = round_to_hour(datetime.now())
    horizon_end = horizon_start + timedelta(weeks=args.weeks)
    ocupados = fetch_citas_ocupadas(engine, args.psicologo_id, horizon_start, horizon_end)

    candidatos = generar_candidatos(model, pacientes, ocupados, args.weeks)
    if candidatos.empty:
        print("⚠ No hay slots que cumplan disponibilidad del paciente + agenda del psicólogo.")
        return

    seleccion = empacar_horario(
        candidatos,
        max_por_semana=args.max_semanal,
        max_por_dia=args.max_por_dia
    )

    # -------- Salida terminal --------
    print(f"\nTop {len(seleccion)} propuestas para psicólogo #{args.psicologo_id}")
    if not seleccion.empty:
        print(
            seleccion[["dia", "fecha_str", "hora_str", "alias", "prio", "score"]]
            .rename(columns={
                "dia": "Día", "fecha_str": "Fecha", "hora_str": "Hora",
                "alias": "Paciente", "prio": "Prioridad", "score": "Score"
            })
            .to_string(index=False, justify="left")
        )

    if args.calendar and not seleccion.empty:
        ascii_calendar(seleccion[["fecha", "hora", "alias", "prio"]], weeks=args.weeks,
                       titulo=f"Psicólogo {args.psicologo_id}")

    if args.csv and not seleccion.empty:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        seleccion.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n💾 Guardado CSV en: {out.resolve()}")

if _name_ == "_main_":
    main()