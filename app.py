import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser

import gpxpy
from fitparse import FitFile
from lxml import etree

# ============================================================
# 1) Tipos de actividad (para Strava CSV principalmente)
# ============================================================

RUN_TYPES = {
    "run",
    "running",
    "carrera",
    "trail run",
    "treadmill",
    "virtual run",
}

BIKE_TYPES = {
    "ride",
    "cycling",
    "bicicleta",
    "bike",
    "road",
    "road cycling",
    "mountain bike",
    "mtb",
    "gravel ride",
    "virtual ride",
    "e-bike ride",
    "ebike ride",
}

GOALS = {
    "correr": [
        ("maraton", "Maratón (en ~12 meses)"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
        ("media", "Media maratón"),
        ("10k", "10K"),
        ("base", "Solo base aeróbica"),
    ],
    "bicicleta": [
        ("gran_fondo", "Gran fondo (larga distancia, en ~12 meses)"),
        ("resistencia", "Mejorar resistencia y bajar pulsaciones"),
        ("puerto", "Subidas / puertos (mejorar umbral en subida)"),
        ("criterium", "Potencia y cambios de ritmo (tipo criterium)"),
        ("base", "Solo base aeróbica"),
    ],
}

# ============================================================
# 2) Librerías de sesiones (movilidad / core / fuerza)
# ============================================================

MOBILITY_LIBRARY = {
    "movilidad_20": [
        "Caderas: 90/90 (2x60s por lado)",
        "Tobillos: rodilla a pared (2x12 por lado)",
        "Isquios: bisagra con banda (2x10)",
        "Columna torácica: rotaciones en cuadrupedia (2x8 por lado)",
        "Glúteo medio: monster walks con minibanda (2x12 pasos por lado)",
    ],
    "movilidad_10": [
        "Tobillos: rodilla a pared (2x10 por lado)",
        "Caderas: 90/90 (1x60s por lado)",
        "Torácica: rotaciones (1x8 por lado)",
    ],
}

CORE_LIBRARY = {
    "core_20": [
        "Plancha frontal 3x40s (20s descanso)",
        "Plancha lateral 3x30s por lado",
        "Dead bug 3x10 por lado",
        "Puente glúteo 3x12",
        "Bird-dog 3x10 por lado",
    ],
    "core_10": [
        "Plancha frontal 2x40s",
        "Dead bug 2x10 por lado",
        "Puente glúteo 2x12",
    ],
}

STRENGTH_LIBRARY = {
    "fuerza_35": [
        "Sentadilla goblet 4x8 (RPE 7)",
        "Peso muerto rumano 4x8 (RPE 7)",
        "Zancadas 3x10 por lado (RPE 7)",
        "Elevación de gemelos 4x12",
        "Remo con mancuerna 3x10 por lado",
    ],
    "fuerza_25": [
        "Sentadilla goblet 3x8 (RPE 7)",
        "Peso muerto rumano 3x8 (RPE 7)",
        "Gemelos 3x12",
        "Remo 2x10 por lado",
    ],
}

# ============================================================
# 3) Utilidades
# ============================================================

def next_monday(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7)


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def parse_time_to_minutes(val) -> Optional[float]:
    """
    Acepta:
    - segundos numéricos
    - 'hh:mm:ss' o 'mm:ss'
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        # Strava a veces usa segundos
        if val > 1000:
            return float(val) / 60.0
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [int(p) for p in parts]
        except Exception:
            return None
        if len(parts) == 3:
            h, m, sec = parts
            return (h * 3600 + m * 60 + sec) / 60.0
        if len(parts) == 2:
            m, sec = parts
            return (m * 60 + sec) / 60.0
    return safe_float(s)


def normalise_colname(c: str) -> str:
    return (
        c.strip()
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("/", " ")
        .replace("-", " ")
    )


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Mapeo típico del export de Strava.
    """
    cols = {normalise_colname(c): c for c in df.columns}

    def pick(*candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return ""

    return {
        "type": pick("activity type", "type"),
        "date": pick("activity date", "date", "start date", "start_time", "start time"),
        "name": pick("activity name", "name", "title"),
        "distance": pick("distance", "distance km", "distance mi"),
        "moving_time": pick("moving time", "moving_time", "duration", "elapsed time"),
        "avg_hr": pick(
            "average heart rate", "avg heart rate", "avg_hr", "averageheartrate"
        ),
        "max_hr": pick("max heart rate", "max_hr", "maximum heart rate"),
    }


def distance_to_km(val) -> Optional[float]:
    f = safe_float(val)
    if f is None:
        return None
    # Heurística: si parece estar en metros (val > 200), convertir a km
    if f > 200:
        return f / 1000.0
    return f


def estimate_hrmax(age: int) -> int:
    return int(round(208 - 0.7 * age))


def hr_zones(hrmax: int) -> Dict[str, Tuple[int, int]]:
    return {
        "Z1": (int(0.60 * hrmax), int(0.70 * hrmax)),
        "Z2": (int(0.70 * hrmax), int(0.80 * hrmax)),
        "Z3": (int(0.80 * hrmax), int(0.87 * hrmax)),
        "Z4": (int(0.87 * hrmax), int(0.93 * hrmax)),
        "Z5": (int(0.93 * hrmax), hrmax),
    }


def weekly_baseline(target_df: pd.DataFrame) -> Dict[str, float]:
    if target_df.empty:
        return {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

    df = target_df.copy()
    df["date_only"] = pd.to_datetime(df["date_only"], errors="coerce").dt.date
    df = df[df["date_only"].notna()]

    today = date.today()
    start = today - timedelta(days=42)
    df = df[df["date_only"] >= start]
    if df.empty:
        return {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

    df["week"] = pd.to_datetime(df["date_only"]).dt.to_period("W-MON")
    w = df.groupby("week")["distance_km"].sum()
    sw = df.groupby("week")["distance_km"].count()
    w_km = float(w.mean()) if len(w) else 0.0
    sessions_w = float(sw.mean()) if len(sw) else 0.0
    long_km = float(df["distance_km"].max()) if df["distance_km"].notna().any() else 0.0

    return {"w_km": w_km, "long_km": long_km, "sessions_w": sessions_w}

# ============================================================
# 4) Parsers (CSV / GPX / TCX / FIT / ZIP) -> DataFrame estándar
# ============================================================

def _to_datetime_safe(s) -> Optional[datetime]:
    try:
        return dtparser.parse(str(s))
    except Exception:
        return None


def _df_standard(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["start_dt", "sport", "distance_km", "moving_min", "avg_hr", "source"]
        )
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["moving_min"] = pd.to_numeric(df["moving_min"], errors="coerce")
    if "avg_hr" not in df.columns:
        df["avg_hr"] = np.nan
    df["avg_hr"] = pd.to_numeric(df["avg_hr"], errors="coerce")
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "sport" not in df.columns:
        df["sport"] = "other"
    return df


def parse_strava_csv(file_bytes: bytes) -> pd.DataFrame:
    df_raw = pd.read_csv(BytesIO(file_bytes))
    colmap = detect_columns(df_raw)

    missing = [k for k, v in colmap.items() if v == "" and k in ("type", "date", "distance")]
    if missing:
        return _df_standard([])

    date_col = colmap["date"]
    type_col = colmap["type"]
    dist_col = colmap["distance"]
    moving_col = colmap["moving_time"]
    avg_hr_col = colmap["avg_hr"]

    df = df_raw.copy()
    df["start_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["start_dt"].notna()]

    def sport_from_type(v) -> str:
        v = str(v).strip().lower()
        if v in RUN_TYPES:
            return "run"
        if v in BIKE_TYPES:
            return "ride"
        return "other"

    df["sport"] = df[type_col].apply(sport_from_type)
    df["distance_km"] = df[dist_col].apply(distance_to_km)

    if moving_col:
        df["moving_min"] = df[moving_col].apply(parse_time_to_minutes)
    else:
        df["moving_min"] = np.nan

    if avg_hr_col:
        df["avg_hr"] = pd.to_numeric(df[avg_hr_col], errors="coerce")
    else:
        df["avg_hr"] = np.nan

    out = df[["start_dt", "sport", "distance_km", "moving_min", "avg_hr"]].copy()
    out["source"] = "strava_csv"
    return out


def parse_gpx(file_bytes: bytes) -> pd.DataFrame:
    gpx = gpxpy.parse(BytesIO(file_bytes))
    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            points = segment.points
            if not points:
                continue
            start_dt = points[0].time
            end_dt = points[-1].time

            dist_m = None
            try:
                dist_m = segment.length_3d()
            except Exception:
                try:
                    dist_m = segment.length_2d()
                except Exception:
                    dist_m = None

            moving_min = None
            if start_dt and end_dt:
                moving_min = (end_dt - start_dt).total_seconds() / 60.0

            rows.append(
                {
                    "start_dt": start_dt,
                    "sport": "other",
                    "distance_km": (dist_m / 1000.0) if dist_m else None,
                    "moving_min": moving_min,
                    "avg_hr": None,
                    "source": "gpx",
                }
            )
    return _df_standard(rows)


def parse_tcx(file_bytes: bytes) -> pd.DataFrame:
    root = etree.parse(BytesIO(file_bytes))
    activities = root.xpath("//*[local-name()='Activity']")
    rows = []
    for act in activities:
        sport_attr = (act.get("Sport") or "").lower()
        sport = (
            "run"
            if "run" in sport_attr
            else "ride"
            if ("bike" in sport_attr or "cycle" in sport_attr)
            else "other"
        )

        laps = act.xpath(".//*[local-name()='Lap']")
        total_dist_m = 0.0
        total_time_s = 0.0
        avg_hr_vals = []
        start_dt = None

        for lap in laps:
            if start_dt is None:
                st_el = lap.get("StartTime")
                if st_el:
                    start_dt = _to_datetime_safe(st_el)

            dist_el = lap.xpath(".//*[local-name()='DistanceMeters']/text()")
            if dist_el:
                try:
                    total_dist_m += float(dist_el[0])
                except Exception:
                    pass

            time_el = lap.xpath(".//*[local-name()='TotalTimeSeconds']/text()")
            if time_el:
                try:
                    total_time_s += float(time_el[0])
                except Exception:
                    pass

            hr_el = lap.xpath(".//*[local-name()='AverageHeartRateBpm']//*[local-name()='Value']/text()")
            if hr_el:
                try:
                    avg_hr_vals.append(float(hr_el[0]))
                except Exception:
                    pass

        avg_hr = float(np.mean(avg_hr_vals)) if avg_hr_vals else None
        rows.append(
            {
                "start_dt": start_dt,
                "sport": sport,
                "distance_km": (total_dist_m / 1000.0) if total_dist_m else None,
                "moving_min": (total_time_s / 60.0) if total_time_s else None,
                "avg_hr": avg_hr,
                "source": "tcx",
            }
        )
    return _df_standard(rows)


def parse_fit(file_bytes: bytes) -> pd.DataFrame:
    fitfile = FitFile(BytesIO(file_bytes))
    rows = []

    sessions = list(fitfile.get_messages("session"))
    if sessions:
        for s in sessions:
            fields = {f.name: f.value for f in s}
            start_dt = fields.get("start_time")
            sport_raw = str(fields.get("sport") or "").lower()
            sport = (
                "run"
                if "run" in sport_raw
                else "ride"
                if ("cycling" in sport_raw or "bike" in sport_raw)
                else "other"
            )
            dist_m = fields.get("total_distance")
            time_s = fields.get("total_timer_time") or fields.get("total_elapsed_time")
            avg_hr = fields.get("avg_heart_rate")

            rows.append(
                {
                    "start_dt": start_dt,
                    "sport": sport,
                    "distance_km": (float(dist_m) / 1000.0) if dist_m else None,
                    "moving_min": (float(time_s) / 60.0) if time_s else None,
                    "avg_hr": float(avg_hr) if avg_hr else None,
                    "source": "fit",
                }
            )
        return _df_standard(rows)

    return _df_standard([])


def parse_zip(file_bytes: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(BytesIO(file_bytes))
    frames = []
    for name in z.namelist():
        lname = name.lower()
        try:
            data = z.read(name)
        except Exception:
            continue

        if lname.endswith(".gpx"):
            frames.append(parse_gpx(data))
        elif lname.endswith(".tcx"):
            frames.append(parse_tcx(data))
        elif lname.endswith(".fit"):
            frames.append(parse_fit(data))
        elif lname.endswith(".csv"):
            f = parse_strava_csv(data)
            if not f.empty:
                frames.append(f)

    if not frames:
        return _df_standard([])
    return pd.concat(frames, ignore_index=True)


def parse_any_upload(uploaded_file) -> pd.DataFrame:
    b = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()

    if name.endswith(".zip"):
        return parse_zip(b)
    if name.endswith(".gpx"):
        return parse_gpx(b)
    if name.endswith(".tcx"):
        return parse_tcx(b)
    if name.endswith(".fit"):
        return parse_fit(b)
    if name.endswith(".csv"):
        return parse_strava_csv(b)

    return _df_standard([])

# ============================================================
# 5) Generación del plan
# ============================================================

@dataclass
class UserProfile:
    age: int
    height_cm: float
    weight_kg: float
    days_per_week: int
    goal: str
    modality: str
    start_date: date


def phase_for_week(week_idx: int) -> str:
    if week_idx <= 16:
        return "Base"
    if week_idx <= 32:
        return "Construcción"
    if week_idx <= 46:
        return "Específico"
    return "Afinado"


def weekly_volume_target(profile: UserProfile, week_idx: int, baseline_w: float) -> float:
    deload = (week_idx % 4 == 0)
    growth = 1.0 + 0.012 * week_idx

    if profile.modality == "correr":
        base = max(18.0, baseline_w) if baseline_w > 0 else 22.0
        cap = 70.0 if profile.goal == "maraton" else 55.0 if profile.goal == "media" else 45.0
    else:
        base = max(60.0, baseline_w) if baseline_w > 0 else 90.0
        cap = 300.0 if profile.goal == "gran_fondo" else 240.0 if profile.goal == "puerto" else 220.0 if profile.goal == "criterium" else 200.0

    target = min(base * growth, cap)
    if deload:
        target *= 0.78
    return round(target, 1)


def long_session_target(profile: UserProfile, week_idx: int, baseline_long: float) -> float:
    deload = (week_idx % 4 == 0)
    growth = 1.0 + 0.015 * week_idx

    if profile.modality == "correr":
        base = max(8.0, baseline_long * 0.85) if baseline_long > 0 else 10.0
        cap = 32.0 if profile.goal == "maraton" else 24.0 if profile.goal == "media" else 18.0 if profile.goal == "10k" else 20.0
    else:
        base = max(25.0, baseline_long * 0.85) if baseline_long > 0 else 40.0
        cap = 140.0 if profile.goal == "gran_fondo" else 110.0 if profile.goal == "puerto" else 90.0 if profile.goal == "criterium" else 100.0

    target = min(base * growth, cap)
    if deload:
        target *= 0.75
    return round(target, 1)


def workout_templates(phase: str, goal: str, hrmax: int, modality: str) -> Dict[str, Dict]:
    z = hr_zones(hrmax)
    if modality == "bicicleta":
        return {
            "easy": {"title": "Rodaje Z2 (bici)", "details": f"Z2 aprox {z['Z2'][0]}–{z['Z2'][1]} ppm. Cadencia cómoda. RPE 3–4/10."},
            "tempo": {"title": "Sweet spot (bici)", "details": "Calentar 15'. 3x10' fuerte pero sostenible (RPE 7/10) con 5' suave. Enfriar 10'."},
            "intervals": {"title": "VO2 (bici)", "details": "Calentar 15'. 6x3' muy fuerte (RPE 8–9/10) con 3' suave. Enfriar 10'."},
            "progressive": {"title": "Progresivo (bici)", "details": "60–90' empezando fácil y acabando 15–20' a ritmo sostenido (RPE 7/10)."},
            "long": {"title": "Salida larga (bici)", "details": "Mayormente Z2. Practica hidratación y comida cada 20–30'."},
            "recovery": {"title": "Recuperación (bici)", "details": "30–45' muy suave (Z1–Z2) + movilidad 10'."},
            "strength": {"title": "Fuerza + core", "details": "Fuerza total (pierna + tronco) y core. Cargas moderadas, técnica limpia."},
        }
    return {
        "easy": {"title": "Rodaje suave", "details": f"Z2 (aprox {z['Z2'][0]}–{z['Z2'][1]} ppm). Ritmo cómodo, respiración controlada."},
        "tempo": {"title": "Tempo", "details": f"Calentamiento 15'. 3x8' en Z3 (aprox {z['Z3'][0]}–{z['Z3'][1]} ppm) con 3' suave. Enfriar 10'."},
        "intervals": {"title": "Intervalos", "details": f"Calentamiento 15'. 6x3' en Z4 (aprox {z['Z4'][0]}–{z['Z4'][1]} ppm) con 2' suave. Enfriar 10'."},
        "progressive": {"title": "Progresivo", "details": "45–60' empezando en Z2 y acabando 10–15' en Z3."},
        "long": {"title": "Tirada larga", "details": "Mayormente Z2. Si fase Específico y objetivo maratón: últimos 20' en Z3 si te encuentras bien."},
        "recovery": {"title": "Recuperación", "details": "30–40' muy suave en Z1–Z2 + movilidad 10'."},
        "strength": {"title": "Fuerza + core", "details": "Fuerza total (pierna + tronco) y core. Cargas moderadas, técnica limpia."},
    }


def distribute_week_km(total_km: float, long_km: float, days: int) -> List[float]:
    days = max(3, int(days))
    remain = max(0.0, total_km - long_km)
    if days == 3:
        parts = [remain * 0.45, remain * 0.55]
    elif days == 4:
        parts = [remain * 0.25, remain * 0.30, remain * 0.45]
    elif days == 5:
        parts = [remain * 0.18, remain * 0.20, remain * 0.25, remain * 0.37]
    else:
        parts = [remain * 0.12, remain * 0.15, remain * 0.18, remain * 0.20, remain * 0.35]
    return [round(x, 1) for x in parts] + [round(long_km, 1)]


def build_plan(profile: UserProfile, baseline: Dict[str, float], hrmax: int) -> pd.DataFrame:
    start = next_monday(profile.start_date)
    rows = []
    for w in range(1, 53):
        phase = phase_for_week(w)
        w_km = weekly_volume_target(profile, w, baseline["w_km"])
        l_km = long_session_target(profile, w, baseline["long_km"])
        if l_km > 0.45 * w_km:
            w_km = round(l_km / 0.45, 1)

        km_sessions = distribute_week_km(w_km, l_km, profile.days_per_week)
        templates = workout_templates(phase, profile.goal, hrmax, profile.modality)
        week_start = start + timedelta(weeks=w - 1)

        if profile.days_per_week >= 6:
            dmap = {0: ("easy", km_sessions[0]), 1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[1]),
                    2: ("easy", km_sessions[2]), 3: ("tempo" if phase != "Base" else "progressive", km_sessions[3]),
                    4: ("recovery", max(4.0, km_sessions[4])), 5: ("easy", km_sessions[5] if len(km_sessions) > 5 else max(6.0, (w_km - l_km) * 0.2)),
                    6: ("long", km_sessions[-1])}
            strength_days = [2, 4]
        elif profile.days_per_week == 5:
            dmap = {0: ("easy", km_sessions[0]), 1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[1]),
                    3: ("tempo" if phase != "Base" else "progressive", km_sessions[2]), 5: ("easy", km_sessions[3]), 6: ("long", km_sessions[-1])}
            strength_days = [2, 4]
        elif profile.days_per_week == 4:
            dmap = {1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[0]),
                    3: ("easy", km_sessions[1]), 5: ("tempo" if phase != "Base" else "progressive", km_sessions[2]), 6: ("long", km_sessions[-1])}
            strength_days = [0, 2]
        else:
            dmap = {1: ("intervals" if phase in ("Construcción", "Específico") else "progressive", km_sessions[0]),
                    3: ("easy", km_sessions[1]), 6: ("long", km_sessions[-1])}
            strength_days = [0, 4]

        for dow in range(7):
            this_day = week_start + timedelta(days=dow)
            day_name = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][dow]
            modality_name = "Correr" if profile.modality == "correr" else "Bicicleta"

            if dow in dmap:
                wtype, km = dmap[dow]
                t = templates[wtype]
                extra = ""
                if wtype == "recovery":
                    extra = "Movilidad 10': " + "; ".join(MOBILITY_LIBRARY["movilidad_10"])
                if wtype in ("easy", "long") and phase == "Base" and (w % 2 == 1):
                    extra = (extra + " | " if extra else "") + "Core 10': " + "; ".join(CORE_LIBRARY["core_10"])
                rows.append({"Fecha": this_day, "Día": day_name, "Modalidad": modality_name, "Fase": phase,
                             "Sesión": t["title"], "Volumen (km)": float(km),
                             "Detalles": t["details"] + ((" | " + extra) if extra else "")})
            elif dow in strength_days:
                strength_key = "fuerza_35" if phase in ("Construcción", "Específico") else "fuerza_25"
                rows.append({"Fecha": this_day, "Día": day_name, "Modalidad": modality_name, "Fase": phase,
                             "Sesión": "Fuerza + core", "Volumen (km)": 0.0,
                             "Detalles": "Fuerza: " + "; ".join(STRENGTH_LIBRARY[strength_key]) + " | Core: " + "; ".join(CORE_LIBRARY["core_20"])})
            else:
                rows.append({"Fecha": this_day, "Día": day_name, "Modalidad": modality_name, "Fase": phase,
                             "Sesión": "Descanso / movilidad", "Volumen (km)": 0.0,
                             "Detalles": "Movilidad 20': " + "; ".join(MOBILITY_LIBRARY["movilidad_20"])})
    return pd.DataFrame(rows)


def plan_to_excel_bytes(plan: pd.DataFrame) -> bytes:
    plan = plan.copy()
    plan["Fecha"] = pd.to_datetime(plan["Fecha"])
    plan["Año"] = plan["Fecha"].dt.year
    plan["Mes"] = plan["Fecha"].dt.month

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for (y, m), chunk in plan.groupby(["Año", "Mes"], sort=True):
            month_name = chunk["Fecha"].dt.strftime("%B").iloc[0]
            sheet_name = f"{month_name[:28]}"
            chunk = chunk.sort_values("Fecha")[
                ["Fecha", "Día", "Modalidad", "Fase", "Sesión", "Volumen (km)", "Detalles"]
            ]
            chunk.to_excel(writer, index=False, sheet_name=sheet_name)
            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"
            widths = [12, 12, 12, 14, 22, 12, 80]
            for i, w in enumerate(widths, start=1):
                ws.column_dimensions[chr(64 + i)].width = w
    return output.getvalue()

# ============================================================
# 6) Feedback semanal
# ============================================================

def _week_start_monday(dt: pd.Timestamp) -> pd.Timestamp:
    d = dt.normalize()
    return d - pd.Timedelta(days=d.weekday())


def make_weekly_feedback(target_df: pd.DataFrame, modality: str, hrmax: int) -> str:
    if target_df.empty or target_df["start_dt"].isna().all():
        return "No hay datos suficientes para generar feedback semanal."

    df = target_df.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df = df[df["start_dt"].notna()].sort_values("start_dt")
    if df.empty:
        return "No hay datos suficientes para generar feedback semanal."

    df["week_start"] = df["start_dt"].apply(_week_start_monday)

    today = pd.Timestamp(date.today())
    this_week_start = _week_start_monday(today)
    complete_weeks = df[df["week_start"] < this_week_start]

    if complete_weeks.empty:
        window_start = today - pd.Timedelta(days=7)
        w = df[df["start_dt"] >= window_start]
        label = f"Últimos 7 días (desde {window_start.date()} hasta {today.date()})"
    else:
        last_week = complete_weeks["week_start"].max()
        w = df[df["week_start"] == last_week]
        label = f"Semana (lunes-domingo) desde {last_week.date()}"

    km = float(w["distance_km"].fillna(0).sum())
    sessions = int(w.shape[0])
    long_km = float(w["distance_km"].fillna(0).max()) if sessions else 0.0
    minutes = float(w["moving_min"].fillna(0).sum()) if "moving_min" in w.columns else 0.0

    avg_hr = None
    if "avg_hr" in w.columns and w["avg_hr"].notna().any():
        avg_hr = float(w["avg_hr"].dropna().mean())

    df_km_week = df.groupby("week_start")["distance_km"].sum().sort_index()
    prev4 = df_km_week[df_km_week.index < (w["week_start"].min() if "week_start" in w else this_week_start)].tail(4)
    prev4_avg = float(prev4.mean()) if len(prev4) else None

    good, warn, actions = [], [], []

    if sessions >= 4:
        good.append("Buena consistencia: 4 o más sesiones.")
    elif sessions >= 3:
        good.append("Consistencia correcta: 3 sesiones.")
    else:
        warn.append("Poca consistencia esta semana. Si puedes, intenta 3–4 sesiones.")

    if prev4_avg and prev4_avg > 0:
        change = (km - prev4_avg) / prev4_avg
        if change > 0.15:
            warn.append(f"El volumen ha subido bastante (+{change*100:.0f}% vs media de 4 semanas). Vigila la fatiga.")
            actions.append("Si notas fatiga, reduce 10–15% la próxima semana y prioriza Z2.")
        elif change < -0.15:
            warn.append(f"El volumen ha bajado (-{abs(change)*100:.0f}% vs media de 4 semanas).")
            actions.append("Si la bajada fue por falta de tiempo, vuelve poco a poco a tu media habitual.")
        else:
            good.append("Volumen estable respecto a tu media reciente.")

    if modality == "correr":
        if long_km >= 18:
            good.append("Tirada larga sólida para construir resistencia.")
        elif long_km >= 12:
            good.append("Tirada larga correcta para tu base actual.")
        else:
            actions.append("Incluye una tirada larga suave semanal (Z2) y súbela poco a poco.")
    else:
        if long_km >= 80:
            good.append("Salida larga sólida para base ciclista.")
        elif long_km >= 50:
            good.append("Salida larga correcta para tu base actual.")
        else:
            actions.append("Incluye una salida larga Z2 y practica hidratación/comida.")

    if avg_hr is not None:
        z = hr_zones(hrmax)
        if avg_hr >= z["Z3"][0]:
            warn.append("La FC media de la semana es alta para construir base. Puede que estés yendo demasiado fuerte.")
            actions.append("Haz rodajes en Z2 y deja la intensidad para 1 día (tempo/intervalos).")
        else:
            good.append("FC media compatible con trabajo aeróbico (base).")

    if not actions:
        actions = [
            "Mantén 1 sesión de calidad (tempo/intervalos) y el resto fácil.",
            "Asegura 1–2 sesiones de fuerza + core (20–35').",
            "Prioriza sueño e hidratación, especialmente si subes volumen.",
        ]

    txt = f"""### Feedback semanal

**Periodo analizado:** {label}  
**Hechos:** {sessions} sesiones · {km:.1f} km · {minutes:.0f} min · sesión más larga {long_km:.1f} km"""
    if avg_hr is not None:
        txt += f" · FC media aprox {avg_hr:.0f} ppm\n"
    else:
        txt += "\n"

    if good:
        txt += "\n**Lo que va bien**\n"
        for g in good:
            txt += f"- {g}\n"

    if warn:
        txt += "\n**Ajustes o alertas**\n"
        for w0 in warn:
            txt += f"- {w0}\n"

    txt += "\n**Próximos pasos (1–3)**\n"
    for i, a in enumerate(actions[:3], start=1):
        txt += f"{i}. {a}\n"
    return txt

# ============================================================
# 7) App Streamlit
# ============================================================

st.set_page_config(page_title="Planificador multi-plataforma", layout="wide")
st.title("V2. Planificador de entrenamiento (Strava / Garmin / Komoot / adidas / otros)")

st.info("Sube tus actividades (CSV de Strava, GPX/TCX/FIT o ZIP con varias actividades) y genera un plan anual en Excel.")

with st.expander("¿No sabes cómo descargar tu CSV de Strava?"):
    st.markdown(
        """
1. Inicia sesión en Strava desde un ordenador y haz clic en tu foto de perfil. Selecciona **Ajustes**.  
2. En el menú lateral izquierdo, entra en **Mi cuenta**.  
3. Accede a **Descarga o elimina tu cuenta** y pulsa en **Solicita tu archivo**.  
4. Strava te enviará un correo con un archivo .zip. Descárgalo y descomprímelo.  
5. **Solo necesitas el archivo `activities.csv`** para subirlo aquí.
"""
    )

uploaded = st.file_uploader(
    "Sube un archivo: CSV (Strava), GPX/TCX/FIT (Garmin/Komoot/otros) o ZIP (export completo)",
    type=["csv", "gpx", "tcx", "fit", "zip"],
)

with st.sidebar:
    st.header("Modalidad")
    modality = st.selectbox("Selecciona modalidad", options=["correr", "bicicleta"])

    st.header("Perfil")
    age = st.number_input("Edad", min_value=12, max_value=90, value=36, step=1)
    height_cm = st.number_input("Altura (cm)", min_value=120, max_value=220, value=180, step=1)
    weight_kg = st.number_input("Peso (kg)", min_value=35.0, max_value=200.0, value=80.0, step=0.5)

    st.header("Objetivo")
    goal = st.selectbox(
        "Selecciona objetivo principal",
        options=GOALS[modality],
        format_func=lambda x: x[1],
    )[0]

    days_per_week = st.slider("Días de entrenamiento por semana", min_value=3, max_value=6, value=5)
    start_date = st.date_input("Fecha de inicio", value=date.today())

if uploaded is None:
    st.stop()

activities = parse_any_upload(uploaded)
if activities.empty:
    st.error("No he podido leer actividades del archivo. Prueba con CSV de Strava, GPX/TCX/FIT o ZIP con actividades.")
    st.stop()

# Clasificar 'other' si hace falta (GPX y algunos zips)
if (activities["sport"] == "other").any():
    with st.expander("Clasificación de actividades sin deporte (solo si hace falta)"):
        st.write("Algunos formatos (como GPX) no indican si es correr o bici. Puedes asignarlo aquí.")
        assign_other = st.selectbox(
            "Asignar actividades 'other' como:",
            options=["no cambiar", "correr", "bicicleta"],
            index=0,
        )
        if assign_other != "no cambiar":
            activities.loc[activities["sport"] == "other", "sport"] = "run" if assign_other == "correr" else "ride"

sport_needed = "run" if modality == "correr" else "ride"
target = activities[activities["sport"] == sport_needed].copy()

if target.empty:
    st.warning(
        "No he encontrado actividades de la modalidad seleccionada. "
        "Si has subido GPX, abre el desplegable de clasificación y asigna 'other'."
    )

# Para baseline
target["date_only"] = target["start_dt"]
baseline = weekly_baseline(target.assign(date_only=target["start_dt"])) if not target.empty else {"w_km": 0.0, "long_km": 0.0, "sessions_w": 0.0}

hrmax = estimate_hrmax(int(age))

st.subheader("Lectura rápida de tus datos")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Actividades detectadas (modalidad)", int(target.shape[0]))
c2.metric("Km/semana (media ~6 semanas)", f"{baseline['w_km']:.1f}")
c3.metric("Sesión más larga aprox", f"{baseline['long_km']:.1f} km")
c4.metric("HRmáx estimada (aprox)", f"{hrmax} ppm")

st.subheader("Feedback")
if st.button("Generar feedback semanal"):
    # feedback trabaja con start_dt
    st.markdown(make_weekly_feedback(target, modality, hrmax))

st.subheader("Plan anual")
profile = UserProfile(
    age=int(age),
    height_cm=float(height_cm),
    weight_kg=float(weight_kg),
    days_per_week=int(days_per_week),
    goal=str(goal),
    modality=str(modality),
    start_date=start_date,
)

plan = build_plan(profile, baseline, hrmax)
st.dataframe(plan.head(21), use_container_width=True)

excel_bytes = plan_to_excel_bytes(plan)
st.download_button(
    label="Descargar plan anual en Excel",
    data=excel_bytes,
    file_name=f"plan_entrenamiento_anual_{modality}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
