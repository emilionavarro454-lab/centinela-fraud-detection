# CENTINELA - API para solicitudes de apertura de cuenta
# Ejecucion: uvicorn app:app --reload

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

RUTA_ARTEFACTO = Path(__file__).resolve().parents[1] / "modelos" / "centinela_v1.joblib"
ARTE = joblib.load(RUTA_ARTEFACTO)
EXPLICADOR = shap.TreeExplainer(ARTE["modelo"])


class Solicitud(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: Literal[0, 1]
    housing_status: str
    phone_home_valid: Literal[0, 1]
    phone_mobile_valid: Literal[0, 1]
    bank_months_count: int
    has_other_cards: Literal[0, 1]
    proposed_credit_limit: float
    foreign_request: Literal[0, 1]
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: Literal[0, 1]
    device_distinct_emails_8w: int


def construir_features(datos):
    d = datos.copy()

    d["velocity_6h"] = d["velocity_6h"].clip(lower=0)

    d["ratio_velocidad_6h_24h"] = (
        d["velocity_6h"] / d["velocity_24h"]
    ).astype("float32")

    d["ratio_velocidad_24h_4w"] = (
        d["velocity_24h"] / d["velocity_4w"]
    ).astype("float32")

    d["presion_zip"] = (
        d["zip_count_4w"] / d["velocity_4w"]
    ).astype("float32")

    for col in [
        "ratio_velocidad_6h_24h",
        "ratio_velocidad_24h_4w",
        "presion_zip"
    ]:
        d[col] = d[col].replace([np.inf, -np.inf], np.nan)

    d["telefonos_validos"] = (
        d["phone_home_valid"] + d["phone_mobile_valid"]
    ).astype("int8")

    d["similitud_baja"] = (
        d["name_email_similarity"] < 0.2
    ).astype("int8")

    d["limite_alto"] = (
        d["proposed_credit_limit"] >= 1500
    ).astype("int8")

    d["dispositivo_multiemail"] = (
        d["device_distinct_emails_8w"] >= 2
    ).astype("int8")

    ausencia = {
        "prev_address_months_count":
            d["prev_address_months_count"] == -1,

        "current_address_months_count":
            d["current_address_months_count"] == -1,

        "bank_months_count":
            d["bank_months_count"] == -1,

        "session_length_in_minutes":
            d["session_length_in_minutes"] == -1,

        "device_distinct_emails_8w":
            d["device_distinct_emails_8w"] == -1,

        "intended_balcon_amount":
            d["intended_balcon_amount"] < 0,
    }

    for col, falta in ausencia.items():
        d[col + "_ausente"] = falta.astype("int8")
        d[col] = d[col].astype("float32").where(~falta)

    flags = [c for c in d.columns if c.endswith("_ausente")]
    d["n_faltantes"] = d[flags].sum(axis=1).astype("int8")

    return d


app = FastAPI(
    title="CENTINELA",
    description="Decision de riesgo en apertura de cuentas",
    version="1.0"
)


@app.get("/salud")
def salud():
    return {
        "estado": "operativo",
        "modelo": "centinela_v1",
        "umbral_operativo": ARTE["umbral_operativo"],
        "entrenamiento": ARTE["esquema_temporal"]
    }


@app.post("/predecir")
def predecir(solicitud: Solicitud):
    datos = solicitud.model_dump()

    fila = construir_features(
        pd.DataFrame([datos])
    )[ARTE["columnas"]]

    X = ARTE["preprocesador"].transform(fila)

    prob = float(
        ARTE["calibrador"].predict(
            ARTE["modelo"].predict_proba(X)[:, 1]
        )[0]
    )

    decision = (
        "REVISION_MANUAL"
        if prob >= ARTE["umbral_operativo"]
        else "APROBACION"
    )

    contrib = pd.Series(
        EXPLICADOR(X).values[0],
        index=X.columns
    )

    top = contrib.sort_values(
        key=lambda s: s.abs(),
        ascending=False
    ).head(3)

    factores = [
        {
            "variable": k,
            "efecto": round(float(v), 3),
            "direccion": (
                "aumenta_riesgo"
                if v > 0
                else "reduce_riesgo"
            )
        }
        for k, v in top.items()
    ]

    exposicion = float(
        datos["proposed_credit_limit"]
    )

    return {
        "probabilidad_fraude": round(prob, 4),
        "decision": decision,
        "umbral_operativo": ARTE["umbral_operativo"],
        "exposicion_si_aprueba_usd": exposicion,
        "coste_esperado_si_aprueba_usd": round(
            prob * exposicion,
            2
        ),
        "coste_revision_usd": ARTE["coste_revision_$"],
        "factores_principales": factores,
    }
