"""
SwiftDrive - Módulo de Inteligencia Artificial.

Clasificador supervisado (DecisionTreeClassifier) que predice si el tráfico
FLUVIAL de una operación portuaria será ALTO (1) o BAJO (0) a partir de:
    Zona portuaria, Sociedad portuaria, Tipo de carga, Año y Mes.

Fuente de datos: "Tráfico Portuario Marítimo en Colombia" - SUPERTRANSPORTE
(datos.gov.co, dataset 5r3g-zv5z).

Uso por consola:
    python Backend/modelo_ia.py                 # entrena, evalúa y guarda el modelo
    python Backend/modelo_ia.py --umbral mediana_positivos
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
RUTA_CSV = BASE_DIR / "data" / "Trafico_Portuario_Maritimo_En_Colombia_20261018.csv"
RUTA_MODELO = BASE_DIR / "modelo_swiftdrive.joblib"

COL_ZONA = "ZONA PORTUARIA"
COL_SOCIEDAD = "SOCIEDAD PORTUARIA"
COL_CARGA = "TIPO DE CARGA"
COL_ANIO = "AÑO VIGENCIA"
COL_MES = "MES VIGENCIA"
COL_OBJETIVO_CONTINUO = "FLUVIAL"
COL_OBJETIVO = "Nivel_Fluvial"

COLUMNAS_CATEGORICAS = [COL_ZONA, COL_SOCIEDAD, COL_CARGA]
COLUMNAS_NUMERICAS = [COL_ANIO, COL_MES]
FEATURES = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS

# El CSV de datos.gov.co trae algunos caracteres dañados (U+FFFD) en nombres.
CORRECCIONES_TEXTO = {
    "COMPA�IA DE PUERTOS ASOCIADOS S.A.": "COMPAÑIA DE PUERTOS ASOCIADOS S.A.",
    "CENIT TRANSPORTE Y LOG�STICA DE HIDROCARBUROS SAS": "CENIT TRANSPORTE Y LOGÍSTICA DE HIDROCARBUROS SAS",
    "C.I. UNI�N DE BANANEROS DE URABA S.A. - UNIBAN": "C.I. UNIÓN DE BANANEROS DE URABA S.A. - UNIBAN",
}

ESTRATEGIAS_UMBRAL = ("mediana", "mediana_positivos")


# --------------------------------------------------------------------------- #
# 1. Carga y preprocesamiento
# --------------------------------------------------------------------------- #
def cargar_datos(ruta_csv: str | Path = RUTA_CSV) -> pd.DataFrame:
    """Lee el CSV oficial y lo deja limpio.

    IMPORTANTE: el archivo usa el punto como separador de MILES
    ("1.389" = mil trescientos ochenta y nueve, "2.023" = año 2023).
    Por eso se usa ``thousands='.'``; sin él pandas lo interpreta como decimal.
    """
    df = pd.read_csv(ruta_csv, thousands=".", encoding="utf-8")
    df.columns = df.columns.str.strip()

    # Texto: quitar espacios sobrantes y corregir caracteres dañados.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().replace(CORRECCIONES_TEXTO)

    # Imputación de nulos con la moda (según la metodología del informe).
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Asegurar tipos numéricos enteros en año y mes.
    df[COL_ANIO] = df[COL_ANIO].astype(int)
    df[COL_MES] = df[COL_MES].astype(int)
    return df


def calcular_umbral(serie: pd.Series, estrategia: str = "mediana") -> float:
    """Calcula el umbral que separa tráfico ALTO de BAJO.

    - ``mediana``: mediana de todos los registros (lo descrito en el informe).
      En este dataset la mediana es 0, por lo que la clase 1 equivale a
      "el registro tiene tráfico fluvial (> 0)".
    - ``mediana_positivos``: mediana solo de los registros con tráfico > 0.
      La clase 1 pasa a ser "tráfico fluvial por encima de lo habitual".
    """
    if estrategia == "mediana":
        return float(serie.median())
    if estrategia == "mediana_positivos":
        positivos = serie[serie > 0]
        return float(positivos.median()) if len(positivos) else 0.0
    raise ValueError(f"Estrategia de umbral no válida: {estrategia}. Use {ESTRATEGIAS_UMBRAL}.")


def crear_objetivo(df: pd.DataFrame, umbral: float) -> pd.Series:
    """Binariza FLUVIAL: 1 = ALTO (> umbral), 0 = BAJO (<= umbral)."""
    return pd.Series(np.where(df[COL_OBJETIVO_CONTINUO] > umbral, 1, 0), index=df.index, name=COL_OBJETIVO)


# --------------------------------------------------------------------------- #
# 2. Modelo
# --------------------------------------------------------------------------- #
def construir_pipeline(random_state: int = 42) -> Pipeline:
    """Codificación de texto a número + Árbol de Decisión en un solo objeto.

    OrdinalEncoder cumple la misma función que LabelEncoder del notebook,
    pero queda guardado junto al modelo, así la predicción usa exactamente
    la misma codificación que el entrenamiento.
    """
    preprocesador = ColumnTransformer(
        transformers=[
            ("categoricas", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), COLUMNAS_CATEGORICAS),
            ("numericas", "passthrough", COLUMNAS_NUMERICAS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("arbol", DecisionTreeClassifier(random_state=random_state)),
        ]
    )


@dataclass
class ResultadoEntrenamiento:
    pipeline: Pipeline
    umbral: float
    estrategia_umbral: str
    accuracy: float
    reporte: str
    reporte_dict: dict
    matriz_confusion: list
    distribucion_clases: dict
    n_train: int
    n_test: int
    catalogo: dict = field(default_factory=dict)
    importancias: dict = field(default_factory=dict)


def construir_catalogo(df: pd.DataFrame) -> dict:
    """Valores válidos de cada variable, para validar entradas y armar la UI."""
    sociedades_por_zona = (
        df.groupby(COL_ZONA)[COL_SOCIEDAD].apply(lambda s: sorted(s.unique())).to_dict()
    )
    return {
        "zonas": sorted(df[COL_ZONA].unique()),
        "sociedades": sorted(df[COL_SOCIEDAD].unique()),
        "cargas": sorted(df[COL_CARGA].unique()),
        "sociedades_por_zona": sociedades_por_zona,
        "anio_min": int(df[COL_ANIO].min()),
        "anio_max": int(df[COL_ANIO].max()),
    }


def entrenar(
    ruta_csv: str | Path = RUTA_CSV,
    estrategia_umbral: str = "mediana",
    test_size: float = 0.2,
    random_state: int = 42,
) -> ResultadoEntrenamiento:
    """Ejecuta el flujo completo: carga -> objetivo -> partición 80/20 -> entrenamiento -> evaluación."""
    df = cargar_datos(ruta_csv)
    umbral = calcular_umbral(df[COL_OBJETIVO_CONTINUO], estrategia_umbral)
    y = crear_objetivo(df, umbral)
    X = df[FEATURES]

    # stratify mantiene la misma proporción de clases en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = construir_pipeline(random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    nombres = ["BAJO (0)", "ALTO (1)"]
    importancias = dict(
        zip(FEATURES, pipeline.named_steps["arbol"].feature_importances_.round(4).tolist())
    )

    return ResultadoEntrenamiento(
        pipeline=pipeline,
        umbral=umbral,
        estrategia_umbral=estrategia_umbral,
        accuracy=float(accuracy_score(y_test, y_pred)),
        reporte=classification_report(y_test, y_pred, target_names=nombres, zero_division=0),
        reporte_dict=classification_report(y_test, y_pred, target_names=nombres, zero_division=0, output_dict=True),
        matriz_confusion=confusion_matrix(y_test, y_pred).tolist(),
        distribucion_clases={int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        n_train=len(X_train),
        n_test=len(X_test),
        catalogo=construir_catalogo(df),
        importancias=importancias,
    )


# --------------------------------------------------------------------------- #
# 3. Persistencia
# --------------------------------------------------------------------------- #
def guardar_modelo(resultado: ResultadoEntrenamiento, ruta: str | Path = RUTA_MODELO) -> Path:
    joblib.dump(resultado, ruta)
    return Path(ruta)


def cargar_modelo(ruta: str | Path = RUTA_MODELO) -> ResultadoEntrenamiento:
    return joblib.load(ruta)


def obtener_modelo(estrategia_umbral: str = "mediana", ruta: str | Path = RUTA_MODELO) -> ResultadoEntrenamiento:
    """Carga el modelo guardado si coincide la estrategia; si no, lo entrena y guarda."""
    ruta = Path(ruta)
    if ruta.exists():
        try:
            modelo = cargar_modelo(ruta)
            if modelo.estrategia_umbral == estrategia_umbral:
                return modelo
        except Exception:
            pass  # archivo de otra versión: se reentrena
    modelo = entrenar(estrategia_umbral=estrategia_umbral)
    guardar_modelo(modelo, ruta)
    return modelo


# --------------------------------------------------------------------------- #
# 4. Predicción de nuevos escenarios
# --------------------------------------------------------------------------- #
class EntradaInvalida(ValueError):
    """Se lanza cuando un valor de entrada no existe en los datos de entrenamiento."""


def validar_entrada(modelo: ResultadoEntrenamiento, zona: str, sociedad: str, carga: str, anio: int, mes: int) -> None:
    cat = modelo.catalogo
    errores = []
    if zona not in cat["zonas"]:
        errores.append(f"Zona '{zona}' no existe. Opciones: {', '.join(cat['zonas'])}")
    if sociedad not in cat["sociedades"]:
        errores.append(f"Sociedad '{sociedad}' no existe en el dataset.")
    elif zona in cat["sociedades_por_zona"] and sociedad not in cat["sociedades_por_zona"][zona]:
        errores.append(f"La sociedad '{sociedad}' no opera en la zona '{zona}'.")
    if carga not in cat["cargas"]:
        errores.append(f"Tipo de carga '{carga}' no existe. Opciones: {', '.join(cat['cargas'])}")
    if not 1 <= int(mes) <= 12:
        errores.append(f"Mes '{mes}' fuera de rango (1-12).")
    if not 2000 <= int(anio) <= 2100:
        errores.append(f"Año '{anio}' no es válido (use 4 dígitos, p. ej. 2026).")
    if errores:
        raise EntradaInvalida(" | ".join(errores))


def predecir_trafico(
    zona: str,
    sociedad: str,
    carga: str,
    anio: int,
    mes: int,
    modelo: ResultadoEntrenamiento | None = None,
) -> dict:
    """Predice el nivel de tráfico fluvial para un escenario nuevo.

    A diferencia de la versión del notebook, NO reemplaza en silencio
    los valores desconocidos: si un valor no existe lanza ``EntradaInvalida``.
    """
    modelo = modelo or obtener_modelo()
    zona, sociedad, carga = zona.strip().upper(), sociedad.strip(), carga.strip().upper()
    validar_entrada(modelo, zona, sociedad, carga, anio, mes)

    nuevo = pd.DataFrame([[zona, sociedad, carga, int(anio), int(mes)]], columns=FEATURES)
    clase = int(modelo.pipeline.predict(nuevo)[0])
    probabilidades = modelo.pipeline.predict_proba(nuevo)[0]
    return {
        "clase": clase,
        "nivel": "ALTO" if clase == 1 else "BAJO",
        "probabilidad_alto": float(probabilidades[1]) if len(probabilidades) > 1 else float(clase),
        "umbral": modelo.umbral,
    }


# --------------------------------------------------------------------------- #
# 5. Comparación de modelos y evaluación temporal
# --------------------------------------------------------------------------- #
MODELOS_COMPARACION = (
    "Árbol de decisión (modelo del informe)",
    "Árbol de decisión (profundidad máx. 8)",
    "Random Forest (200 árboles)",
    "Regresión logística",
)
ANIO_CORTE_DEFECTO = 2024


def _preprocesador_ordinal() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categoricas", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), COLUMNAS_CATEGORICAS),
            ("numericas", "passthrough", COLUMNAS_NUMERICAS),
        ]
    )


def construir_modelo_comparacion(nombre: str, random_state: int = 42) -> Pipeline:
    """Crea el pipeline de uno de los modelos de ``MODELOS_COMPARACION``."""
    if nombre == MODELOS_COMPARACION[0]:
        return Pipeline([("pre", _preprocesador_ordinal()), ("clf", DecisionTreeClassifier(random_state=random_state))])
    if nombre == MODELOS_COMPARACION[1]:
        return Pipeline(
            [("pre", _preprocesador_ordinal()), ("clf", DecisionTreeClassifier(max_depth=8, random_state=random_state))]
        )
    if nombre == MODELOS_COMPARACION[2]:
        return Pipeline(
            [("pre", _preprocesador_ordinal()), ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))]
        )
    if nombre == MODELOS_COMPARACION[3]:
        # La regresión logística necesita one-hot y variables escaladas.
        pre = ColumnTransformer(
            [
                ("categoricas", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),
                ("numericas", StandardScaler(), COLUMNAS_NUMERICAS),
            ]
        )
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
    raise ValueError(f"Modelo desconocido: {nombre}")


def _metricas(y_real, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_real, y_pred),
        "Precisión (ALTO)": precision_score(y_real, y_pred, zero_division=0),
        "Recall (ALTO)": recall_score(y_real, y_pred, zero_division=0),
        "F1 (ALTO)": f1_score(y_real, y_pred, zero_division=0),
    }


def comparar_modelos(
    estrategia_umbral: str = "mediana",
    ruta_csv: str | Path = RUTA_CSV,
    anio_corte: int = ANIO_CORTE_DEFECTO,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compara los modelos con dos formas de evaluar.

    1. **Aleatoria 80/20** (estratificada): la del informe. Mezcla años, por lo que
       el modelo "ve" meses de 2026 al entrenar.
    2. **Temporal**: entrena con los años <= ``anio_corte`` y prueba con los
       posteriores. Simula el uso real: predecir periodos que aún no se conocen.
       El umbral se calcula solo con los datos de entrenamiento.
    """
    df = cargar_datos(ruta_csv)
    filas = []

    # --- Evaluación aleatoria ---
    umbral = calcular_umbral(df[COL_OBJETIVO_CONTINUO], estrategia_umbral)
    y = crear_objetivo(df, umbral)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], y, test_size=0.2, random_state=random_state, stratify=y
    )
    nombre_aleatoria = "Aleatoria 80/20"
    for nombre in MODELOS_COMPARACION:
        modelo = construir_modelo_comparacion(nombre, random_state).fit(X_tr, y_tr)
        filas.append({"Modelo": nombre, "Evaluación": nombre_aleatoria, **_metricas(y_te, modelo.predict(X_te))})

    # --- Evaluación temporal ---
    train = df[df[COL_ANIO] <= anio_corte]
    test = df[df[COL_ANIO] > anio_corte]
    nombre_temporal = f"Temporal (≤{anio_corte} → {anio_corte + 1}+)"
    if len(train) and len(test):
        umbral_t = calcular_umbral(train[COL_OBJETIVO_CONTINUO], estrategia_umbral)
        y_tr_t, y_te_t = crear_objetivo(train, umbral_t), crear_objetivo(test, umbral_t)
        for nombre in MODELOS_COMPARACION:
            modelo = construir_modelo_comparacion(nombre, random_state).fit(train[FEATURES], y_tr_t)
            filas.append(
                {"Modelo": nombre, "Evaluación": nombre_temporal, **_metricas(y_te_t, modelo.predict(test[FEATURES]))}
            )
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# 6. Interpretabilidad del árbol
# --------------------------------------------------------------------------- #
def codigos_categorias(modelo: ResultadoEntrenamiento) -> dict:
    """Código numérico que el árbol asigna a cada categoría (0, 1, 2, ...)."""
    codificador = modelo.pipeline.named_steps["preprocesamiento"].named_transformers_["categoricas"]
    return {col: list(cats) for col, cats in zip(COLUMNAS_CATEGORICAS, codificador.categories_)}


def info_arbol(modelo: ResultadoEntrenamiento) -> dict:
    arbol = modelo.pipeline.named_steps["arbol"]
    return {"profundidad": int(arbol.get_depth()), "hojas": int(arbol.get_n_leaves())}


def reglas_arbol(modelo: ResultadoEntrenamiento, max_depth: int = 3) -> str:
    """Reglas del árbol en texto (solo los primeros niveles)."""
    return export_text(
        modelo.pipeline.named_steps["arbol"], feature_names=FEATURES, max_depth=max_depth
    )


def figura_arbol(modelo: ResultadoEntrenamiento, max_depth: int = 3):
    """Dibuja los primeros niveles del árbol (matplotlib). Devuelve la figura."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(
        modelo.pipeline.named_steps["arbol"],
        feature_names=FEATURES,
        class_names=["BAJO", "ALTO"],
        max_depth=max_depth,
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=False,
        ax=ax,
    )
    ax.set_title(f"Primeros {max_depth} niveles del árbol de decisión (el árbol completo es mucho más profundo)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Ejecución por consola
# --------------------------------------------------------------------------- #
def _main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y evalúa el modelo SwiftDrive.")
    parser.add_argument("--umbral", choices=ESTRATEGIAS_UMBRAL, default="mediana")
    parser.add_argument("--comparar", action="store_true", help="compara modelos (aleatoria y temporal)")
    args = parser.parse_args()

    resultado = entrenar(estrategia_umbral=args.umbral)
    ruta = guardar_modelo(resultado)

    print("=== SwiftDrive - Modelo de clasificación de tráfico fluvial ===")
    print(f"Estrategia de umbral: {resultado.estrategia_umbral} -> umbral = {resultado.umbral:,.0f} t")
    print(f"Distribución de clases (0=BAJO, 1=ALTO): {resultado.distribucion_clases}")
    print(f"Entrenamiento: {resultado.n_train} registros | Prueba: {resultado.n_test} registros")
    print(f"Accuracy: {resultado.accuracy:.4f}\n")
    print(resultado.reporte)
    print("Matriz de confusión [[VN, FP], [FN, VP]]:", resultado.matriz_confusion)
    print("Importancia de variables:", resultado.importancias)
    print(f"\nModelo guardado en: {ruta}")

    ejemplo = predecir_trafico(
        "BARRANCABERMEJA",
        "ECOPETROL S.A.",
        "GRANEL LIQUIDO",
        2026,
        6,
        modelo=resultado,
    )
    if args.comparar:
        print("\n=== Comparación de modelos ===")
        print(comparar_modelos(args.umbral).round(3).to_string(index=False))

    print("\nEjemplo -> Barrancabermeja / Ecopetrol / Granel líquido / 2026-06:", ejemplo)


if __name__ == "__main__":
    _main()
