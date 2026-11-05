"""
SwiftDrive - Interfaz web (Streamlit).

Ejecutar desde la raíz del repositorio:
    streamlit run Frontend/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permitir importar el módulo del Backend
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "Backend"))

import modelo_ia  # noqa: E402

MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

st.set_page_config(page_title="SwiftDrive", page_icon="⚓", layout="wide")


@st.cache_resource(show_spinner="Entrenando el modelo...")
def obtener_modelo(estrategia: str):
    return modelo_ia.obtener_modelo(estrategia)


@st.cache_data(show_spinner="Comparando modelos (puede tardar unos segundos)...")
def obtener_comparacion(estrategia: str):
    return modelo_ia.comparar_modelos(estrategia)


@st.cache_data
def obtener_datos():
    return modelo_ia.cargar_datos()


# --------------------------------------------------------------------------- #
# Barra lateral: configuración del modelo
# --------------------------------------------------------------------------- #
st.sidebar.title("⚓ SwiftDrive")
st.sidebar.caption("Predicción del nivel de tráfico fluvial en las sociedades portuarias de Colombia.")

etiquetas_umbral = {
    "mediana": "Mediana de todos los registros (informe)",
    "mediana_positivos": "Mediana de registros con tráfico > 0",
}
estrategia = st.sidebar.radio(
    "Umbral para definir tráfico ALTO",
    options=list(etiquetas_umbral),
    format_func=etiquetas_umbral.get,
)
modelo = obtener_modelo(estrategia)

if modelo.umbral == 0:
    st.sidebar.info(
        "La mediana de FLUVIAL es 0 t: con este umbral, **ALTO** significa "
        "que la operación *tiene* tráfico fluvial."
    )
else:
    st.sidebar.info(f"**ALTO** significa más de **{modelo.umbral:,.0f} t** de tráfico fluvial.")

st.sidebar.markdown(
    "**Fuente:** Superintendencia de Transporte — "
    "[datos.gov.co](https://www.datos.gov.co/Transporte/Trafico-Portuario-Mar-timo-En-Colombia/5r3g-zv5z/about_data)"
)

tab_pred, tab_metricas, tab_comp, tab_arbol, tab_datos = st.tabs(
    ["🔮 Predicción", "📊 Desempeño del modelo", "🧪 Comparación de modelos", "🌳 Árbol de decisión", "🗂️ Datos"]
)

# --------------------------------------------------------------------------- #
# Pestaña 1: Predicción
# --------------------------------------------------------------------------- #
with tab_pred:
    st.header("Predecir el nivel de tráfico fluvial")
    cat = modelo.catalogo

    col1, col2 = st.columns(2)
    with col1:
        zona = st.selectbox("Zona portuaria", cat["zonas"])
        sociedad = st.selectbox(
            "Sociedad portuaria",
            cat["sociedades_por_zona"].get(zona, cat["sociedades"]),
            help="Solo se muestran las sociedades que operan en la zona elegida.",
        )
        carga = st.selectbox("Tipo de carga", cat["cargas"])
    with col2:
        anio = st.number_input(
            "Año", min_value=cat["anio_min"], max_value=cat["anio_max"] + 5,
            value=cat["anio_max"], step=1,
        )
        mes = st.selectbox("Mes", range(1, 13), format_func=lambda m: MESES[m - 1])

    if st.button("Predecir", type="primary"):
        try:
            r = modelo_ia.predecir_trafico(zona, sociedad, carga, int(anio), int(mes), modelo=modelo)
        except modelo_ia.EntradaInvalida as e:
            st.error(str(e))
        else:
            if r["clase"] == 1:
                st.error(f"### Tráfico fluvial ALTO\nProbabilidad estimada: **{r['probabilidad_alto']:.0%}**")
                st.write("Recomendación: reforzar la planificación de zarpes, personal y protocolos de seguridad.")
            else:
                st.success(f"### Tráfico fluvial BAJO\nProbabilidad de ALTO: **{r['probabilidad_alto']:.0%}**")
                st.write("Operación con demanda fluvial baja: capacidad normal es suficiente.")
            if int(anio) > cat["anio_max"]:
                st.warning(
                    f"El dataset llega hasta {cat['anio_max']}. Para años posteriores el árbol "
                    "extrapola usando el último año conocido."
                )

        # Historial real del mismo escenario, para contrastar.
        df = obtener_datos()
        hist = df[
            (df[modelo_ia.COL_ZONA] == zona)
            & (df[modelo_ia.COL_SOCIEDAD] == sociedad)
            & (df[modelo_ia.COL_CARGA] == carga)
        ]
        if not hist.empty:
            st.subheader("Histórico real de esta sociedad y tipo de carga")
            serie = (
                hist.assign(Fecha=pd.to_datetime(dict(year=hist[modelo_ia.COL_ANIO], month=hist[modelo_ia.COL_MES], day=1)))
                .groupby("Fecha")[modelo_ia.COL_OBJETIVO_CONTINUO].sum()
                .rename("Tráfico fluvial (t)")
            )
            st.line_chart(serie)
        else:
            st.caption("No hay registros históricos para esta combinación de sociedad y tipo de carga.")

# --------------------------------------------------------------------------- #
# Pestaña 2: Métricas
# --------------------------------------------------------------------------- #
with tab_metricas:
    st.header("Desempeño en el conjunto de prueba (20 %)")
    rep = modelo.reporte_dict
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{modelo.accuracy:.1%}")
    m2.metric("Precisión (ALTO)", f"{rep['ALTO (1)']['precision']:.1%}")
    m3.metric("Recall (ALTO)", f"{rep['ALTO (1)']['recall']:.1%}")
    m4.metric("F1 (ALTO)", f"{rep['ALTO (1)']['f1-score']:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matriz de confusión")
        st.dataframe(
            pd.DataFrame(
                modelo.matriz_confusion,
                index=["Real BAJO", "Real ALTO"],
                columns=["Predicho BAJO", "Predicho ALTO"],
            )
        )
        st.subheader("Distribución de clases")
        st.dataframe(
            pd.DataFrame(
                {"Registros": [modelo.distribucion_clases.get(0, 0), modelo.distribucion_clases.get(1, 0)]},
                index=["BAJO (0)", "ALTO (1)"],
            )
        )
    with c2:
        st.subheader("Importancia de variables")
        st.bar_chart(pd.Series(modelo.importancias, name="Importancia").sort_values(ascending=False))

    st.subheader("Reporte de clasificación")
    st.code(modelo.reporte)
    st.caption(f"Entrenamiento: {modelo.n_train} registros · Prueba: {modelo.n_test} registros · DecisionTreeClassifier")


# --------------------------------------------------------------------------- #
# Pestaña 3: Comparación de modelos
# --------------------------------------------------------------------------- #
with tab_comp:
    st.header("Comparación de modelos")
    st.write(
        "Se entrenan cuatro modelos con los mismos datos y se evalúan de dos formas:\n"
        "- **Aleatoria 80/20:** la del informe. Mezcla todos los años, así que el modelo ya vio meses de 2026 al entrenar.\n"
        "- **Temporal:** entrena con 2018–2024 y prueba con 2025–2026. Es la prueba más honesta: "
        "simula predecir periodos que todavía no se conocen."
    )
    comp = obtener_comparacion(estrategia)
    metrica = st.selectbox("Métrica a graficar", ["F1 (ALTO)", "Recall (ALTO)", "Precisión (ALTO)", "Accuracy"])
    nombres_cortos = dict(
        zip(modelo_ia.MODELOS_COMPARACION, ["Árbol (informe)", "Árbol prof. 8", "Random Forest", "Reg. logística"])
    )
    grafica = comp.assign(Modelo=comp["Modelo"].map(nombres_cortos)).pivot(
        index="Modelo", columns="Evaluación", values=metrica
    )
    st.bar_chart(grafica, stack=False, horizontal=True)
    st.dataframe(
        comp.style.format({c: "{:.3f}" for c in ["Accuracy", "Precisión (ALTO)", "Recall (ALTO)", "F1 (ALTO)"]}),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "**Cómo leerlo:** la clase ALTO es la minoritaria (~7 %), por eso el F1 de ALTO es más informativo que el accuracy. "
        "Si un modelo baja mucho en la evaluación temporal frente a la aleatoria, está memorizando el pasado en lugar de generalizar."
    )

# --------------------------------------------------------------------------- #
# Pestaña 4: Árbol de decisión
# --------------------------------------------------------------------------- #
with tab_arbol:
    st.header("¿Cómo decide el árbol?")
    info = modelo_ia.info_arbol(modelo)
    a1, a2 = st.columns(2)
    a1.metric("Profundidad del árbol", info["profundidad"])
    a2.metric("Hojas (reglas finales)", info["hojas"])

    niveles = st.slider("Niveles a mostrar", 1, 4, 3)
    st.pyplot(modelo_ia.figura_arbol(modelo, max_depth=niveles))
    st.caption(
        "Cada caja es una pregunta del tipo *variable ≤ valor*. Si la respuesta es sí se sigue por la izquierda, si no por la derecha. "
        "El color indica la clase mayoritaria (azul = ALTO, naranja = BAJO)."
    )

    with st.expander("Códigos numéricos de cada categoría (para leer las preguntas del árbol)"):
        st.write(
            "El árbol trabaja con números. Por ejemplo, `ZONA PORTUARIA <= 0.5` significa "
            "*zona con código 0*."
        )
        for col, categorias in modelo_ia.codigos_categorias(modelo).items():
            st.markdown(f"**{col}**")
            st.dataframe(
                pd.DataFrame({"Código": range(len(categorias)), "Categoría": categorias}),
                width="stretch",
                hide_index=True,
            )
    with st.expander("Reglas en texto"):
        st.code(modelo_ia.reglas_arbol(modelo, max_depth=niveles))

# --------------------------------------------------------------------------- #
# Pestaña 5: Datos
# --------------------------------------------------------------------------- #
with tab_datos:
    st.header("Dataset: Tráfico Portuario Marítimo en Colombia")
    df = obtener_datos()
    st.write(f"{len(df):,} registros · {df.shape[1]} columnas · años {cat['anio_min']}–{cat['anio_max']}")

    fluvial_zona = (
        df.groupby(modelo_ia.COL_ZONA)[modelo_ia.COL_OBJETIVO_CONTINUO].sum().sort_values(ascending=False)
    )
    st.subheader("Tráfico fluvial total por zona (t)")
    st.bar_chart(fluvial_zona[fluvial_zona > 0])

    st.subheader("Vista previa")
    st.dataframe(df.head(200), width="stretch")
