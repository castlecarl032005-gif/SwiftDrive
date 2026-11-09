"""
SwiftDrive - Interfaz web con autenticación por roles (Streamlit).

Ejecución desde la raíz del repositorio:
    pip install -r requirements.txt
    streamlit run Frontend/app.py

Configura las credenciales en .streamlit/secrets.toml.
"""

import hmac
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


# --------------------------------------------------------------------------- #
# 1. Autenticación y roles
# --------------------------------------------------------------------------- #
def obtener_usuarios_configurados():
    """Lee los usuarios desde Streamlit Secrets; no hay contraseñas en el código."""
    try:
        usuarios = st.secrets.get("usuarios", {})
        return dict(usuarios)
    except Exception:
        return {}


def verificar_credenciales(nombre_usuario: str, contrasena: str):
    """Devuelve el rol si las credenciales coinciden; de lo contrario, None."""
    usuarios = obtener_usuarios_configurados()
    for rol, datos in usuarios.items():
        usuario_configurado = str(datos.get("usuario", ""))
        contrasena_configurada = str(datos.get("contrasena", ""))
        if not usuario_configurado or not contrasena_configurada:
            continue
        usuario_ok = hmac.compare_digest(nombre_usuario, usuario_configurado)
        contrasena_ok = hmac.compare_digest(contrasena, contrasena_configurada)
        if usuario_ok and contrasena_ok and rol in ("admin", "usuario"):
            return rol
    return None


def mostrar_inicio_sesion():
    st.title("⚓ SwiftDrive")
    st.subheader("Inicio de sesión")
    st.write("Ingresa con tu cuenta para acceder a las herramientas de predicción de tráfico fluvial.")

    if not obtener_usuarios_configurados():
        st.error("No hay usuarios configurados. El administrador del proyecto debe configurar las credenciales.")
        with st.expander("Cómo configurar las cuentas"):
            st.markdown(
                "Crea el archivo `.streamlit/secrets.toml` en la raíz del proyecto con esta estructura. "
                "Cambia los nombres y contraseñas por valores propios:"
            )
            st.code(
                '[usuarios.admin]\\n'
                'usuario = "administrador"\\n'
                'contrasena = "CAMBIA_ESTA_CLAVE"\\n\\n'
                '[usuarios.usuario]\\n'
                'usuario = "consulta"\\n'
                'contrasena = "CAMBIA_ESTA_OTRA_CLAVE"',
                language="toml",
            )
            st.info(
                "En Streamlit Community Cloud, configura el mismo contenido en "
                "Settings → Secrets. No subas secrets.toml a GitHub."
            )
        return

    with st.form("formulario_login"):
        nombre_usuario = st.text_input("Usuario")
        contrasena = st.text_input("Contraseña", type="password")
        enviar = st.form_submit_button("Iniciar sesión", type="primary", use_container_width=True)

    if enviar:
        rol = verificar_credenciales(nombre_usuario.strip(), contrasena)
        if rol:
            st.session_state["autenticado"] = True
            st.session_state["rol"] = rol
            st.session_state["nombre_usuario"] = nombre_usuario.strip()
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")


if not st.session_state.get("autenticado", False):
    mostrar_inicio_sesion()
    st.stop()

rol_actual = st.session_state.get("rol", "usuario")
nombre_actual = st.session_state.get("nombre_usuario", "usuario")

# --------------------------------------------------------------------------- #
# 2. Barra lateral
# --------------------------------------------------------------------------- #
st.sidebar.title("⚓ SwiftDrive")
st.sidebar.caption("Predicción del nivel de tráfico fluvial en las sociedades portuarias de Colombia.")
st.sidebar.success(
    f"Sesión: {nombre_actual}\n\nRol: {'Administrador' if rol_actual == 'admin' else 'Usuario'}"
)
if st.sidebar.button("Cerrar sesión", use_container_width=True):
    for clave in ("autenticado", "rol", "nombre_usuario"):
        st.session_state.pop(clave, None)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Fuente:** Superintendencia de Transporte — "
    "[datos.gov.co](https://www.datos.gov.co/Transporte/Trafico-Portuario-Mar-timo-En-Colombia/5r3g-zv5z/about_data)"
)

# Cargar solamente un modelo que ya exista. El entrenamiento ahora se inicia
# manualmente desde la pestaña exclusiva del administrador.
def cargar_modelo_disponible():
    if modelo_ia.RUTA_MODELO.exists():
        try:
            return modelo_ia.cargar_modelo()
        except Exception as error:
            st.error(f"No fue posible cargar el modelo guardado: {error}")
    return None


modelo = cargar_modelo_disponible()

if modelo is not None:
    st.sidebar.info(
        f"Modelo disponible · Umbral: {modelo.umbral:,.0f} t · "
        f"Estrategia: {modelo.estrategia_umbral}"
    )
else:
    st.sidebar.warning("Todavía no hay un modelo entrenado y guardado.")

# --------------------------------------------------------------------------- #
# 3. Pestañas según el rol
# --------------------------------------------------------------------------- #
if rol_actual == "admin":
    (
        tab_entrenar,
        tab_pred,
        tab_metricas,
        tab_comp,
        tab_arbol,
        tab_datos,
    ) = st.tabs(
        [
            "🛠️ Entrenar",
            "🔮 Predecir",
            "📊 Evaluación",
            "🧪 Comparación",
            "🌳 Árbol de decisión",
            "🗂️ Datos",
        ]
    )
else:
    (tab_pred,) = st.tabs(["🔮 Predecir"])


# --------------------------------------------------------------------------- #
# Pestaña exclusiva del administrador: Entrenar
# --------------------------------------------------------------------------- #
if rol_actual == "admin":
    with tab_entrenar:
        st.header("Entrenamiento del modelo")
        st.write(
            "Esta sección permite al administrador entrenar el clasificador con los datos históricos. "
            "Al finalizar, el modelo se guarda para que los usuarios puedan realizar predicciones."
        )

        etiquetas_umbral = {
            "mediana": "Mediana de todos los registros (metodología del informe)",
            "mediana_positivos": "Mediana de registros con tráfico mayor que cero",
        }
        estrategia_entrenamiento = st.radio(
            "Criterio para definir tráfico ALTO",
            options=list(etiquetas_umbral),
            format_func=etiquetas_umbral.get,
            help=(
                "Con 'mediana', ALTO significa que el registro tiene tráfico fluvial. "
                "Con 'mediana_positivos', ALTO significa que supera la mediana de los registros que sí tuvieron tráfico."
            ),
        )

        if modelo is not None:
            st.info(
                f"Modelo actualmente guardado: estrategia **{modelo.estrategia_umbral}**, "
                f"umbral **{modelo.umbral:,.0f} toneladas**."
            )

        confirmar = st.checkbox(
            "Confirmo que deseo entrenar nuevamente el modelo y reemplazar el archivo guardado."
        )
        if st.button(
            "Iniciar entrenamiento",
            type="primary",
            disabled=not confirmar,
            use_container_width=True,
        ):
            try:
                with st.spinner("Entrenando y evaluando el modelo. Esto puede tardar unos segundos..."):
                    nuevo_modelo = modelo_ia.entrenar(estrategia_umbral=estrategia_entrenamiento)
                    modelo_ia.guardar_modelo(nuevo_modelo)
                    modelo = nuevo_modelo
                st.success("Entrenamiento completado. El nuevo modelo quedó guardado correctamente.")
                st.metric("Exactitud (accuracy)", f"{modelo.accuracy:.1%}")
                st.write(
                    f"Registros de entrenamiento: **{modelo.n_train:,}** · "
                    f"Registros de prueba: **{modelo.n_test:,}**"
                )
                st.dataframe(
                    pd.DataFrame(modelo.reporte_dict).transpose(),
                    use_container_width=True,
                )
                st.caption(
                    "El modelo guardado será el que utilice la pestaña Predecir. "
                    "En Streamlit Cloud, el almacenamiento local puede reiniciarse al redesplegar la aplicación."
                )
            except Exception as error:
                st.error(f"No se pudo completar el entrenamiento: {error}")


# --------------------------------------------------------------------------- #
# Pestaña Predecir: disponible para ambos roles
# --------------------------------------------------------------------------- #
with tab_pred:
    st.header("Predecir el nivel de tráfico fluvial")

    if modelo is None:
        st.warning(
            "Todavía no hay un modelo entrenado. Solicita al administrador que ingrese a la pestaña "
            "**Entrenar** y genere el primer modelo."
        )
    else:
        cat = modelo.catalogo
        st.caption(
            "Completa los datos de la operación portuaria. El modelo clasificará el nivel de tráfico como BAJO o ALTO."
        )

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
                "Año",
                min_value=cat["anio_min"],
                max_value=cat["anio_max"] + 5,
                value=cat["anio_max"],
                step=1,
            )
            mes = st.selectbox("Mes", range(1, 13), format_func=lambda m: MESES[m - 1])

        if st.button("Realizar predicción", type="primary"):
            try:
                r = modelo_ia.predecir_trafico(
                    zona, sociedad, carga, int(anio), int(mes), modelo=modelo
                )
            except modelo_ia.EntradaInvalida as error:
                st.error(str(error))
            else:
                if r["clase"] == 1:
                    st.error(
                        f"### Tráfico fluvial ALTO\n"
                        f"Probabilidad estimada de ALTO: **{r['probabilidad_alto']:.0%}**"
                    )
                    st.write(
                        "Recomendación: reforzar la planificación de zarpes, personal y protocolos de seguridad."
                    )
                else:
                    st.success(
                        f"### Tráfico fluvial BAJO\n"
                        f"Probabilidad estimada de ALTO: **{r['probabilidad_alto']:.0%}**"
                    )
                    st.write("Operación con demanda fluvial baja: capacidad normal es suficiente.")

                if int(anio) > cat["anio_max"]:
                    st.warning(
                        f"El dataset llega hasta {cat['anio_max']}. Para años posteriores el árbol "
                        "extrapola usando el último año conocido."
                    )

            # Histórico real del mismo escenario, para contrastar.
            df = modelo_ia.cargar_datos()
            historico = df[
                (df[modelo_ia.COL_ZONA] == zona)
                & (df[modelo_ia.COL_SOCIEDAD] == sociedad)
                & (df[modelo_ia.COL_CARGA] == carga)
            ]
            if not historico.empty:
                st.subheader("Histórico real de esta sociedad y tipo de carga")
                serie = (
                    historico.assign(
                        Fecha=pd.to_datetime(
                            dict(
                                year=historico[modelo_ia.COL_ANIO],
                                month=historico[modelo_ia.COL_MES],
                                day=1,
                            )
                        )
                    )
                    .groupby("Fecha")[modelo_ia.COL_OBJETIVO_CONTINUO]
                    .sum()
                    .rename("Tráfico fluvial (t)")
                )
                st.line_chart(serie)
            else:
                st.caption("No hay registros históricos para esta combinación.")


# --------------------------------------------------------------------------- #
# Secciones de administración y evaluación
# --------------------------------------------------------------------------- #
if rol_actual == "admin":
    with tab_metricas:
        st.header("Evaluación del modelo")
        if modelo is None:
            st.warning("Primero debes entrenar un modelo desde la pestaña Entrenar.")
        else:
            st.write("Resultados obtenidos sobre el conjunto de prueba separado durante el entrenamiento.")
            rep = modelo.reporte_dict
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{modelo.accuracy:.1%}")
            m2.metric("Precisión (ALTO)", f"{rep['ALTO (1)']['precision']:.1%}")
            m3.metric("Recall (ALTO)", f"{rep['ALTO (1)']['recall']:.1%}")
            m4.metric("F1 (ALTO)", f"{rep['ALTO (1)']['f1-score']:.1%}")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Matriz de confusión")
                matriz = pd.DataFrame(
                    modelo.matriz_confusion,
                    index=["Real BAJO", "Real ALTO"],
                    columns=["Predicho BAJO", "Predicho ALTO"],
                )
                st.dataframe(matriz, use_container_width=True)
                fig, ax = plt.subplots(figsize=(5, 4))
                imagen = ax.imshow(matriz.values, cmap="Blues")
                ax.set_xticks([0, 1], labels=["Predicho BAJO", "Predicho ALTO"])
                ax.set_yticks([0, 1], labels=["Real BAJO", "Real ALTO"])
                ax.set_title("Matriz de confusión")
                for fila in range(2):
                    for columna in range(2):
                        ax.text(
                            columna,
                            fila,
                            str(matriz.values[fila, columna]),
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=13,
                        )
                fig.colorbar(imagen, ax=ax, label="Cantidad de registros")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with c2:
                st.subheader("Importancia de variables")
                st.bar_chart(
                    pd.Series(modelo.importancias, name="Importancia").sort_values(ascending=False)
                )
                st.subheader("Distribución de clases")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Registros": [
                                modelo.distribucion_clases.get(0, 0),
                                modelo.distribucion_clases.get(1, 0),
                            ]
                        },
                        index=["BAJO (0)", "ALTO (1)"],
                    ),
                    use_container_width=True,
                )

            st.subheader("Reporte de clasificación")
            st.code(modelo.reporte)
            st.caption(
                f"Entrenamiento: {modelo.n_train:,} registros · "
                f"Prueba: {modelo.n_test:,} registros · DecisionTreeClassifier"
            )

    with tab_comp:
        st.header("Comparación de modelos")
        st.write(
            "Se comparan cuatro algoritmos con dos estrategias de evaluación:\n"
            "- **Aleatoria 80/20:** mezcla registros de todos los años.\n"
            "- **Temporal:** entrena con años hasta 2024 y prueba con años posteriores."
        )
        estrategia_comp = modelo.estrategia_umbral if modelo is not None else "mediana"
        try:
            comp = modelo_ia.comparar_modelos(estrategia_comp)
            metrica = st.selectbox(
                "Métrica a graficar",
                ["F1 (ALTO)", "Recall (ALTO)", "Precisión (ALTO)", "Accuracy"],
            )
            nombres_cortos = dict(
                zip(
                    modelo_ia.MODELOS_COMPARACION,
                    ["Árbol (informe)", "Árbol prof. 8", "Random Forest", "Reg. logística"],
                )
            )
            grafica = comp.assign(Modelo=comp["Modelo"].map(nombres_cortos)).pivot(
                index="Modelo", columns="Evaluación", values=metrica
            )
            st.bar_chart(grafica, horizontal=True)
            st.dataframe(
                comp.style.format(
                    {
                        c: "{:.3f}"
                        for c in [
                            "Accuracy",
                            "Precisión (ALTO)",
                            "Recall (ALTO)",
                            "F1 (ALTO)",
                        ]
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.info(
                "**Cómo leerlo:** la clase ALTO es minoritaria, por eso F1 y Recall de ALTO "
                "son métricas importantes además de Accuracy."
            )
        except Exception as error:
            st.error(f"No fue posible comparar los modelos: {error}")

    with tab_arbol:
        st.header("¿Cómo decide el árbol?")
        if modelo is None:
            st.warning("Primero debes entrenar un modelo.")
        else:
            info = modelo_ia.info_arbol(modelo)
            a1, a2 = st.columns(2)
            a1.metric("Profundidad del árbol", info["profundidad"])
            a2.metric("Hojas (reglas finales)", info["hojas"])

            niveles = st.slider("Niveles a mostrar", 1, 4, 3)
            st.pyplot(modelo_ia.figura_arbol(modelo, max_depth=niveles))
            st.caption(
                "Cada caja es una pregunta del tipo *variable ≤ valor*. Si la respuesta es sí se sigue "
                "por la izquierda, si no por la derecha."
            )

            with st.expander("Códigos numéricos de cada categoría"):
                for columna, categorias in modelo_ia.codigos_categorias(modelo).items():
                    st.markdown(f"**{columna}**")
                    st.dataframe(
                        pd.DataFrame(
                            {"Código": range(len(categorias)), "Categoría": categorias}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            with st.expander("Reglas en texto"):
                st.code(modelo_ia.reglas_arbol(modelo, max_depth=niveles))

    with tab_datos:
        st.header("Dataset: Tráfico Portuario Marítimo en Colombia")
        df = modelo_ia.cargar_datos()
        st.write(
            f"{len(df):,} registros · {df.shape[1]} columnas · "
            f"años {df[modelo_ia.COL_ANIO].min()}–{df[modelo_ia.COL_ANIO].max()}"
        )
        fluvial_zona = (
            df.groupby(modelo_ia.COL_ZONA)[modelo_ia.COL_OBJETIVO_CONTINUO]
            .sum()
            .sort_values(ascending=False)
        )
        st.subheader("Tráfico fluvial total por zona (t)")
        st.bar_chart(fluvial_zona[fluvial_zona > 0])
        st.subheader("Vista previa")
        st.dataframe(df.head(200), use_container_width=True)
