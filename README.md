# ⚓ SwiftDrive

Sistema de inteligencia artificial que **predice si el tráfico fluvial de una operación portuaria en Colombia será ALTO o BAJO**. Usa como entradas la zona portuaria, la sociedad portuaria, el tipo de carga, el año y el mes.

Proyecto de Inteligencia Artificial, Grupo T02, Universidad Simón Bolívar:
Carlos Daniel Castillo Solano · Santiago Rafael Leyva Pérez · José Mario Yepes López.

## Estructura

```
Backend/
  modelo_ia.py          # carga de datos, entrenamiento, evaluación y predicción
  data/                 # dataset oficial (CSV)
Frontend/
  app.py                # interfaz web con Streamlit
notebooks/
  SwiftDrive_modelo.ipynb   # versión para Google Colab (secciones 1-8 del informe)
tests/
  test_modelo_ia.py     # pruebas automáticas
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
```

Entrenar y ver las métricas por consola:

```bash
python Backend/modelo_ia.py
```

Abrir la aplicación web en http://localhost:8501:

```bash
streamlit run Frontend/app.py
```

Ejecutar las pruebas:

```bash
python -m pytest tests
```

**En Google Colab:** abre `notebooks/SwiftDrive_modelo.ipynb`, sube el CSV de `Backend/data/` y ejecuta todas las celdas.

## Modelo

| Elemento | Detalle |
|---|---|
| Tipo de aprendizaje | Supervisado, clasificación binaria |
| Algoritmo | `DecisionTreeClassifier` (scikit-learn) |
| Entradas (X) | `ZONA PORTUARIA`, `SOCIEDAD PORTUARIA`, `TIPO DE CARGA`, `AÑO VIGENCIA`, `MES VIGENCIA` |
| Objetivo (y) | `Nivel_Fluvial`: 1 = ALTO (`FLUVIAL` > umbral), 0 = BAJO |
| Partición | 80 % entrenamiento / 20 % prueba (estratificada, `random_state=42`) |

### Resultados (umbral = mediana, conjunto de prueba)

| Métrica | BAJO (0) | ALTO (1) |
|---|---|---|
| Precisión | 0,99 | 0,82 |
| Recall | 0,99 | 0,89 |
| F1 | 0,99 | 0,85 |

Accuracy: **97,8 %**. Las variables más importantes son la zona portuaria (48 %) y la sociedad (29 %).

### Sobre el umbral

La mediana de `FLUVIAL` es **0 toneladas**, porque solo el 7 % de los registros tiene tráfico fluvial. Con la regla del informe ("mayor que la mediana"), **ALTO significa que la operación tiene tráfico fluvial**. Por eso las clases quedan desbalanceadas: 9.001 BAJO frente a 682 ALTO.

La aplicación y el script permiten una segunda opción, `--umbral mediana_positivos`. Esta usa la mediana de los registros con tráfico mayor que 0 (unas 4.312 t), de modo que ALTO pasa a significar "tráfico fluvial por encima de lo habitual". Con esa opción, el F1 de la clase ALTO baja a 0,72.

## Comparación de modelos y evaluación temporal

La aplicación (pestaña **🧪 Comparación de modelos**), el notebook (sección 9) y `python Backend/modelo_ia.py --comparar` comparan cuatro modelos de dos maneras:

- **Aleatoria 80/20:** la del informe. Mezcla todos los años.
- **Temporal:** entrena con 2018-2024 y prueba con 2025-2026 (1.659 registros). Simula predecir periodos que aún no se conocen.

F1 de la clase ALTO (umbral = mediana):

| Modelo | Aleatoria 80/20 | Temporal |
|---|---|---|
| Árbol de decisión (modelo del informe) | 0,852 | 0,820 |
| Árbol de decisión (profundidad máx. 8) | 0,812 | 0,848 |
| Random Forest (200 árboles) | 0,892 | 0,820 |
| Regresión logística | 0,862 | 0,809 |

**Lectura:** en la evaluación aleatoria Random Forest gana, pero en la temporal, que es la más realista, la ventaja desaparece. Las diferencias entre modelos en la prueba temporal son pequeñas (la prueba tiene solo 122 casos ALTO), así que el árbol de decisión es una elección razonable, y además es el más fácil de explicar.

## Interpretabilidad del árbol

La pestaña **🌳 Árbol de decisión** y la sección 10 del notebook dibujan los primeros niveles del árbol y la tabla con el código numérico de cada categoría. La primera pregunta del árbol es la zona portuaria (Barrancabermeja frente al resto), lo que concuerda con que la zona sea la variable más importante.

## Correcciones respecto al primer notebook de Colab

1. **Separador de miles:** el CSV escribe `1.389` para 1389 y `2.023` para el año 2023. Sin `thousands='.'`, pandas lo leía como decimal.
2. **Predicción silenciosamente errónea:** la función anterior reemplazaba los valores desconocidos por el primero del catálogo sin avisar. Por ejemplo, `"ZONA PORTUARIA DE CARTAGENA"` y `"GRANEL LÍQUIDO"` no existen en el dataset. Ahora se valida la entrada y se muestran las opciones válidas.
3. **Caracteres dañados** en nombres de sociedades (`COMPA�IA` → `COMPAÑIA`).
4. **Partición estratificada**, para mantener la proporción de clases en entrenamiento y prueba.

## Fuente de datos

Superintendencia de Transporte, *Tráfico Portuario Marítimo en Colombia* (actualizado el 1 de octubre de 2026), en
[datos.gov.co](https://www.datos.gov.co/Transporte/Trafico-Portuario-Mar-timo-En-Colombia/5r3g-zv5z/about_data).
