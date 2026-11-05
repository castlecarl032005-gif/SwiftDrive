"""Pruebas del módulo Backend/modelo_ia.py.  Ejecutar: python -m pytest tests"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Backend"))
import modelo_ia  # noqa: E402


@pytest.fixture(scope="module")
def datos():
    return modelo_ia.cargar_datos()


@pytest.fixture(scope="module")
def modelo():
    return modelo_ia.entrenar()


def test_lectura_separador_de_miles(datos):
    assert datos.shape == (9683, 14)
    assert datos["AÑO VIGENCIA"].min() == 2018
    assert datos["AÑO VIGENCIA"].max() == 2026
    # "1.389" debe leerse como 1389, no como 1.389
    assert datos["FLUVIAL"].max() > 1000


def test_correccion_caracteres(datos):
    sociedades = set(datos["SOCIEDAD PORTUARIA"])
    assert "COMPAÑIA DE PUERTOS ASOCIADOS S.A." in sociedades
    assert not any("�" in s for s in sociedades)


def test_umbrales(datos):
    assert modelo_ia.calcular_umbral(datos["FLUVIAL"], "mediana") == 0
    assert modelo_ia.calcular_umbral(datos["FLUVIAL"], "mediana_positivos") > 0
    with pytest.raises(ValueError):
        modelo_ia.calcular_umbral(datos["FLUVIAL"], "otra")


def test_metricas_minimas(modelo):
    assert modelo.n_train + modelo.n_test == 9683
    assert modelo.accuracy > 0.9
    assert modelo.reporte_dict["ALTO (1)"]["f1-score"] > 0.7


def test_prediccion_valida(modelo):
    r = modelo_ia.predecir_trafico("BARRANCABERMEJA", "ECOPETROL S.A.", "GRANEL LIQUIDO", 2026, 6, modelo=modelo)
    assert r["nivel"] == "ALTO" and r["clase"] == 1
    r = modelo_ia.predecir_trafico(
        "CARTAGENA", "SOCIEDAD PORTUARIA REGIONAL DE CARTAGENA SA", "CONTENEDORES", 2026, 6, modelo=modelo
    )
    assert r["nivel"] == "BAJO"


@pytest.mark.parametrize(
    "args",
    [
        ("ZONA PORTUARIA DE CARTAGENA", "ECOPETROL S.A.", "GRANEL LIQUIDO", 2026, 6),  # zona inexistente
        ("BARRANCABERMEJA", "ECOPETROL S.A.", "GRANEL LÍQUIDO", 2026, 6),  # carga con tilde
        ("BARRANCABERMEJA", "ECOPETROL S.A.", "GRANEL LIQUIDO", 2026, 13),  # mes inválido
        ("BARRANCABERMEJA", "ECOPETROL S.A.", "GRANEL LIQUIDO", 2.026, 6),  # año mal leído
    ],
)
def test_entrada_invalida(modelo, args):
    with pytest.raises(modelo_ia.EntradaInvalida):
        modelo_ia.predecir_trafico(*args, modelo=modelo)


def test_comparacion_de_modelos():
    comp = modelo_ia.comparar_modelos()
    assert set(comp["Modelo"]) == set(modelo_ia.MODELOS_COMPARACION)
    assert comp["Evaluación"].nunique() == 2  # aleatoria + temporal
    assert len(comp) == 2 * len(modelo_ia.MODELOS_COMPARACION)
    assert comp["Accuracy"].between(0, 1).all()
    assert (comp["F1 (ALTO)"] > 0.6).all()


def test_interpretabilidad_del_arbol(modelo):
    info = modelo_ia.info_arbol(modelo)
    assert info["profundidad"] > 3 and info["hojas"] > 10
    codigos = modelo_ia.codigos_categorias(modelo)
    assert set(codigos) == set(modelo_ia.COLUMNAS_CATEGORICAS)
    assert "BARRANCABERMEJA" in codigos[modelo_ia.COL_ZONA]
    assert "ZONA PORTUARIA" in modelo_ia.reglas_arbol(modelo, max_depth=2)
    fig = modelo_ia.figura_arbol(modelo, max_depth=2)
    assert fig.axes
