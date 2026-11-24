import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelos.clasificador_cnn.evaluacion_modelo import (
    UMBRAL_CLASIFICACION,
    cargar_modelo_entrenado,
)
from modelos.estimador_cnn.evaluacion_modelo import (
    diferencia_fase_grados,
    evaluar_estimador_guardado,
)
from modelos.generador_senales import FRECUENCIA_MUESTREO


CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
RUTA_COMPARACION = CARPETA_RESULTADOS / "comparacion_parametros_estimador.png"
RUTA_ERRORES = CARPETA_RESULTADOS / "distribucion_errores_estimador.png"
RUTA_ANALISIS_COMPLETO = CARPETA_RESULTADOS / "ejemplo_analisis_completo.png"

# Mostramos solo los primeros 30 ms para distinguir los ciclos de la onda.
TIEMPO_VISIBLE_SEGUNDOS = 0.030


def _validar_resultado(resultado):
    """Comprueba que existan predicciones y valores reales compatibles."""

    predicciones = np.asarray(resultado["predicciones"], dtype=np.float32)
    reales = np.asarray(resultado["parametros_reales"], dtype=np.float32)
    if predicciones.shape != reales.shape:
        raise ValueError("Predicciones y valores reales deben tener la misma forma.")
    if predicciones.ndim != 2 or predicciones.shape[1] != 3:
        raise ValueError("Se esperan frecuencia, amplitud y fase por senal.")
    if len(predicciones) == 0:
        raise ValueError("No existen senales para crear las graficas.")
    return predicciones, reales


def crear_grafica_comparacion(
    resultado,
    ruta_salida=RUTA_COMPARACION,
):
    """Compara los parámetros reales y estimados en el conjunto de prueba."""

    predicciones, reales = _validar_resultado(resultado)
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    configuraciones = [
        {
            "indice": 0,
            "titulo": "Frecuencia",
            "etiqueta": "Frecuencia (Hz)",
            "error": resultado["error_frecuencia_hz"],
            "texto_error": "Hz",
        },
        {
            "indice": 1,
            "titulo": "Amplitud",
            "etiqueta": "Amplitud",
            "error": resultado["error_amplitud"],
            "texto_error": "",
        },
        {
            "indice": 2,
            "titulo": "Fase",
            "etiqueta": "Fase (grados)",
            "error": resultado["error_fase_grados"],
            "texto_error": "grados",
        },
    ]

    figura, graficas = plt.subplots(1, 3, figsize=(16, 5))
    for grafica, configuracion in zip(graficas, configuraciones):
        indice = configuracion["indice"]
        valores_reales = reales[:, indice]
        valores_estimados = predicciones[:, indice]
        minimo = float(min(valores_reales.min(), valores_estimados.min()))
        maximo = float(max(valores_reales.max(), valores_estimados.max()))
        margen = max((maximo - minimo) * 0.08, 0.01)

        grafica.scatter(
            valores_reales,
            valores_estimados,
            color="tab:blue",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
        )
        grafica.plot(
            [minimo - margen, maximo + margen],
            [minimo - margen, maximo + margen],
            linestyle="--",
            color="black",
            label="Estimación perfecta",
        )
        grafica.set_xlim(minimo - margen, maximo + margen)
        grafica.set_ylim(minimo - margen, maximo + margen)
        grafica.set_title(
            f"{configuracion['titulo']}\n"
            f"Error promedio: {configuracion['error']:.3f} "
            f"{configuracion['texto_error']}"
        )
        grafica.set_xlabel(f"{configuracion['etiqueta']} real")
        grafica.set_ylabel(f"{configuracion['etiqueta']} estimada")
        grafica.grid(True, alpha=0.25)
        grafica.legend()

    figura.suptitle(
        f"Valores reales y estimados en {len(predicciones)} senoidales de prueba",
        fontsize=15,
    )
    figura.tight_layout(rect=(0, 0, 1, 0.93))
    figura.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    return figura, ruta_salida


def crear_grafica_errores(resultado, ruta_salida=RUTA_ERRORES):
    """Muestra como se distribuyen los errores individuales del estimador."""

    predicciones, reales = _validar_resultado(resultado)
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    errores = [
        np.abs(predicciones[:, 0] - reales[:, 0]),
        np.abs(predicciones[:, 1] - reales[:, 1]),
        diferencia_fase_grados(predicciones[:, 2], reales[:, 2]),
    ]
    titulos = [
        "Error de frecuencia",
        "Error de amplitud",
        "Error angular de fase",
    ]
    unidades = ["Hz", "", "grados"]
    colores = ["tab:blue", "tab:green", "tab:orange"]

    figura, graficas = plt.subplots(1, 3, figsize=(16, 5))
    for grafica, valores, titulo, unidad, color in zip(
        graficas,
        errores,
        titulos,
        unidades,
        colores,
    ):
        promedio = float(np.mean(valores))
        grafica.hist(
            valores,
            bins=min(10, len(valores)),
            color=color,
            alpha=0.8,
            edgecolor="white",
        )
        grafica.axvline(
            promedio,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"Promedio: {promedio:.3f} {unidad}".strip(),
        )
        grafica.set_title(titulo)
        etiqueta_error = f"Error absoluto ({unidad})" if unidad else "Error absoluto"
        grafica.set_xlabel(etiqueta_error)
        grafica.set_ylabel("Cantidad de señales")
        grafica.yaxis.set_major_locator(MaxNLocator(integer=True))
        grafica.grid(axis="y", alpha=0.25)
        grafica.legend()

    figura.suptitle(
        f"Distribución de errores en {len(predicciones)} senoidales de prueba",
        fontsize=15,
    )
    figura.tight_layout(rect=(0, 0, 1, 0.93))
    figura.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    return figura, ruta_salida


def crear_grafica_analisis_completo(
    muestras,
    parametros_reales,
    parametros_estimados,
    probabilidad_senoidal,
    ruta_salida=RUTA_ANALISIS_COMPLETO,
):
    """Muestra la decisión del detector y la estimación para una señal de prueba."""

    muestras = np.asarray(muestras, dtype=np.float32)
    parametros_reales = np.asarray(parametros_reales, dtype=np.float32)
    parametros_estimados = np.asarray(parametros_estimados, dtype=np.float32)
    if muestras.ndim != 1:
        raise ValueError("Las muestras deben tener una sola dimensión.")
    if parametros_reales.shape != (3,) or parametros_estimados.shape != (3,):
        raise ValueError("Se esperan frecuencia, amplitud y fase reales y estimadas.")

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    tiempo = np.arange(len(muestras)) / FRECUENCIA_MUESTREO
    visibles = tiempo <= TIEMPO_VISIBLE_SEGUNDOS
    figura = plt.figure(figsize=(14, 8))
    distribucion = figura.add_gridspec(2, 3, height_ratios=[1.35, 1])

    grafica_senal = figura.add_subplot(distribucion[0, :])
    grafica_senal.plot(tiempo[visibles] * 1000, muestras[visibles], color="tab:blue")
    es_senoidal = probabilidad_senoidal >= UMBRAL_CLASIFICACION
    decision = "senoidal" if es_senoidal else "no senoidal"
    grafica_senal.set_title(
        f"Detector → {decision} | P(senoidal) {probabilidad_senoidal:.2%} | "
        f"Umbral {UMBRAL_CLASIFICACION:.2%}"
    )
    grafica_senal.set_xlabel("Tiempo (milisegundos)")
    grafica_senal.set_ylabel("Amplitud")
    grafica_senal.grid(True, alpha=0.25)

    nombres = ["Frecuencia", "Amplitud", "Fase"]
    unidades = ["Hz", "", "grados"]
    decimales = [2, 3, 2]

    for indice, (nombre, unidad, numero_decimales) in enumerate(
        zip(nombres, unidades, decimales)
    ):
        grafica = figura.add_subplot(distribucion[1, indice])
        valores = [parametros_reales[indice], parametros_estimados[indice]]
        barras = grafica.bar(
            ["Real", "Estimada"],
            valores,
            color=["tab:gray", "tab:blue"],
        )
        grafica.set_title(nombre)
        grafica.set_ylabel(unidad)
        grafica.grid(axis="y", alpha=0.25)
        limite = max(valores) * 1.22
        grafica.set_ylim(0, limite if limite > 0 else 1)

        for barra, valor in zip(barras, valores):
            grafica.text(
                barra.get_x() + barra.get_width() / 2,
                barra.get_height(),
                f"{valor:.{numero_decimales}f}",
                ha="center",
                va="bottom",
            )

    titulo = (
        "Detección y estimación de una senoidal de prueba"
        if es_senoidal
        else "Estimación directa de una senoidal de prueba"
    )
    figura.suptitle(titulo, fontsize=16)
    figura.tight_layout(rect=(0, 0, 1, 0.94))
    figura.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    return figura, ruta_salida


def crear_visualizaciones_estimador():
    """Genera las tres gráficas con las mismas señales de la evaluación."""

    evaluacion, _, muestras = evaluar_estimador_guardado()
    detector, _ = cargar_modelo_entrenado()
    entrada = torch.as_tensor(muestras[:1], dtype=torch.float32)
    with torch.no_grad():
        probabilidad_senoidal = detector(entrada).sigmoid().item()

    _, ruta_comparacion = crear_grafica_comparacion(evaluacion)
    _, ruta_errores = crear_grafica_errores(evaluacion)
    _, ruta_analisis = crear_grafica_analisis_completo(
        muestras[0, 0],
        evaluacion["parametros_reales"][0],
        evaluacion["predicciones"][0],
        probabilidad_senoidal,
    )

    return [ruta_comparacion, ruta_errores, ruta_analisis]


def main():
    rutas = crear_visualizaciones_estimador()
    print("Gráficas del estimador guardadas")
    for ruta in rutas:
        print(f"- {ruta}")
    plt.show()


if __name__ == "__main__":
    main()
