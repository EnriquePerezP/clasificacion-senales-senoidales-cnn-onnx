"""Gráficas para presentar los resultados obtenidos en el Bloque 7."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

# Permite ejecutar el archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasificador_senales.generador_senales import FRECUENCIA_MUESTREO


CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
RUTA_GRAFICA_RESUMEN = CARPETA_RESULTADOS / "resumen_evaluacion.png"
RUTA_GRAFICA_EJEMPLOS = CARPETA_RESULTADOS / "ejemplos_clasificacion.png"
RUTA_GRAFICA_ESPACIO = CARPETA_RESULTADOS / "espacio_aprendido.png"

# Treinta milisegundos permiten apreciar varios ciclos sin saturar la gráfica.
TIEMPO_VISIBLE_SEGUNDOS = 0.030


def crear_grafica_resumen(resultado, ruta_salida=RUTA_GRAFICA_RESUMEN):
    """Crea la matriz de confusión y una comparación de métricas."""

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    figura, graficas = plt.subplots(1, 2, figsize=(12, 5))

    matriz = np.asarray(
        [
            [resultado["verdaderos_negativos"], resultado["falsos_positivos"]],
            [resultado["falsos_negativos"], resultado["verdaderos_positivos"]],
        ]
    )

    imagen = graficas[0].imshow(matriz, cmap="Blues")
    graficas[0].set_title("Matriz de confusión")
    graficas[0].set_xlabel("Predicción de la CNN")
    graficas[0].set_ylabel("Respuesta correcta")
    graficas[0].set_xticks([0, 1], ["No senoidal", "Senoidal"])
    graficas[0].set_yticks([0, 1], ["No senoidal", "Senoidal"])

    limite_color = matriz.max() / 2 if matriz.size else 0
    for fila in range(2):
        for columna in range(2):
            valor = matriz[fila, columna]
            color_texto = "white" if valor > limite_color else "black"
            graficas[0].text(
                columna,
                fila,
                str(valor),
                ha="center",
                va="center",
                fontsize=16,
                color=color_texto,
            )

    figura.colorbar(imagen, ax=graficas[0], label="Cantidad de señales")

    nombres_metricas = [
        "Exactitud",
        "Precisión al decir\nsenoidal",
        "Senoidales\ndetectadas",
    ]
    valores_metricas = [
        resultado["exactitud"],
        resultado["precision_senoidal"],
        resultado["deteccion_senoidales"],
    ]
    barras = graficas[1].bar(
        nombres_metricas,
        valores_metricas,
        color=["tab:blue", "tab:green", "tab:orange"],
    )
    graficas[1].set_title("Resultados principales")
    graficas[1].set_ylabel("Porcentaje")
    graficas[1].set_ylim(0, 1.08)
    graficas[1].set_yticks(
        np.linspace(0, 1, 6),
        [f"{valor:.0%}" for valor in np.linspace(0, 1, 6)],
    )
    graficas[1].grid(axis="y", alpha=0.3)

    for barra, valor in zip(barras, valores_metricas):
        graficas[1].text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.02,
            f"{valor:.2%}",
            ha="center",
            va="bottom",
        )

    figura.suptitle("Evaluación del detector de señales senoidales", fontsize=14)
    figura.tight_layout(rect=(0, 0, 1, 0.95))
    figura.savefig(ruta_salida, dpi=160, bbox_inches="tight")

    return figura, ruta_salida


def crear_grafica_ejemplos(resultado, ruta_salida=RUTA_GRAFICA_EJEMPLOS):
    """Dibuja ejemplos reales de los aciertos y errores encontrados."""

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    nombres = {
        "senoidal_correcta": "Senoidal detectada correctamente",
        "no_senoidal_correcta": "No senoidal rechazada correctamente",
        "falso_positivo": "Error: aceptó una no senoidal",
        "falso_negativo": "Error: no detectó una senoidal",
    }
    orden = [
        "senoidal_correcta",
        "no_senoidal_correcta",
        "falso_positivo",
        "falso_negativo",
    ]
    ejemplos_disponibles = [
        (categoria, resultado["ejemplos"][categoria])
        for categoria in orden
        if categoria in resultado["ejemplos"]
    ]

    if not ejemplos_disponibles:
        raise ValueError("No existen ejemplos para crear la gráfica.")

    figura, graficas = plt.subplots(
        len(ejemplos_disponibles),
        1,
        figsize=(11, 2.7 * len(ejemplos_disponibles)),
        squeeze=False,
    )
    graficas = graficas[:, 0]

    for grafica, (categoria, ejemplo) in zip(graficas, ejemplos_disponibles):
        muestras = ejemplo["muestras"]
        tiempo = np.arange(len(muestras)) / FRECUENCIA_MUESTREO
        visibles = tiempo <= TIEMPO_VISIBLE_SEGUNDOS
        probabilidad = ejemplo["probabilidad_senoidal"]

        grafica.plot(tiempo[visibles] * 1000, muestras[visibles])
        grafica.set_title(
            f"{nombres[categoria]} | "
            f"probabilidad de senoidal: {probabilidad:.2%}"
        )
        grafica.set_ylabel("Amplitud")
        grafica.grid(True, alpha=0.3)

    graficas[-1].set_xlabel("Tiempo en milisegundos")
    figura.suptitle("Ejemplos tomados del grupo de prueba", fontsize=14)
    figura.tight_layout(rect=(0, 0, 1, 0.96))
    figura.savefig(ruta_salida, dpi=160, bbox_inches="tight")

    return figura, ruta_salida


def _reducir_a_dos_dimensiones(caracteristicas):
    """Resume las características de la CNN en dos ejes mediante PCA."""

    caracteristicas = np.asarray(caracteristicas, dtype=np.float64)
    if caracteristicas.ndim != 2 or len(caracteristicas) < 2:
        raise ValueError("Se necesitan varias señales para crear esta gráfica.")

    caracteristicas_centradas = caracteristicas - caracteristicas.mean(axis=0)
    _, _, direcciones = np.linalg.svd(
        caracteristicas_centradas,
        full_matrices=False,
    )
    coordenadas = caracteristicas_centradas @ direcciones[:2].T

    # En el caso excepcional de tener una sola característica, agregamos un
    # segundo eje vacío para conservar el formato de la gráfica.
    if coordenadas.shape[1] == 1:
        coordenadas = np.column_stack(
            [coordenadas[:, 0], np.zeros(len(coordenadas))]
        )

    return coordenadas


def _agregar_elipse(grafica, puntos, color):
    """Dibuja el área aproximada en la que se concentra un grupo."""

    if len(puntos) < 3:
        return

    centro = puntos.mean(axis=0)
    covarianza = np.cov(puntos, rowvar=False)
    valores, vectores = np.linalg.eigh(covarianza)
    orden = valores.argsort()[::-1]
    valores = np.maximum(valores[orden], 0)
    vectores = vectores[:, orden]

    # 5.991 es la referencia habitual para una elipse bidimensional del 95 %.
    ancho, alto = 2 * np.sqrt(valores * 5.991)
    angulo = np.degrees(np.arctan2(vectores[1, 0], vectores[0, 0]))

    elipse = Ellipse(
        xy=centro,
        width=ancho,
        height=alto,
        angle=angulo,
        facecolor=color,
        edgecolor=color,
        alpha=0.18,
        linewidth=2,
    )
    grafica.add_patch(elipse)


def crear_grafica_espacio_aprendido(
    modelo,
    cargador_prueba,
    tipos,
    ruta_salida=RUTA_GRAFICA_ESPACIO,
    *,
    umbral,
):
    """Muestra cómo organiza la CNN las señales antes de clasificarlas."""

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    tipos = np.asarray(tipos)

    dispositivo = next(modelo.parameters()).device
    grupos_caracteristicas = []
    grupos_clases = []
    grupos_predicciones = []

    modelo.eval()
    with torch.no_grad():
        for muestras, clases in cargador_prueba:
            muestras = muestras.to(dispositivo)
            caracteristicas = modelo.extraer_caracteristicas(muestras)
            salidas = modelo.clasificador(caracteristicas)
            predicciones = (torch.sigmoid(salidas) >= umbral).to(clases.dtype)

            grupos_caracteristicas.append(caracteristicas.cpu().numpy())
            grupos_clases.append(clases.cpu().numpy().reshape(-1))
            grupos_predicciones.append(predicciones.cpu().numpy().reshape(-1))

    caracteristicas = np.concatenate(grupos_caracteristicas)
    clases = np.concatenate(grupos_clases).astype(int)
    predicciones = np.concatenate(grupos_predicciones).astype(int)

    if len(tipos) != len(caracteristicas):
        raise ValueError("Debe existir un tipo por cada señal evaluada.")

    coordenadas = _reducir_a_dos_dimensiones(caracteristicas)
    figura, grafica = plt.subplots(figsize=(10, 7))

    colores = {
        "senoidal": "tab:blue",
        "cuadrada": "tab:orange",
        "triangular": "tab:green",
    }
    etiquetas = {
        "senoidal": "Senoidal (clase 1)",
        "cuadrada": "Cuadrada (clase 0)",
        "triangular": "Triangular (clase 0)",
    }

    for tipo in ("senoidal", "cuadrada", "triangular"):
        pertenece_al_tipo = tipos == tipo
        if not np.any(pertenece_al_tipo):
            continue

        puntos = coordenadas[pertenece_al_tipo]
        color = colores[tipo]
        grafica.scatter(
            puntos[:, 0],
            puntos[:, 1],
            color=color,
            alpha=0.8,
            label=etiquetas[tipo],
        )
        _agregar_elipse(grafica, puntos, color)

    errores = predicciones != clases
    if np.any(errores):
        grafica.scatter(
            coordenadas[errores, 0],
            coordenadas[errores, 1],
            marker="x",
            color="black",
            s=90,
            linewidths=2,
            label="Clasificación incorrecta",
        )

    grafica.set_title("Cómo separa la CNN los tipos de señal")
    grafica.set_xlabel("Resumen visual 1 (PCA)")
    grafica.set_ylabel("Resumen visual 2 (PCA)")
    grafica.grid(True, alpha=0.3)
    grafica.legend()
    figura.tight_layout()
    figura.savefig(ruta_salida, dpi=160, bbox_inches="tight")

    return figura, ruta_salida


def crear_visualizaciones(
    resultado,
    modelo=None,
    cargador_prueba=None,
    tipos=None,
    mostrar=True,
):
    """Guarda las gráficas y opcionalmente abre sus ventanas."""

    figura_resumen, ruta_resumen = crear_grafica_resumen(resultado)
    figura_ejemplos, ruta_ejemplos = crear_grafica_ejemplos(resultado)
    figura_espacio = None
    ruta_espacio = None

    if modelo is not None and cargador_prueba is not None and tipos is not None:
        figura_espacio, ruta_espacio = crear_grafica_espacio_aprendido(
            modelo,
            cargador_prueba,
            tipos,
            umbral=resultado["umbral"],
        )

    print()
    print(f"Gráfica de resultados guardada en: {ruta_resumen}")
    print(f"Gráfica de ejemplos guardada en: {ruta_ejemplos}")
    if ruta_espacio is not None:
        print(f"Gráfica del espacio aprendido guardada en: {ruta_espacio}")

    if mostrar:
        plt.show()
    else:
        plt.close(figura_resumen)
        plt.close(figura_ejemplos)
        if figura_espacio is not None:
            plt.close(figura_espacio)

    return {
        "resumen": ruta_resumen,
        "ejemplos": ruta_ejemplos,
        "espacio_aprendido": ruta_espacio,
    }


def main():
    analizador = argparse.ArgumentParser(
        description="Evalúa la CNN entrenada y genera las gráficas de resultados."
    )
    analizador.add_argument(
        "--sin-mostrar",
        action="store_true",
        help="Guarda las gráficas sin abrir sus ventanas.",
    )
    argumentos = analizador.parse_args()

    from clasificador_senales.clasificador_cnn.evaluacion_modelo import (
        main as ejecutar_evaluacion,
    )

    ejecutar_evaluacion(mostrar=not argumentos.sin_mostrar)


if __name__ == "__main__":
    main()
