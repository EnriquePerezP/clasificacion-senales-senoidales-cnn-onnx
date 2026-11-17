from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasificador_senales.generador_senales import (
    AMPLITUD_MAXIMA,
    AMPLITUD_MINIMA,
    FASE_MAXIMA,
    FASE_MINIMA,
    FRECUENCIA_MAXIMA,
    FRECUENCIA_MINIMA,
    generar_cuadrada,
    generar_senoidal,
    generar_triangular,
)


CANTIDAD_POR_TIPO = 2
TIEMPO_VISIBLE = 0.030
SEMILLA = 45


def generar_parametros_aleatorios(generador_aleatorio):
    """Selecciona una frecuencia, amplitud y fase dentro de los límites."""

    frecuencia = generador_aleatorio.uniform(
        FRECUENCIA_MINIMA,
        FRECUENCIA_MAXIMA,
    )

    amplitud = generador_aleatorio.uniform(
        AMPLITUD_MINIMA,
        AMPLITUD_MAXIMA,
    )

    fase_grados = generador_aleatorio.uniform(
        FASE_MINIMA,
        FASE_MAXIMA,
    )

    return frecuencia, amplitud, fase_grados


def generar_ejemplos_aleatorios(cantidad_por_tipo, semilla=None):
    """Genera ejemplos senoidales, cuadrados y triangulares etiquetados."""

    generador_aleatorio = np.random.default_rng(semilla)

    tipos_senal = [
        ("senoidal", 1, generar_senoidal),
        ("cuadrada", 0, generar_cuadrada),
        ("triangular", 0, generar_triangular),
    ]

    ejemplos = []
    identificador = 1

    for nombre, clase, generador in tipos_senal:
        for _ in range(cantidad_por_tipo):
            frecuencia, amplitud, fase_grados = generar_parametros_aleatorios(
                generador_aleatorio
            )

            tiempo, muestras = generador(
                frecuencia=frecuencia,
                amplitud=amplitud,
                fase_grados=fase_grados,
            )

            ejemplos.append(
                {
                    "id": identificador,
                    "tipo": nombre,
                    "clase": clase,
                    "frecuencia": frecuencia,
                    "amplitud": amplitud,
                    "fase_grados": fase_grados,
                    "tiempo": tiempo,
                    "muestras": muestras,
                }
            )

            identificador += 1

    # Mezclamos el orden para que las clases no queden agrupadas.
    orden_aleatorio = generador_aleatorio.permutation(len(ejemplos))
    ejemplos = [ejemplos[indice] for indice in orden_aleatorio]

    return ejemplos


def mostrar_ejemplos(ejemplos):
    """Imprime los parámetros y dibuja cada ejemplo generado."""

    figura, graficas = plt.subplots(
        len(ejemplos),
        1,
        figsize=(11, 2.5 * len(ejemplos)),
    )

    graficas = np.atleast_1d(graficas)

    for grafica, datos in zip(graficas, ejemplos):
        tiempo = datos["tiempo"]
        muestras = datos["muestras"]
        muestras_visibles = tiempo <= TIEMPO_VISIBLE

        print(
            f'Ejemplo {datos["id"]}: '
            f'tipo={datos["tipo"]}, '
            f'clase={datos["clase"]}, '
            f'frecuencia={datos["frecuencia"]:.2f} Hz, '
            f'amplitud={datos["amplitud"]:.2f}, '
            f'fase={datos["fase_grados"]:.2f}°'
        )

        grafica.plot(
            tiempo[muestras_visibles] * 1000,
            muestras[muestras_visibles],
        )

        grafica.set_title(
            f'{datos["tipo"].capitalize()} | '
            f'clase {datos["clase"]} | '
            f'{datos["frecuencia"]:.2f} Hz | '
            f'amplitud {datos["amplitud"]:.2f} | '
            f'fase {datos["fase_grados"]:.2f}°'
        )
        grafica.set_xlabel("Tiempo en milisegundos")
        grafica.set_ylabel("Amplitud")
        grafica.grid(True)

    figura.suptitle("Tipos de señales con parámetros aleatorios", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()


def main():
    ejemplos = generar_ejemplos_aleatorios(
        cantidad_por_tipo=CANTIDAD_POR_TIPO,
        semilla=SEMILLA,
    )

    mostrar_ejemplos(ejemplos)


if __name__ == "__main__":
    main()

