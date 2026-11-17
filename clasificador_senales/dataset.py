import csv
import sys
from pathlib import Path

import numpy as np

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasificador_senales.generador_senales import (
    AMPLITUD_MAXIMA,
    AMPLITUD_MINIMA,
    CONFIGURACION,
    FASE_MAXIMA,
    FASE_MINIMA,
    FRECUENCIA_MAXIMA,
    FRECUENCIA_MINIMA,
    FRECUENCIA_MUESTREO,
    NUMERO_MUESTRAS,
    generar_cuadrada,
    generar_senoidal,
    generar_triangular,
)
from clasificador_senales.ruido import agregar_ruido

# La clase 1 contiene senoidales; la clase 0, cuadradas y triangulares.
# Las cantidades actuales dan 300 ejemplos por clase.
CANTIDADES_SENALES = CONFIGURACION["Cantidad de senales"]
CANTIDAD_SENOIDALES = CANTIDADES_SENALES["Cantidad de senoidales"]
CANTIDAD_CUADRADAS = CANTIDADES_SENALES["Cantidad de cuadradas"]
CANTIDAD_TRIANGULARES = CANTIDADES_SENALES["Cantidad de triangulares"]
SEMILLA = CONFIGURACION["Semilla"]

# Usamos la activación y las intensidades definidas en configuracion.yml.
PARAMETROS_RUIDO = CONFIGURACION["Ruido"]
INTENSIDADES_RUIDO = PARAMETROS_RUIDO["Intensidad de ruido"]
RUIDO_ACTIVADO = PARAMETROS_RUIDO["Activacion de ruido"]
INTENSIDAD_RUIDO_MINIMA = INTENSIDADES_RUIDO["Ruido minimo"]
INTENSIDAD_RUIDO_MAXIMA = INTENSIDADES_RUIDO["Ruido maximo"]


CARPETA_DATOS = Path(__file__).resolve().parent / "datos"
RUTA_DATASET = CARPETA_DATOS / "dataset_senales.csv"


def _validar_cantidades(cantidad_senoidales, cantidad_cuadradas, cantidad_triangulares):
    cantidades = (
        cantidad_senoidales,
        cantidad_cuadradas,
        cantidad_triangulares,
    )

    if any(not isinstance(cantidad, int) for cantidad in cantidades):
        raise TypeError("Las cantidades de señales deben ser números enteros.")

    if any(cantidad < 0 for cantidad in cantidades):
        raise ValueError("Las cantidades de señales no pueden ser negativas.")

    if sum(cantidades) == 0:
        raise ValueError("El dataset debe contener por lo menos una señal.")


def _crear_orden_de_senales(
    cantidad_senoidales,
    cantidad_cuadradas,
    cantidad_triangulares,
    generador_aleatorio,
):
    """Crea y mezcla la lista de tipos que tendrá el dataset."""

    tipos = (
        [("senoidal", 1, generar_senoidal)] * cantidad_senoidales
        + [("cuadrada", 0, generar_cuadrada)] * cantidad_cuadradas
        + [("triangular", 0, generar_triangular)] * cantidad_triangulares
    )

    generador_aleatorio.shuffle(tipos)
    return tipos


def crear_dataset(
    ruta_salida=RUTA_DATASET,
    cantidad_senoidales=CANTIDAD_SENOIDALES,
    cantidad_cuadradas=CANTIDAD_CUADRADAS,
    cantidad_triangulares=CANTIDAD_TRIANGULARES,
    ruido_activado=RUIDO_ACTIVADO,
    semilla=SEMILLA,
):
    """
    Genera señales etiquetadas y las guarda en un archivo CSV.

    Cada fila representa una señal completa. Primero aparecen sus datos
    descriptivos y después sus 1,024 muestras.
    """

    _validar_cantidades(
        cantidad_senoidales,
        cantidad_cuadradas,
        cantidad_triangulares,
    )

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    generador_aleatorio = np.random.default_rng(semilla)
    tipos = _crear_orden_de_senales(
        cantidad_senoidales,
        cantidad_cuadradas,
        cantidad_triangulares,
        generador_aleatorio,
    )

    columnas_informacion = [
        "id",
        "tipo",
        "clase",
        "frecuencia_hz",
        "amplitud",
        "fase_grados",
        "frecuencia_muestreo_hz",
        "numero_muestras",
        "ruido_activado",
        "intensidad_ruido",
    ]
    columnas_muestras = [
        f"muestra_{indice:04d}" for indice in range(NUMERO_MUESTRAS)
    ]

    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(columnas_informacion + columnas_muestras)

        for numero, (tipo, clase, funcion_generadora) in enumerate(tipos, start=1):
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

            _, muestras_limpias = funcion_generadora(
                frecuencia=frecuencia,
                amplitud=amplitud,
                fase_grados=fase_grados,
            )

            if ruido_activado:
                intensidad_ruido = generador_aleatorio.uniform(
                    INTENSIDAD_RUIDO_MINIMA,
                    INTENSIDAD_RUIDO_MAXIMA,
                )
                semilla_ruido = int(
                    generador_aleatorio.integers(0, np.iinfo(np.uint32).max)
                )
            else:
                intensidad_ruido = 0.0
                semilla_ruido = None

            muestras = agregar_ruido(
                muestras=muestras_limpias,
                intensidad=intensidad_ruido,
                activado=ruido_activado,
                semilla=semilla_ruido,
            )

            informacion = [
                f"senal_{numero:06d}",
                tipo,
                clase,
                f"{frecuencia:.6f}",
                f"{amplitud:.6f}",
                f"{fase_grados:.6f}",
                FRECUENCIA_MUESTREO,
                NUMERO_MUESTRAS,
                int(ruido_activado),
                f"{intensidad_ruido:.6f}",
            ]
            valores_muestras = [f"{valor:.8f}" for valor in muestras]

            escritor.writerow(informacion + valores_muestras)

    cantidad_total = len(tipos)
    return {
        "ruta": ruta_salida,
        "cantidad_total": cantidad_total,
        "cantidad_clase_1": cantidad_senoidales,
        "cantidad_clase_0": cantidad_cuadradas + cantidad_triangulares,
        "numero_muestras": NUMERO_MUESTRAS,
        "ruido_activado": ruido_activado,
    }


def main():
    resumen = crear_dataset()

    print("Dataset generado correctamente")
    print(f'Archivo: {resumen["ruta"]}')
    print(f'Total de señales: {resumen["cantidad_total"]}')
    print(f'Senoidales, clase 1: {resumen["cantidad_clase_1"]}')
    print(f'No senoidales, clase 0: {resumen["cantidad_clase_0"]}')
    print(f'Muestras por señal: {resumen["numero_muestras"]}')
    estado_ruido = "activado" if resumen["ruido_activado"] else "desactivado"
    print(f"Ruido: {estado_ruido}")


if __name__ == "__main__":
    main()
