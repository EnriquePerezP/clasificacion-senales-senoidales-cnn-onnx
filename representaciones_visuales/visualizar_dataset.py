"""Busca una señal del dataset por su identificador y la muestra en una gráfica."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Permite ejecutar el archivo directamente, incluso desde otra carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelos.dataset import RUTA_DATASET


def buscar_senal_por_id(identificador, ruta_dataset=RUTA_DATASET):
    """Devuelve los datos de una señal concreta del CSV."""

    ruta_dataset = Path(ruta_dataset)
    try:
        archivo = ruta_dataset.open(newline="", encoding="utf-8-sig")
    except FileNotFoundError as error:
        mensaje = f"No existe el dataset en {ruta_dataset}."
        if ruta_dataset == RUTA_DATASET:
            mensaje += " Genéralo con python -m modelos.dataset."
        raise FileNotFoundError(mensaje) from error

    with archivo:
        lector = csv.DictReader(archivo, strict=True)
        encabezados = lector.fieldnames or []
        if "id" not in encabezados:
            raise ValueError("El CSV debe contener la columna 'id'.")
        if len(set(encabezados)) != len(encabezados):
            raise ValueError("El CSV contiene columnas con nombres repetidos.")
        fila = next((fila for fila in lector if fila["id"] == identificador), None)

    if fila is None:
        raise ValueError(f"No se encontró el identificador {identificador}.")
    if None in fila or None in fila.values():
        raise ValueError(f"La fila de {identificador} tiene valores faltantes o de más.")

    try:
        datos = {
            "id": fila["id"],
            "tipo": fila["tipo"],
            "clase": int(fila["clase"]),
            "frecuencia_hz": float(fila["frecuencia_hz"]),
            "amplitud": float(fila["amplitud"]),
            "fase_grados": float(fila["fase_grados"]),
            "frecuencia_muestreo_hz": float(fila["frecuencia_muestreo_hz"]),
            "numero_muestras": int(fila["numero_muestras"]),
            "ruido_activado": bool(int(fila["ruido_activado"])),
            "intensidad_ruido": float(fila["intensidad_ruido"]),
        }
    except KeyError as error:
        raise ValueError(f"Falta la columna {error} en el CSV.") from error
    except ValueError as error:
        raise ValueError(
            f"Los parámetros de {identificador} contienen valores no válidos."
        ) from error

    frecuencia_muestreo = datos["frecuencia_muestreo_hz"]
    if not np.isfinite(frecuencia_muestreo) or frecuencia_muestreo <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva y finita.")

    columnas_muestras = [
        nombre for nombre in encabezados if nombre.startswith("muestra_")
    ]
    numero_muestras = datos["numero_muestras"]
    if not columnas_muestras or numero_muestras != len(columnas_muestras):
        raise ValueError(
            f"{identificador} declara {numero_muestras} muestras y se encontraron "
            f"{len(columnas_muestras)} columnas de muestras en el CSV. "
            "La cantidad debe coincidir y ser mayor que cero."
        )

    columnas_ordenadas = [
        f"muestra_{indice:04d}" for indice in range(numero_muestras)
    ]
    if set(columnas_ordenadas) != set(columnas_muestras):
        raise ValueError(
            "Las columnas de muestras deben numerarse consecutivamente desde muestra_0000."
        )

    try:
        muestras = np.array([float(fila[columna]) for columna in columnas_ordenadas])
    except ValueError as error:
        raise ValueError(
            f"La señal {identificador} contiene muestras vacías o no numéricas."
        ) from error

    if not np.all(np.isfinite(muestras)):
        raise ValueError(f"La señal {identificador} contiene muestras no finitas.")
    datos["muestras"] = muestras
    return datos


def mostrar_informacion(datos):
    """Imprime en la terminal los parámetros guardados de la señal."""

    estado_ruido = "sí" if datos["ruido_activado"] else "no"

    print("Señal encontrada")
    print(f'ID: {datos["id"]}')
    print(f'Tipo: {datos["tipo"]}')
    print(f'Clase: {datos["clase"]}')
    print(f'Frecuencia: {datos["frecuencia_hz"]:.2f} Hz')
    print(f'Amplitud: {datos["amplitud"]:.2f}')
    print(f'Fase: {datos["fase_grados"]:.2f} grados')
    print(f'Muestras: {datos["numero_muestras"]}')
    print(f'Ruido activado: {estado_ruido}')
    print(f'Intensidad del ruido: {datos["intensidad_ruido"]:.2f}')


def graficar_senal(datos):
    """Abre una gráfica temporal de la señal seleccionada."""

    import matplotlib.pyplot as plt

    tiempo_segundos = np.arange(datos["numero_muestras"]) / datos[
        "frecuencia_muestreo_hz"
    ]

    plt.figure(figsize=(11, 5))
    plt.plot(tiempo_segundos * 1000, datos["muestras"])
    plt.title(
        f'{datos["id"]} | {datos["tipo"].capitalize()} | '
        f'{datos["frecuencia_hz"]:.2f} Hz'
    )
    plt.xlabel("Tiempo en milisegundos")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    analizador = argparse.ArgumentParser(
        description="Visualiza una señal del dataset mediante su ID."
    )
    analizador.add_argument(
        "identificador",
        help="Identificador de la señal, por ejemplo: senal_000123",
    )
    analizador.add_argument(
        "--archivo",
        type=Path,
        default=RUTA_DATASET,
        help="Ruta de otro dataset CSV, si se desea utilizar uno diferente.",
    )
    argumentos = analizador.parse_args()

    try:
        datos = buscar_senal_por_id(
            identificador=argumentos.identificador,
            ruta_dataset=argumentos.archivo,
        )
    except (OSError, ValueError, csv.Error) as error:
        analizador.error(str(error))

    mostrar_informacion(datos)
    graficar_senal(datos)


if __name__ == "__main__":
    main()
