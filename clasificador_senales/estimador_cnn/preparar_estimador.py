import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasificador_senales.generador_senales import (
    AMPLITUD_MAXIMA,
    AMPLITUD_MINIMA,
    FRECUENCIA_MAXIMA,
    FRECUENCIA_MINIMA,
    NUMERO_MUESTRAS,
)
from clasificador_senales.dataset import RUTA_DATASET, SEMILLA
from clasificador_senales.clasificador_cnn.division_datos import (
    PROPORCION_ENTRENAMIENTO,
    PROPORCION_PRUEBA,
    PROPORCION_VALIDACION,
    TAMANO_LOTE,
    _validar_proporciones,
)


def _normalizar(valor, minimo, maximo):
    return (valor - minimo) / (maximo - minimo)


def codificar_objetivos(parametros_reales):
    """
    Convierte frecuencia, amplitud y fase a valores comodos para la CNN.

    La frecuencia y la amplitud quedan entre 0 y 1. La fase se representa
    mediante seno y coseno para evitar un salto artificial entre 359 y 0.
    """

    parametros_reales = np.asarray(parametros_reales, dtype=np.float32)
    if parametros_reales.ndim != 2 or parametros_reales.shape[1] != 3:
        raise ValueError(
            "Los parametros deben tener la forma "
            "(cantidad, frecuencia, amplitud y fase)."
        )

    frecuencia = parametros_reales[:, 0]
    amplitud = parametros_reales[:, 1]
    fase_radianes = np.deg2rad(parametros_reales[:, 2])

    return np.column_stack(
        (
            _normalizar(frecuencia, FRECUENCIA_MINIMA, FRECUENCIA_MAXIMA),
            _normalizar(amplitud, AMPLITUD_MINIMA, AMPLITUD_MAXIMA),
            np.sin(fase_radianes),
            np.cos(fase_radianes),
        )
    ).astype(np.float32)


def decodificar_predicciones(predicciones):
    """Convierte las cuatro salidas internas a Hz, amplitud y grados."""

    predicciones = np.asarray(predicciones, dtype=np.float32)
    if predicciones.ndim != 2 or predicciones.shape[1] != 4:
        raise ValueError("Las predicciones deben tener la forma (cantidad, 4).")

    frecuencia_normalizada = np.clip(predicciones[:, 0], 0.0, 1.0)
    amplitud_normalizada = np.clip(predicciones[:, 1], 0.0, 1.0)

    frecuencia = (
        FRECUENCIA_MINIMA
        + frecuencia_normalizada * (FRECUENCIA_MAXIMA - FRECUENCIA_MINIMA)
    )
    amplitud = (
        AMPLITUD_MINIMA
        + amplitud_normalizada * (AMPLITUD_MAXIMA - AMPLITUD_MINIMA)
    )
    fase = np.mod(
        np.rad2deg(np.arctan2(predicciones[:, 2], predicciones[:, 3])),
        360.0,
    )

    return np.column_stack((frecuencia, amplitud, fase)).astype(np.float32)


def cargar_senoidales(ruta_dataset=RUTA_DATASET):
    """Lee el CSV y conserva solo las senoidales con sus parametros reales."""

    ruta_dataset = Path(ruta_dataset)
    if not ruta_dataset.is_file():
        raise FileNotFoundError(f"No se encontro el dataset: {ruta_dataset}")

    with ruta_dataset.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames is None:
            raise ValueError("El CSV no contiene encabezados.")

        columnas_muestras = [
            nombre for nombre in lector.fieldnames if nombre.startswith("muestra_")
        ]
        columnas_esperadas = [
            f"muestra_{indice:04d}" for indice in range(NUMERO_MUESTRAS)
        ]
        columnas_necesarias = {
            "tipo",
            "frecuencia_hz",
            "amplitud",
            "fase_grados",
        }
        if not columnas_necesarias.issubset(set(lector.fieldnames)):
            raise ValueError("El CSV no contiene todos los parametros necesarios.")
        if columnas_muestras != columnas_esperadas:
            raise ValueError(
                f"El CSV debe contener {NUMERO_MUESTRAS} muestras en orden."
            )

        filas = [fila for fila in lector if fila["tipo"] == "senoidal"]

    if len(filas) < 3:
        raise ValueError(
            "Se necesitan al menos tres senoidales para entrenamiento, "
            "validacion y prueba."
        )

    try:
        muestras = np.asarray(
            [
                [float(fila[columna]) for columna in columnas_muestras]
                for fila in filas
            ],
            dtype=np.float32,
        )
        parametros_reales = np.asarray(
            [
                [
                    float(fila["frecuencia_hz"]),
                    float(fila["amplitud"]),
                    float(fila["fase_grados"]),
                ]
                for fila in filas
            ],
            dtype=np.float32,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("El CSV contiene valores no validos.") from error

    if not np.all(np.isfinite(muestras)) or not np.all(
        np.isfinite(parametros_reales)
    ):
        raise ValueError("El dataset contiene valores infinitos o no validos.")

    return muestras[:, np.newaxis, :], parametros_reales


def dividir_senoidales(muestras, parametros_reales, semilla=SEMILLA):
    """Divide las senoidales segun las proporciones de configuracion.yml."""

    muestras = np.asarray(muestras, dtype=np.float32)
    parametros_reales = np.asarray(parametros_reales, dtype=np.float32)

    if muestras.ndim != 3 or muestras.shape[1] != 1:
        raise ValueError("Las muestras deben tener la forma (cantidad, 1, muestras).")
    if parametros_reales.ndim != 2 or parametros_reales.shape[1] != 3:
        raise ValueError("Debe haber frecuencia, amplitud y fase por senal.")
    if len(muestras) != len(parametros_reales):
        raise ValueError("Debe existir un grupo de parametros por cada senal.")
    if len(muestras) < 3:
        raise ValueError("Se necesitan por lo menos tres senoidales.")

    _validar_proporciones(
        PROPORCION_ENTRENAMIENTO,
        PROPORCION_VALIDACION,
        PROPORCION_PRUEBA,
    )

    generador = np.random.default_rng(semilla)
    indices = generador.permutation(len(muestras))
    cantidad_entrenamiento = int(len(indices) * PROPORCION_ENTRENAMIENTO)
    cantidad_validacion = int(len(indices) * PROPORCION_VALIDACION)
    cantidad_prueba = len(indices) - cantidad_entrenamiento - cantidad_validacion

    if min(cantidad_entrenamiento, cantidad_validacion, cantidad_prueba) <= 0:
        raise ValueError("El dataset es demasiado pequeno para dividirlo.")

    inicio_validacion = cantidad_entrenamiento
    inicio_prueba = cantidad_entrenamiento + cantidad_validacion
    grupos_indices = {
        "entrenamiento": indices[:inicio_validacion],
        "validacion": indices[inicio_validacion:inicio_prueba],
        "prueba": indices[inicio_prueba:],
    }

    divisiones = {}
    for nombre, indices_grupo in grupos_indices.items():
        reales = parametros_reales[indices_grupo].copy()
        divisiones[nombre] = {
            "muestras": muestras[indices_grupo].copy(),
            "objetivos": codificar_objetivos(reales),
            "parametros_reales": reales,
        }
    return divisiones


def preparar_datos_estimacion(ruta_dataset=RUTA_DATASET, semilla=SEMILLA):
    """Carga las senoidales y forma las tres divisiones."""

    muestras, parametros_reales = cargar_senoidales(ruta_dataset)
    return dividir_senoidales(muestras, parametros_reales, semilla)


def crear_cargadores_estimacion(
    divisiones,
    tamano_lote=TAMANO_LOTE,
    semilla=SEMILLA,
):
    """Convierte cada division en lotes utilizables por PyTorch."""

    if not isinstance(tamano_lote, int) or isinstance(tamano_lote, bool):
        raise TypeError("El tamano del lote debe ser un numero entero.")
    if tamano_lote <= 0:
        raise ValueError("El tamano del lote debe ser mayor que cero.")

    nombres = {"entrenamiento", "validacion", "prueba"}
    if set(divisiones) != nombres:
        raise ValueError("Faltan divisiones de entrenamiento, validacion o prueba.")

    cargadores = {}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        grupo = divisiones[nombre]
        dataset = TensorDataset(
            torch.as_tensor(grupo["muestras"], dtype=torch.float32),
            torch.as_tensor(grupo["objetivos"], dtype=torch.float32),
        )
        generador = None
        if nombre == "entrenamiento":
            generador = torch.Generator().manual_seed(semilla)
        cargadores[nombre] = DataLoader(
            dataset,
            batch_size=tamano_lote,
            shuffle=(nombre == "entrenamiento"),
            drop_last=False,
            generator=generador,
        )
    return cargadores


def preparar_cargadores_estimacion(
    ruta_dataset=RUTA_DATASET,
    tamano_lote=TAMANO_LOTE,
    semilla=SEMILLA,
):
    """Realiza toda la preparacion en una sola llamada."""

    divisiones = preparar_datos_estimacion(ruta_dataset, semilla)
    return crear_cargadores_estimacion(divisiones, tamano_lote, semilla)


def main():
    divisiones = preparar_datos_estimacion()
    cargadores = crear_cargadores_estimacion(divisiones)

    print("Senoidales preparadas para estimar parametros")
    print(f"Archivo: {RUTA_DATASET}")
    print(f"Tamano configurado del lote: {TAMANO_LOTE}")
    for nombre in ("entrenamiento", "validacion", "prueba"):
        cantidad = len(divisiones[nombre]["muestras"])
        print(
            f"{nombre.capitalize()}: {cantidad} senales, "
            f"{len(cargadores[nombre])} lotes"
        )

    muestras, objetivos = next(iter(cargadores["entrenamiento"]))
    print(f"Forma de un lote de senales: {tuple(muestras.shape)}")
    print(f"Forma de sus respuestas: {tuple(objetivos.shape)}")


if __name__ == "__main__":
    main()
