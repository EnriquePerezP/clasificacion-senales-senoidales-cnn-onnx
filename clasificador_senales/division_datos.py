import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasificador_senales.dataset import RUTA_DATASET, SEMILLA
from clasificador_senales.generador_senales import CONFIGURACION, NUMERO_MUESTRAS


PARAMETROS_DIVISION = CONFIGURACION["Division de datos"]
PARAMETROS_ENTRENAMIENTO = CONFIGURACION["Entrenamiento"]

PROPORCION_ENTRENAMIENTO = PARAMETROS_DIVISION["Proporcion de entrenamiento"]
PROPORCION_VALIDACION = PARAMETROS_DIVISION["Proporcion de validacion"]
PROPORCION_PRUEBA = PARAMETROS_DIVISION["Proporcion de prueba"]
TAMANO_LOTE = PARAMETROS_ENTRENAMIENTO["Tamano de lote"]



def cargar_dataset(ruta_dataset=RUTA_DATASET, incluir_tipos=False):
    """
    Lee el CSV y devuelve las entradas y respuestas correctas de la CNN.

    Si ``incluir_tipos`` es verdadero también devuelve los nombres originales:
    senoidal, cuadrada o triangular. Se utilizarán solamente para la gráfica.

    Formas devueltas:
        muestras: (cantidad_de_señales, 1, numero_de_muestras)
        clases:   (cantidad_de_señales,)

    El segundo valor de la forma de ``muestras`` es 1 porque cada señal
    tiene un solo canal.
    """

    ruta_dataset = Path(ruta_dataset)
    if not ruta_dataset.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {ruta_dataset}")

    with ruta_dataset.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)

        if lector.fieldnames is None:
            raise ValueError("El CSV no contiene encabezados.")

        columnas_muestras = [
            nombre
            for nombre in lector.fieldnames
            if nombre.startswith("muestra_")
        ]
        columnas_esperadas = [
            f"muestra_{indice:04d}" for indice in range(NUMERO_MUESTRAS)
        ]

        if "clase" not in lector.fieldnames:
            raise ValueError("El CSV no contiene la columna 'clase'.")

        if incluir_tipos and "tipo" not in lector.fieldnames:
            raise ValueError("El CSV no contiene la columna 'tipo'.")

        if columnas_muestras != columnas_esperadas:
            raise ValueError(
                f"El CSV debe contener {NUMERO_MUESTRAS} columnas de muestras "
                "en el orden esperado."
            )

        filas = list(lector)

    if not filas:
        raise ValueError("El dataset no contiene señales.")

    try:
        muestras = np.asarray(
            [
                [float(fila[columna]) for columna in columnas_muestras]
                for fila in filas
            ],
            dtype=np.float32,
        )
        clases = np.asarray(
            [int(fila["clase"]) for fila in filas],
            dtype=np.int64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("El CSV contiene muestras o clases no válidas.") from error

    if not np.all(np.isfinite(muestras)):
        raise ValueError("El dataset contiene muestras infinitas o no válidas.")

    clases_encontradas = set(clases.tolist())
    if not clases_encontradas.issubset({0, 1}):
        raise ValueError("Las clases del dataset solo pueden ser 0 o 1.")

    if clases_encontradas != {0, 1}:
        raise ValueError("El dataset necesita ejemplos de las clases 0 y 1.")

    # Añadimos la dimensión del canal que necesita Conv1d:
    # (cantidad, muestras) pasa a ser (cantidad, 1, muestras).
    muestras = muestras[:, np.newaxis, :]

    if incluir_tipos:
        tipos = np.asarray([fila["tipo"] for fila in filas])
        return muestras, clases, tipos

    return muestras, clases


def _validar_proporciones(
    proporcion_entrenamiento,
    proporcion_validacion,
    proporcion_prueba,
):
    proporciones = (
        proporcion_entrenamiento,
        proporcion_validacion,
        proporcion_prueba,
    )

    if any(proporcion <= 0 or proporcion >= 1 for proporcion in proporciones):
        raise ValueError("Cada proporción debe ser mayor que 0 y menor que 1.")

    if not np.isclose(sum(proporciones), 1.0):
        raise ValueError("Las proporciones deben sumar 1.0.")


def dividir_dataset(
    muestras,
    clases,
    proporcion_entrenamiento=PROPORCION_ENTRENAMIENTO,
    proporcion_validacion=PROPORCION_VALIDACION,
    proporcion_prueba=PROPORCION_PRUEBA,
    semilla=SEMILLA,
    tipos=None,
):
    """
    Separa los datos conservando la proporción de las clases 0 y 1.

    Devuelve un diccionario con tres grupos: ``entrenamiento``,
    ``validacion`` y ``prueba``. Cada grupo contiene sus muestras y clases.
    """

    muestras = np.asarray(muestras)
    clases = np.asarray(clases)

    if muestras.ndim != 3:
        raise ValueError(
            "Las muestras deben tener la forma "
            "(cantidad_de_señales, 1, numero_de_muestras)."
        )

    if clases.ndim != 1:
        raise ValueError("Las clases deben estar en una sola dimensión.")

    if len(muestras) != len(clases):
        raise ValueError("Debe existir una clase por cada señal.")

    if tipos is not None:
        tipos = np.asarray(tipos)
        if tipos.ndim != 1 or len(tipos) != len(clases):
            raise ValueError("Debe existir un tipo por cada señal.")

    _validar_proporciones(
        proporcion_entrenamiento,
        proporcion_validacion,
        proporcion_prueba,
    )

    generador = np.random.default_rng(semilla)
    indices_por_grupo = {
        "entrenamiento": [],
        "validacion": [],
        "prueba": [],
    }

    # Separamos cada clase por su cuenta. Así los tres grupos mantienen
    # cantidades equivalentes de senoidales y no senoidales.
    for clase in (0, 1):
        indices_clase = np.flatnonzero(clases == clase)

        if len(indices_clase) < 3:
            raise ValueError(
                f"La clase {clase} necesita al menos tres ejemplos para "
                "aparecer en los tres grupos."
            )

        generador.shuffle(indices_clase)

        cantidad_entrenamiento = int(
            len(indices_clase) * proporcion_entrenamiento
        )
        cantidad_validacion = int(
            len(indices_clase) * proporcion_validacion
        )
        inicio_validacion = cantidad_entrenamiento
        inicio_prueba = cantidad_entrenamiento + cantidad_validacion

        indices_por_grupo["entrenamiento"].extend(
            indices_clase[:inicio_validacion]
        )
        indices_por_grupo["validacion"].extend(
            indices_clase[inicio_validacion:inicio_prueba]
        )
        indices_por_grupo["prueba"].extend(indices_clase[inicio_prueba:])

    divisiones = {}
    for nombre, indices in indices_por_grupo.items():
        indices = np.asarray(indices, dtype=np.int64)
        generador.shuffle(indices)
        grupo = {
            "muestras": muestras[indices].copy(),
            "clases": clases[indices].copy(),
        }
        if tipos is not None:
            grupo["tipos"] = tipos[indices].copy()
        divisiones[nombre] = grupo

    return divisiones


def preparar_datos(ruta_dataset=RUTA_DATASET, semilla=SEMILLA):
    """Carga y divide el dataset en una sola llamada."""

    muestras, clases, tipos = cargar_dataset(
        ruta_dataset,
        incluir_tipos=True,
    )
    return dividir_dataset(
        muestras,
        clases,
        semilla=semilla,
        tipos=tipos,
    )


def crear_cargadores(divisiones, tamano_lote=TAMANO_LOTE, semilla=SEMILLA):
    """
    Convierte las divisiones en lotes que PyTorch puede entregar a la CNN.

    El grupo de entrenamiento se mezcla en cada recorrido. Validación y prueba
    conservan su orden porque en ellos el modelo solamente será evaluado.
    """

    if not isinstance(tamano_lote, int) or isinstance(tamano_lote, bool):
        raise TypeError("El tamaño del lote debe ser un número entero.")

    if tamano_lote <= 0:
        raise ValueError("El tamaño del lote debe ser mayor que cero.")

    nombres_esperados = {"entrenamiento", "validacion", "prueba"}
    if set(divisiones) != nombres_esperados:
        raise ValueError(
            "Las divisiones deben contener entrenamiento, validacion y prueba."
        )

    cargadores = {}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        grupo = divisiones[nombre]
        muestras = np.asarray(grupo["muestras"], dtype=np.float32)
        clases = np.asarray(grupo["clases"], dtype=np.float32)

        if muestras.ndim != 3:
            raise ValueError(f"Las muestras de {nombre} no tienen tres dimensiones.")

        if clases.ndim != 1 or len(muestras) != len(clases):
            raise ValueError(f"Las clases de {nombre} no coinciden con sus señales.")

        # La CNN produce una respuesta con forma (cantidad, 1). Añadimos esa
        # segunda dimensión a las clases para que ambas formas coincidan.
        tensor_muestras = torch.from_numpy(muestras.copy())
        tensor_clases = torch.from_numpy(clases.copy()).unsqueeze(dim=1)
        dataset_torch = TensorDataset(tensor_muestras, tensor_clases)

        generador_torch = torch.Generator().manual_seed(semilla)
        cargadores[nombre] = DataLoader(
            dataset=dataset_torch,
            batch_size=tamano_lote,
            shuffle=nombre == "entrenamiento",
            drop_last=False,
            generator=generador_torch,
        )

    return cargadores


def preparar_cargadores(
    ruta_dataset=RUTA_DATASET,
    tamano_lote=TAMANO_LOTE,
    semilla=SEMILLA,
):
    """Carga, divide y organiza el dataset en lotes de PyTorch."""

    divisiones = preparar_datos(ruta_dataset=ruta_dataset, semilla=semilla)
    return crear_cargadores(
        divisiones=divisiones,
        tamano_lote=tamano_lote,
        semilla=semilla,
    )


def _describir_grupo(nombre, grupo):
    cantidades = Counter(grupo["clases"].tolist())
    print(
        f"{nombre.capitalize()}: {len(grupo['clases'])} señales "
        f"(clase 0: {cantidades[0]}, clase 1: {cantidades[1]})"
    )


def main():
    """Muestra un resumen para comprobar la preparación de los datos."""

    divisiones = preparar_datos()
    cargadores = crear_cargadores(divisiones)

    print("Dataset preparado correctamente")
    print(f"Archivo: {RUTA_DATASET}")
    for nombre in ("entrenamiento", "validacion", "prueba"):
        _describir_grupo(nombre, divisiones[nombre])

    forma = divisiones["entrenamiento"]["muestras"].shape
    print(f"Forma de entrada para la CNN: {forma}")

    muestras_lote, clases_lote = next(iter(cargadores["entrenamiento"]))
    print(f"Tamaño configurado del lote: {TAMANO_LOTE}")
    print(f"Forma de un lote de señales: {tuple(muestras_lote.shape)}")
    print(f"Forma de sus respuestas correctas: {tuple(clases_lote.shape)}")

    print(len(cargadores["entrenamiento"]))  # 14

if __name__ == "__main__":
    main()
