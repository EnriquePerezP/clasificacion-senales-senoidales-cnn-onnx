import sys
from math import isfinite
from numbers import Real
from pathlib import Path

import torch
from torch import nn

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from clasificador_senales.clasificador_cnn.division_datos import (
    PARAMETROS_ENTRENAMIENTO,
    preparar_cargadores,
)
from clasificador_senales.clasificador_cnn.modelo_cnn import DetectorSenoidalCNN
from clasificador_senales.dataset import SEMILLA


TASA_APRENDIZAJE = PARAMETROS_ENTRENAMIENTO["Tasa de aprendizaje"]
NUMERO_EPOCAS = PARAMETROS_ENTRENAMIENTO["Numero de epocas"]

CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
RUTA_MEJOR_MODELO = CARPETA_MODELOS / "detector_senoidal_cnn.pt"


def crear_componentes_entrenamiento(modelo, tasa_aprendizaje=TASA_APRENDIZAJE):
    """Crea la función que mide el error y el optimizador que corrige la CNN."""

    if not isinstance(tasa_aprendizaje, Real) or isinstance(tasa_aprendizaje, bool):
        raise TypeError("La tasa de aprendizaje debe ser un número.")
    if not isfinite(tasa_aprendizaje) or tasa_aprendizaje <= 0:
        raise ValueError("La tasa de aprendizaje debe ser finita y mayor que cero.")

    # La CNN tiene dos posibles respuestas: clase 0 o clase 1.
    # BCEWithLogitsLoss compara la salida de la CNN con la respuesta correcta.
    funcion_perdida = nn.BCEWithLogitsLoss()

    # Adam modifica poco a poco los parámetros internos para reducir el error.
    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=tasa_aprendizaje,
    )

    return funcion_perdida, optimizador


def entrenar_una_epoca(
    modelo,
    cargador_entrenamiento,
    funcion_perdida,
    optimizador,
    dispositivo=None,
):
    """Hace que la CNN recorra una vez todos los lotes de entrenamiento."""

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    modelo.to(dispositivo)
    modelo.train()

    perdida_acumulada = 0.0
    cantidad_aciertos = 0
    cantidad_ejemplos = 0

    for muestras, clases_correctas in cargador_entrenamiento:
        muestras = muestras.to(dispositivo)
        clases_correctas = clases_correctas.to(dispositivo)

        # Eliminamos las correcciones calculadas para el lote anterior.
        optimizador.zero_grad()

        # 1. La CNN intenta clasificar las señales del lote.
        salidas = modelo(muestras)

        # 2. Medimos la diferencia con las respuestas correctas.
        perdida = funcion_perdida(salidas, clases_correctas)

        # 3. PyTorch calcula qué parámetros contribuyeron al error.
        perdida.backward()

        # 4. El optimizador realiza una pequeña corrección.
        optimizador.step()

        cantidad_lote = len(clases_correctas)
        perdida_acumulada += perdida.item() * cantidad_lote
        cantidad_ejemplos += cantidad_lote

        # Una salida mayor o igual que cero corresponde a la clase 1.
        # Una salida menor que cero corresponde a la clase 0.
        predicciones = (salidas >= 0).to(dtype=clases_correctas.dtype)
        cantidad_aciertos += int((predicciones == clases_correctas).sum().item())

    if cantidad_ejemplos == 0:
        raise ValueError("El cargador de entrenamiento está vacío.")

    return {
        "perdida": perdida_acumulada / cantidad_ejemplos,
        "exactitud": cantidad_aciertos / cantidad_ejemplos,
        "cantidad_ejemplos": cantidad_ejemplos,
    }


def validar_modelo(
    modelo,
    cargador_validacion,
    funcion_perdida,
    dispositivo=None,
):
    """
    Mide el resultado con señales de validación sin modificar la CNN.

    Esta función se parece al recorrido de entrenamiento, pero no calcula
    correcciones ni utiliza un optimizador.
    """

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    modelo.to(dispositivo)
    modelo.eval()

    perdida_acumulada = 0.0
    cantidad_aciertos = 0
    cantidad_ejemplos = 0

    # no_grad evita que PyTorch prepare correcciones durante la validación.
    with torch.no_grad():
        for muestras, clases_correctas in cargador_validacion:
            muestras = muestras.to(dispositivo)
            clases_correctas = clases_correctas.to(dispositivo)

            salidas = modelo(muestras)
            perdida = funcion_perdida(salidas, clases_correctas)

            cantidad_lote = len(clases_correctas)
            perdida_acumulada += perdida.item() * cantidad_lote
            cantidad_ejemplos += cantidad_lote

            predicciones = (salidas >= 0).to(dtype=clases_correctas.dtype)
            cantidad_aciertos += int(
                (predicciones == clases_correctas).sum().item()
            )

    if cantidad_ejemplos == 0:
        raise ValueError("El cargador de validación está vacío.")

    return {
        "perdida": perdida_acumulada / cantidad_ejemplos,
        "exactitud": cantidad_aciertos / cantidad_ejemplos,
        "cantidad_ejemplos": cantidad_ejemplos,
    }


def guardar_modelo(
    ruta_salida,
    modelo,
    optimizador,
    epoca,
    resultado_validacion,
):
    """Guarda el estado necesario para utilizar o continuar el modelo."""

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoca": epoca,
            "numero_muestras": modelo.numero_muestras,
            "estado_modelo": modelo.state_dict(),
            "estado_optimizador": optimizador.state_dict(),
            "perdida_validacion": resultado_validacion["perdida"],
            "exactitud_validacion": resultado_validacion["exactitud"],
        },
        ruta_salida,
    )


def entrenar_varias_epocas(
    modelo,
    cargadores,
    funcion_perdida,
    optimizador,
    numero_epocas=NUMERO_EPOCAS,
    ruta_mejor_modelo=RUTA_MEJOR_MODELO,
    dispositivo=None,
    mostrar_progreso=True,
):
    """Entrena, valida y guarda la versión con menor pérdida de validación."""

    if not isinstance(numero_epocas, int) or isinstance(numero_epocas, bool):
        raise TypeError("El número de épocas debe ser un número entero.")

    if numero_epocas <= 0:
        raise ValueError("El número de épocas debe ser mayor que cero.")

    if "entrenamiento" not in cargadores or "validacion" not in cargadores:
        raise ValueError("Se necesitan los cargadores de entrenamiento y validación.")

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    mejor_perdida_validacion = float("inf")
    mejor_epoca = 0
    historial = []

    for epoca in range(1, numero_epocas + 1):
        resultado_entrenamiento = entrenar_una_epoca(
            modelo=modelo,
            cargador_entrenamiento=cargadores["entrenamiento"],
            funcion_perdida=funcion_perdida,
            optimizador=optimizador,
            dispositivo=dispositivo,
        )
        resultado_validacion = validar_modelo(
            modelo=modelo,
            cargador_validacion=cargadores["validacion"],
            funcion_perdida=funcion_perdida,
            dispositivo=dispositivo,
        )

        es_mejor_modelo = (
            resultado_validacion["perdida"] < mejor_perdida_validacion
        )
        if es_mejor_modelo:
            mejor_perdida_validacion = resultado_validacion["perdida"]
            mejor_epoca = epoca
            guardar_modelo(
                ruta_salida=ruta_mejor_modelo,
                modelo=modelo,
                optimizador=optimizador,
                epoca=epoca,
                resultado_validacion=resultado_validacion,
            )

        historial.append(
            {
                "epoca": epoca,
                "perdida_entrenamiento": resultado_entrenamiento["perdida"],
                "exactitud_entrenamiento": resultado_entrenamiento["exactitud"],
                "perdida_validacion": resultado_validacion["perdida"],
                "exactitud_validacion": resultado_validacion["exactitud"],
                "es_mejor_modelo": es_mejor_modelo,
            }
        )

        if mostrar_progreso:
            marca_mejor = "  <- mejor modelo guardado" if es_mejor_modelo else ""
            print(
                f"Época {epoca:02d}/{numero_epocas} | "
                f"entrenamiento: {resultado_entrenamiento['exactitud']:.2%} | "
                f"validación: {resultado_validacion['exactitud']:.2%} | "
                f"pérdida validación: {resultado_validacion['perdida']:.4f}"
                f"{marca_mejor}"
            )

    return {
        "historial": historial,
        "mejor_epoca": mejor_epoca,
        "mejor_perdida_validacion": mejor_perdida_validacion,
        "ruta_mejor_modelo": Path(ruta_mejor_modelo),
    }


def main():
    """Entrena varias épocas y guarda la mejor versión encontrada."""

    torch.manual_seed(SEMILLA)

    cargadores = preparar_cargadores(semilla=SEMILLA)
    modelo = DetectorSenoidalCNN()
    funcion_perdida, optimizador = crear_componentes_entrenamiento(modelo)

    print("Inicio del entrenamiento")
    print(f"Señales de entrenamiento: {len(cargadores['entrenamiento'].dataset)}")
    print(f"Señales de validación: {len(cargadores['validacion'].dataset)}")
    print(f"Tamaño del lote: {cargadores['entrenamiento'].batch_size}")
    print(f"Número de épocas: {NUMERO_EPOCAS}")
    print(f"Tasa de aprendizaje: {TASA_APRENDIZAJE}")

    resultado = entrenar_varias_epocas(
        modelo=modelo,
        cargadores=cargadores,
        funcion_perdida=funcion_perdida,
        optimizador=optimizador,
        numero_epocas=NUMERO_EPOCAS,
        ruta_mejor_modelo=RUTA_MEJOR_MODELO,
    )

    print("Entrenamiento terminado")
    print(f"Mejor época: {resultado['mejor_epoca']}")
    print(f"Menor pérdida de validación: {resultado['mejor_perdida_validacion']:.4f}")
    print(f"Modelo guardado en: {resultado['ruta_mejor_modelo']}")


if __name__ == "__main__":
    main()
