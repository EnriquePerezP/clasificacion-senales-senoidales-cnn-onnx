import argparse
import sys
from math import isfinite
from numbers import Real
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelos.clasificador_cnn.division_datos import PARAMETROS_ENTRENAMIENTO
from modelos.dataset import SEMILLA
from modelos.ejecucion_entrenamiento import (
    MedicionEpoca,
    agregar_argumentos_entrenamiento,
    configurar_entrenamiento,
    crear_modelo_inicial,
    guardar_checkpoint,
    guardar_registro,
    resolver_dispositivo,
    resolver_ruta_checkpoint,
    sincronizar,
)
from modelos.estimador_cnn.modelo_estimador import EstimadorParametrosCNN
from modelos.estimador_cnn.preparar_estimador import (
    crear_cargadores_estimacion,
    preparar_datos_estimacion,
)


TASA_APRENDIZAJE = PARAMETROS_ENTRENAMIENTO["Tasa de aprendizaje"]
NUMERO_EPOCAS = PARAMETROS_ENTRENAMIENTO["Numero de epocas"]

CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
RUTA_MEJOR_ESTIMADOR = CARPETA_MODELOS / "estimador_parametros_cnn.pt"


def crear_componentes_entrenamiento(
    modelo,
    tasa_aprendizaje=TASA_APRENDIZAJE,
):
    """Crea la medida de error y el optimizador del estimador."""

    if not isinstance(tasa_aprendizaje, Real) or isinstance(tasa_aprendizaje, bool):
        raise TypeError("La tasa de aprendizaje debe ser un numero.")
    if not isfinite(tasa_aprendizaje) or tasa_aprendizaje <= 0:
        raise ValueError("La tasa de aprendizaje debe ser finita y mayor que cero.")

    # MSELoss mide la distancia entre los cuatro valores esperados y los
    # cuatro valores calculados por la CNN.
    funcion_perdida = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    return funcion_perdida, optimizador


def recorrer_cargador(
    modelo,
    cargador,
    funcion_perdida,
    dispositivo=None,
    optimizador=None,
):
    """
    Recorre un grupo completo.

    Si recibe un optimizador, aprende. Si no lo recibe, solamente valida.
    """

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    modelo.to(dispositivo)
    esta_entrenando = optimizador is not None
    modelo.train(mode=esta_entrenando)

    perdidas = []
    tamanos = []
    cantidad_ejemplos = 0

    contexto = torch.enable_grad() if esta_entrenando else torch.no_grad()
    with contexto:
        for muestras, objetivos in cargador:
            muestras = muestras.to(dispositivo)
            objetivos = objetivos.to(dispositivo)

            if esta_entrenando:
                optimizador.zero_grad()

            predicciones = modelo(muestras)
            perdida = funcion_perdida(predicciones, objetivos)

            if esta_entrenando:
                perdida.backward()
                optimizador.step()

            cantidad_lote = len(objetivos)
            perdidas.append(perdida.detach())
            tamanos.append(cantidad_lote)
            cantidad_ejemplos += cantidad_lote

    if cantidad_ejemplos == 0:
        raise ValueError("El cargador esta vacio.")

    return {
        "perdida": sum(
            perdida * tamano for perdida, tamano in zip(torch.stack(perdidas).tolist(), tamanos)
        ) / cantidad_ejemplos,
        "cantidad_ejemplos": cantidad_ejemplos,
    }


def guardar_estimador(
    ruta_salida,
    modelo,
    optimizador,
    epoca,
    perdida_validacion,
):
    """Guarda la mejor version del estimador y sus datos principales."""

    guardar_checkpoint(
        ruta_salida,
        {
            "epoca": epoca,
            "numero_muestras": modelo.numero_muestras,
            "cantidad_salidas": modelo.CANTIDAD_SALIDAS,
            "estado_modelo": modelo.state_dict(),
            "estado_optimizador": optimizador.state_dict(),
            "perdida_validacion": perdida_validacion,
        },
    )


def entrenar_estimador(
    modelo,
    cargadores,
    funcion_perdida,
    optimizador,
    numero_epocas=NUMERO_EPOCAS,
    ruta_mejor_modelo=None,
    dispositivo=None,
    mostrar_progreso=True,
):
    """Entrena varias epocas y conserva la de menor error de validacion."""

    if not isinstance(numero_epocas, int) or isinstance(numero_epocas, bool):
        raise TypeError("El numero de epocas debe ser un numero entero.")
    if numero_epocas <= 0:
        raise ValueError("El numero de epocas debe ser mayor que cero.")
    if "entrenamiento" not in cargadores or "validacion" not in cargadores:
        raise ValueError("Se necesitan entrenamiento y validacion.")
    dispositivo = torch.device(dispositivo or "cpu")
    ruta_mejor_modelo = resolver_ruta_checkpoint(
        ruta_mejor_modelo, RUTA_MEJOR_ESTIMADOR, dispositivo
    )

    mejor_perdida = float("inf")
    mejor_epoca = 0
    historial = []

    sincronizar(dispositivo)
    inicio = perf_counter()
    for epoca in range(1, numero_epocas + 1):
        with MedicionEpoca(dispositivo) as medicion:
            resultado_entrenamiento = recorrer_cargador(
                modelo,
                cargadores["entrenamiento"],
                funcion_perdida,
                dispositivo,
                optimizador,
            )
            resultado_validacion = recorrer_cargador(
                modelo,
                cargadores["validacion"],
                funcion_perdida,
                dispositivo,
            )

        es_mejor = resultado_validacion["perdida"] < mejor_perdida
        if es_mejor:
            mejor_perdida = resultado_validacion["perdida"]
            mejor_epoca = epoca
            guardar_estimador(
                ruta_mejor_modelo,
                modelo,
                optimizador,
                epoca,
                mejor_perdida,
            )

        historial.append(
            {
                "epoca": epoca,
                "perdida_entrenamiento": resultado_entrenamiento["perdida"],
                "perdida_validacion": resultado_validacion["perdida"],
                "es_mejor_modelo": es_mejor,
                **medicion.datos,
            }
        )

        if mostrar_progreso:
            marca = "  <- mejor estimador guardado" if es_mejor else ""
            print(
                f"Epoca {epoca:02d}/{numero_epocas} | "
                f"error entrenamiento: {resultado_entrenamiento['perdida']:.5f} | "
                f"error validacion: {resultado_validacion['perdida']:.5f} | "
                f"{medicion.datos['tiempo_epoca_s']:.3f} s"
                f"{marca}"
            )

    sincronizar(dispositivo)
    return {
        "tiempo_total_s": perf_counter() - inicio,
        "historial": historial,
        "mejor_epoca": mejor_epoca,
        "mejor_perdida_validacion": mejor_perdida,
        "ruta_mejor_modelo": Path(ruta_mejor_modelo),
        "cargadores": cargadores,
        "tasa_aprendizaje": optimizador.param_groups[0]["lr"],
    }


def main(argv=None):
    """Prepara, entrena y guarda el estimador."""

    from modelos.estimador_cnn.evaluacion_modelo import cargar_estimador, evaluar_estimador

    parser = argparse.ArgumentParser(description="Entrena el estimador CNN.")
    agregar_argumentos_entrenamiento(parser)
    args = parser.parse_args(argv)
    dispositivo = resolver_dispositivo(args.dispositivo)
    configurar_entrenamiento(dispositivo, hilos_cpu=args.hilos_cpu)
    carpeta = args.salida / dispositivo.type / "estimador"

    divisiones = preparar_datos_estimacion(semilla=SEMILLA)
    cargadores = crear_cargadores_estimacion(divisiones, semilla=SEMILLA)
    modelo, huella = crear_modelo_inicial(EstimadorParametrosCNN, dispositivo)
    funcion_perdida, optimizador = crear_componentes_entrenamiento(modelo)

    print(f"Inicio del entrenamiento del estimador en {dispositivo}")
    print(f"Senoidales de entrenamiento: {len(cargadores['entrenamiento'].dataset)}")
    print(f"Senoidales de validacion: {len(cargadores['validacion'].dataset)}")
    print(f"Muestras por senal: {modelo.numero_muestras}")
    print(f"Tamano del lote: {cargadores['entrenamiento'].batch_size}")
    print(f"Numero de epocas: {NUMERO_EPOCAS}")
    print(f"Tasa de aprendizaje: {TASA_APRENDIZAJE}")

    resultado = entrenar_estimador(
        modelo,
        cargadores,
        funcion_perdida,
        optimizador,
        ruta_mejor_modelo=carpeta / RUTA_MEJOR_ESTIMADOR.name,
        dispositivo=dispositivo,
    )

    modelo, _ = cargar_estimador(resultado["ruta_mejor_modelo"], dispositivo)
    prueba = evaluar_estimador(
        modelo,
        divisiones["prueba"]["muestras"],
        divisiones["prueba"]["parametros_reales"],
        dispositivo,
    )
    prueba.pop("predicciones")
    prueba.pop("parametros_reales")
    prueba["perdida"] = recorrer_cargador(
        modelo, cargadores["prueba"], funcion_perdida, dispositivo
    )["perdida"]
    guardar_registro(carpeta, "estimador", dispositivo, huella, resultado, [prueba])
    print("Entrenamiento terminado")
    print(f"MAE de frecuencia en prueba: {prueba['error_frecuencia_hz']:.2f} Hz")
    print(f"Mejor epoca: {resultado['mejor_epoca']}")
    print(f"Menor error de validacion: {resultado['mejor_perdida_validacion']:.5f}")
    print(f"Estimador guardado en: {resultado['ruta_mejor_modelo']}")


if __name__ == "__main__":
    main()
