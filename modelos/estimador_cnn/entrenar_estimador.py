from pathlib import Path

import torch
from torch import nn

from .crear_dataset import SEMILLA
from .modelo_estimador import EstimadorParametrosCNN
from .preparar_estimacion import preparar_cargadores_estimacion


TASA_APRENDIZAJE = 0.001
NUMERO_EPOCAS = 60

CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
RUTA_MEJOR_ESTIMADOR = CARPETA_MODELOS / "estimador_parametros_cnn.pt"


def crear_componentes_entrenamiento(
    modelo,
    tasa_aprendizaje=TASA_APRENDIZAJE,
):
    """Crea la medida de error y el optimizador del estimador."""

    if tasa_aprendizaje <= 0:
        raise ValueError("La tasa de aprendizaje debe ser mayor que cero.")

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

    perdida_acumulada = 0.0
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
            perdida_acumulada += perdida.item() * cantidad_lote
            cantidad_ejemplos += cantidad_lote

    if cantidad_ejemplos == 0:
        raise ValueError("El cargador esta vacio.")

    return {
        "perdida": perdida_acumulada / cantidad_ejemplos,
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

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoca": epoca,
            "numero_muestras": modelo.numero_muestras,
            "cantidad_salidas": modelo.CANTIDAD_SALIDAS,
            "estado_modelo": modelo.state_dict(),
            "estado_optimizador": optimizador.state_dict(),
            "perdida_validacion": perdida_validacion,
        },
        ruta_salida,
    )


def entrenar_estimador(
    modelo,
    cargadores,
    funcion_perdida,
    optimizador,
    numero_epocas=NUMERO_EPOCAS,
    ruta_mejor_modelo=RUTA_MEJOR_ESTIMADOR,
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
    if dispositivo is None:
        dispositivo = torch.device("cpu")

    mejor_perdida = float("inf")
    mejor_epoca = 0
    historial = []

    for epoca in range(1, numero_epocas + 1):
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
            }
        )

        if mostrar_progreso:
            marca = "  <- mejor estimador guardado" if es_mejor else ""
            print(
                f"Epoca {epoca:02d}/{numero_epocas} | "
                f"error entrenamiento: {resultado_entrenamiento['perdida']:.5f} | "
                f"error validacion: {resultado_validacion['perdida']:.5f}"
                f"{marca}"
            )

    return {
        "historial": historial,
        "mejor_epoca": mejor_epoca,
        "mejor_perdida_validacion": mejor_perdida,
        "ruta_mejor_modelo": Path(ruta_mejor_modelo),
    }


def main():
    """Prepara, entrena y guarda el estimador."""

    torch.manual_seed(SEMILLA)
    cargadores = preparar_cargadores_estimacion(semilla=SEMILLA)
    modelo = EstimadorParametrosCNN()
    funcion_perdida, optimizador = crear_componentes_entrenamiento(modelo)

    print("Inicio del entrenamiento del estimador")
    print(f"Senoidales de entrenamiento: {len(cargadores['entrenamiento'].dataset)}")
    print(f"Senoidales de validacion: {len(cargadores['validacion'].dataset)}")
    print(f"Numero de epocas: {NUMERO_EPOCAS}")

    resultado = entrenar_estimador(
        modelo,
        cargadores,
        funcion_perdida,
        optimizador,
    )

    print("Entrenamiento terminado")
    print(f"Mejor epoca: {resultado['mejor_epoca']}")
    print(f"Menor error de validacion: {resultado['mejor_perdida_validacion']:.5f}")
    print(f"Estimador guardado en: {resultado['ruta_mejor_modelo']}")


if __name__ == "__main__":
    main()
