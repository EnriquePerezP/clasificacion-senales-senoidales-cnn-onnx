from pathlib import Path

import numpy as np
import yaml


def cargar_configuracion():
    ruta = Path(__file__).resolve().parent.parent / "configuracion.yml"

    with ruta.open("r", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


CONFIGURACION = cargar_configuracion()
PARAMETROS_RUIDO = CONFIGURACION["Ruido"]
INTENSIDADES_RUIDO = PARAMETROS_RUIDO["Intensidad de ruido"]
RUIDO_ACTIVADO = PARAMETROS_RUIDO["Activacion de ruido"]

# Las intensidades se definen únicamente en configuracion.yml.
RUIDO_MUY_BAJO = INTENSIDADES_RUIDO["Ruido muy bajo"]
RUIDO_BAJO = INTENSIDADES_RUIDO["Ruido bajo"]
RUIDO_MEDIO = INTENSIDADES_RUIDO["Ruido medio"]
RUIDO_ALTO = INTENSIDADES_RUIDO["Ruido alto"]



def agregar_ruido(muestras, intensidad=RUIDO_BAJO, activado=RUIDO_ACTIVADO, semilla=None):
    """
    Devuelve una copia de la señal, con ruido opcional.

    Parámetros:
        muestras: valores de la señal limpia.
        intensidad: proporción de ruido respecto a la amplitud de la señal.
                    Por ejemplo, 0.10 equivale aproximadamente al 10 %.
        activado: si es False, devuelve la señal limpia sin cambios.
        semilla: número opcional para repetir exactamente el mismo ruido.

    La señal recibida nunca se modifica.
    """

    señal_limpia = np.asarray(muestras, dtype=float)

    if señal_limpia.ndim != 1:
        raise ValueError("La señal debe contener una sola dimensión de muestras.")

    if señal_limpia.size == 0:
        raise ValueError("La señal no puede estar vacía.")

    if not np.all(np.isfinite(señal_limpia)):
        raise ValueError("La señal contiene valores no válidos.")

    if intensidad < 0:
        raise ValueError("La intensidad del ruido no puede ser negativa.")

    # La copia protege la señal original, incluso cuando el ruido está apagado.
    señal_resultante = señal_limpia.copy()

    if not activado or intensidad == 0:
        return señal_resultante

    amplitud_referencia = np.max(np.abs(señal_limpia))

    # Una señal completamente plana no tiene amplitud con la cual calcular
    # una proporción de ruido, así que se devuelve sin cambios.
    if amplitud_referencia == 0:
        return señal_resultante

    generador_aleatorio = np.random.default_rng(semilla)

    # El ruido gaussiano produce valores aleatorios alrededor de cero.
    ruido = generador_aleatorio.normal(
        loc=0.0,
        scale=intensidad * amplitud_referencia,
        size=señal_limpia.shape,
    )

    return señal_resultante + ruido
