from pathlib import Path
import yaml
import numpy as np

def cargar_configuracion():
    ruta = Path(__file__).with_name("configuracion.yml")

    with ruta.open("r", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


CONFIGURACION = cargar_configuracion()
PARAMETROS_SENAL = CONFIGURACION["Parametros de la senal"]
PARAMETROS_MUESTREO = CONFIGURACION["Parametros de muestreo"]

RANGO_AMPLITUDES = PARAMETROS_SENAL["Rango de amplitudes"]
RANGO_FASES = PARAMETROS_SENAL["Rango de fases"]
RANGO_FRECUENCIAS = PARAMETROS_SENAL["Rango de frecuencias"]

AMPLITUD_MAXIMA = RANGO_AMPLITUDES["Amplitud maxima"]
AMPLITUD_MINIMA = RANGO_AMPLITUDES["Amplitud minima"]
FASE_MAXIMA = RANGO_FASES["Fase maxima"]
FASE_MINIMA = RANGO_FASES["Fase minima"]

FRECUENCIA_MINIMA = RANGO_FRECUENCIAS["Frecuencia minima"]
FRECUENCIA_MAXIMA = RANGO_FRECUENCIAS["Frecuencia maxima"]
FRECUENCIA_MUESTREO = PARAMETROS_MUESTREO["Frecuencia de muestreo"]
NUMERO_MUESTRAS = PARAMETROS_MUESTREO["Numero muestras"]


# ============================================================
# GENERADOR DE ONDA SENOIDAL
# ============================================================

def generar_senoidal(frecuencia, amplitud, fase_grados):
    """
    Genera una onda senoidal limpia.

    Parámetros:
        frecuencia: cantidad de ciclos por segundo, en Hz.
        amplitud: altura máxima de la señal.
        fase_grados: punto donde comienza la onda, en grados.

    Devuelve:
        tiempo: instante correspondiente a cada muestra.
        señal: valores de la onda senoidal.
    """

    # Comprobamos que los parámetros estén dentro de los límites.
    if not FRECUENCIA_MINIMA <= frecuencia <= FRECUENCIA_MAXIMA:
        raise ValueError(
            f"La frecuencia debe estar entre "
            f"{FRECUENCIA_MINIMA} y {FRECUENCIA_MAXIMA} Hz."
        )

    if not AMPLITUD_MINIMA <= amplitud <= AMPLITUD_MAXIMA:
        raise ValueError(
            f"La amplitud debe estar entre "
            f"{AMPLITUD_MINIMA} y {AMPLITUD_MAXIMA}."
        )

    if not FASE_MINIMA <= fase_grados <= FASE_MAXIMA:
        raise ValueError(
            f"La fase debe estar entre "
            f"{FASE_MINIMA} y {FASE_MAXIMA} grados."
        )

    # Creamos los 1,024 momentos de tiempo.
    tiempo = np.arange(NUMERO_MUESTRAS) / FRECUENCIA_MUESTREO

    # NumPy utiliza radianes, por eso convertimos los grados.
    fase_radianes = np.deg2rad(fase_grados)

    # Aplicamos la fórmula de la onda senoidal.
    señal = amplitud * np.sin(
        2 * np.pi * frecuencia * tiempo + fase_radianes
    )

    return tiempo, señal


def generar_cuadrada(frecuencia, amplitud, fase_grados):
    """Genera una onda cuadrada limpia."""

    tiempo, senoidal = generar_senoidal(
        frecuencia=frecuencia,
        amplitud=amplitud,
        fase_grados=fase_grados,
    )

    señal = np.where(senoidal >= 0, amplitud, -amplitud)

    return tiempo, señal


def generar_triangular(frecuencia, amplitud, fase_grados):
    """Genera una onda triangular limpia."""

    tiempo, senoidal = generar_senoidal(
        frecuencia=frecuencia,
        amplitud=amplitud,
        fase_grados=fase_grados,
    )

    # Convertimos la senoidal en una señal triangular y conservamos
    # la amplitud solicitada.
    senoidal_normalizada = senoidal / amplitud
    senoidal_normalizada = np.clip(senoidal_normalizada, -1, 1)

    señal = amplitud * (2 / np.pi) * np.arcsin(senoidal_normalizada)

    return tiempo, señal

