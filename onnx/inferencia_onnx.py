import sys
from numbers import Real
from pathlib import Path

import numpy as np

# Permite ejecutar este archivo directamente desde PyCharm o la terminal.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelos.generador_senales import (
    AMPLITUD_MAXIMA,
    AMPLITUD_MINIMA,
    CONFIGURACION,
    FASE_MAXIMA,
    FASE_MINIMA,
    FRECUENCIA_MAXIMA,
    FRECUENCIA_MINIMA,
    NUMERO_MUESTRAS,
    generar_senoidal,
)


CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
RUTA_DETECTOR_ONNX = CARPETA_MODELOS / "detector_senoidal_cnn_ruido.onnx"
RUTA_ESTIMADOR_ONNX = CARPETA_MODELOS / "estimador_parametros_cnn.onnx"
UMBRAL_CLASIFICACION = CONFIGURACION["Evaluacion"]["Umbral de clasificacion"]
NOMBRE_ENTRADA = "muestras"
NOMBRE_SALIDA_DETECTOR = "probabilidad_senoidal"
NOMBRE_SALIDA_ESTIMADOR = "parametros_internos"


def cargar_sistema_onnx(
    ruta_detector=RUTA_DETECTOR_ONNX,
    ruta_estimador=RUTA_ESTIMADOR_ONNX,
):
    """Carga las dos sesiones ONNX para reutilizarlas con varias señales."""

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "ONNX Runtime no está instalado. Instala las dependencias de requirements.txt."
        ) from error

    ruta_detector = Path(ruta_detector)
    ruta_estimador = Path(ruta_estimador)

    if not ruta_detector.is_file():
        raise FileNotFoundError(
            f"No se encontró el detector ONNX: {ruta_detector}. "
            "Ejecuta primero onnx/exportacion_onnx.py."
        )
    if not ruta_estimador.is_file():
        raise FileNotFoundError(
            f"No se encontró el estimador ONNX: {ruta_estimador}. "
            "Ejecuta primero onnx/exportacion_onnx.py."
        )

    proveedores = ["CPUExecutionProvider"]
    detector = ort.InferenceSession(
        str(ruta_detector),
        providers=proveedores,
    )
    estimador = ort.InferenceSession(
        str(ruta_estimador),
        providers=proveedores,
    )
    for nombre, sesion in (("detector", detector), ("estimador", estimador)):
        entradas = sesion.get_inputs()
        if len(entradas) != 1 or entradas[0].name != NOMBRE_ENTRADA:
            raise ValueError(f"El {nombre} ONNX debe tener una entrada llamada muestras.")
        entrada = entradas[0]
        if entrada.type != "tensor(float)" or entrada.shape[1:] != [1, NUMERO_MUESTRAS]:
            raise ValueError(
                f"El {nombre} ONNX tiene una entrada {entrada.shape} de tipo {entrada.type}. "
                f"Se espera float32 con un canal y {NUMERO_MUESTRAS} muestras por señal "
                "según configuracion.yml."
            )
    return detector, estimador


def preparar_entrada_onnx(muestras):
    """Convierte una señal al formato (1, 1, numero_muestras) en float32."""

    muestras = np.asarray(muestras, dtype=np.float32)
    if muestras.ndim != 1:
        raise ValueError("La señal debe ser una lista de muestras.")
    if len(muestras) != NUMERO_MUESTRAS:
        raise ValueError(
            f"La señal debe contener {NUMERO_MUESTRAS} muestras."
        )
    if not np.all(np.isfinite(muestras)):
        raise ValueError("La señal contiene valores no válidos.")

    return np.ascontiguousarray(muestras[np.newaxis, np.newaxis, :])


def decodificar_salida_estimador(salida_estimador):
    """Convierte las cuatro salidas ONNX a Hz, amplitud y grados."""

    salida_estimador = np.asarray(salida_estimador, dtype=np.float32)
    if salida_estimador.shape != (1, 4):
        raise ValueError(
            "El estimador ONNX debe devolver cuatro valores por señal."
        )
    if not np.all(np.isfinite(salida_estimador)):
        raise ValueError("El estimador ONNX devolvió valores no válidos.")

    frecuencia_normalizada = np.clip(salida_estimador[0, 0], 0.0, 1.0)
    amplitud_normalizada = np.clip(salida_estimador[0, 1], 0.0, 1.0)
    frecuencia = FRECUENCIA_MINIMA + frecuencia_normalizada * (
        FRECUENCIA_MAXIMA - FRECUENCIA_MINIMA
    )
    amplitud = AMPLITUD_MINIMA + amplitud_normalizada * (
        AMPLITUD_MAXIMA - AMPLITUD_MINIMA
    )
    fase = np.mod(
        np.rad2deg(np.arctan2(salida_estimador[0, 2], salida_estimador[0, 3])),
        360.0,
    )
    return float(frecuencia), float(amplitud), float(fase)


def analizar_senal_onnx(
    muestras,
    detector,
    estimador,
    umbral=UMBRAL_CLASIFICACION,
):
    """Analiza una señal tomada a la frecuencia de muestreo del YAML."""

    if not isinstance(umbral, Real) or isinstance(umbral, bool):
        raise TypeError("El umbral de clasificación debe ser un número.")
    if not 0 < umbral < 1:
        raise ValueError("El umbral debe ser mayor que 0 y menor que 1.")

    entrada = preparar_entrada_onnx(muestras)
    salida_detector = detector.run(
        [NOMBRE_SALIDA_DETECTOR],
        {NOMBRE_ENTRADA: entrada},
    )[0]
    salida_detector = np.asarray(salida_detector, dtype=np.float32)
    if salida_detector.size != 1:
        raise ValueError("El detector ONNX debe devolver una probabilidad por señal.")

    probabilidad = float(salida_detector.item())
    if not 0 <= probabilidad <= 1:
        raise ValueError("El detector ONNX devolvió una probabilidad no válida.")
    resultado = {
        "es_senoidal": probabilidad >= umbral,
        "probabilidad_senoidal": probabilidad,
    }
    if not resultado["es_senoidal"]:
        return resultado

    salida_estimador = estimador.run(
        [NOMBRE_SALIDA_ESTIMADOR],
        {NOMBRE_ENTRADA: entrada},
    )[0]
    frecuencia, amplitud, fase = decodificar_salida_estimador(salida_estimador)
    resultado.update(
        {
            "frecuencia_hz": frecuencia,
            "amplitud": amplitud,
            "fase_grados": fase,
        }
    )
    return resultado


def main():
    """Comprueba el sistema ONNX con una senoidal de valores conocidos."""

    frecuencia_real = (FRECUENCIA_MINIMA + FRECUENCIA_MAXIMA) / 2
    amplitud_real = (AMPLITUD_MINIMA + AMPLITUD_MAXIMA) / 2
    fase_real = (FASE_MINIMA + FASE_MAXIMA) / 2
    _, muestras = generar_senoidal(
        frecuencia=frecuencia_real,
        amplitud=amplitud_real,
        fase_grados=fase_real,
    )

    detector, estimador = cargar_sistema_onnx()
    resultado = analizar_senal_onnx(muestras, detector, estimador)

    print("Inferencia realizada únicamente con ONNX Runtime")
    print(f"Probabilidad senoidal: {resultado['probabilidad_senoidal']:.2%}")
    if not resultado["es_senoidal"]:
        print("Resultado: la señal no se considera senoidal.")
        return

    print("Resultado: señal senoidal")
    print(
        f"Frecuencia estimada: {resultado['frecuencia_hz']:.2f} Hz "
        f"(real {frecuencia_real:.2f} Hz)"
    )
    print(
        f"Amplitud estimada: {resultado['amplitud']:.3f} "
        f"(real {amplitud_real:.3f})"
    )
    print(
        f"Fase estimada: {resultado['fase_grados']:.2f} grados "
        f"(real {fase_real:.2f} grados)"
    )


if __name__ == "__main__":
    main()
