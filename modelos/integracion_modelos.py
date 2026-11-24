import sys
from numbers import Real
from pathlib import Path

import numpy as np
import torch

# Permite ejecutar este archivo directamente desde PyCharm o la terminal.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelos.clasificador_cnn.evaluacion_modelo import (
    UMBRAL_CLASIFICACION,
    cargar_modelo_entrenado,
)
from modelos.clasificador_cnn.prueba_clasificador_ruido.experimento_ruido import (
    RUTA_MODELO_RUIDO,
)
from modelos.estimador_cnn.entrenar_estimador import RUTA_MEJOR_ESTIMADOR
from modelos.estimador_cnn.evaluacion_modelo import cargar_estimador
from modelos.estimador_cnn.preparar_estimador import decodificar_predicciones
from modelos.generador_senales import (
    AMPLITUD_MAXIMA,
    AMPLITUD_MINIMA,
    FASE_MAXIMA,
    FASE_MINIMA,
    FRECUENCIA_MAXIMA,
    FRECUENCIA_MINIMA,
    NUMERO_MUESTRAS,
    generar_senoidal,
)


def cargar_sistema(
    ruta_detector=RUTA_MODELO_RUIDO,
    ruta_estimador=RUTA_MEJOR_ESTIMADOR,
    dispositivo=None,
):
    """Carga ambas CNN y comprueba que su entrada coincida con el YAML."""

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    ruta_detector = Path(ruta_detector)
    if not ruta_detector.is_file():
        raise FileNotFoundError(
            f"No se encontró el detector en {ruta_detector}. "
            "Para generar el detector entrenado con ruido, ejecuta "
            "python -m modelos.clasificador_cnn.prueba_clasificador_ruido.experimento_ruido."
        )

    detector, _ = cargar_modelo_entrenado(ruta_detector, dispositivo)
    estimador, _ = cargar_estimador(ruta_estimador, dispositivo)
    for nombre, modelo in (("detector", detector), ("estimador", estimador)):
        if modelo.numero_muestras != NUMERO_MUESTRAS:
            raise ValueError(
                f"El {nombre} espera {modelo.numero_muestras} muestras por señal, "
                f"pero configuracion.yml indica {NUMERO_MUESTRAS}. "
                "Usa la configuración de muestreo con la que entrenaste el modelo."
            )
    return detector, estimador


def analizar_senal(
    muestras,
    detector,
    estimador,
    dispositivo=None,
    umbral=UMBRAL_CLASIFICACION,
):
    """Clasifica muestras tomadas a la frecuencia de muestreo del YAML.

    Devuelve frecuencia, amplitud y fase solo si la señal se clasifica como senoidal.
    """

    if dispositivo is None:
        dispositivo = torch.device("cpu")
    if not isinstance(umbral, Real) or isinstance(umbral, bool):
        raise TypeError("El umbral de clasificación debe ser un número.")
    if not 0 < umbral < 1:
        raise ValueError("El umbral debe ser mayor que 0 y menor que 1.")

    muestras = np.asarray(muestras, dtype=np.float32)
    if muestras.ndim != 1:
        raise ValueError("La señal debe ser un vector de muestras.")
    if len(muestras) != NUMERO_MUESTRAS:
        raise ValueError(f"La señal debe contener {NUMERO_MUESTRAS} muestras.")
    if not np.all(np.isfinite(muestras)):
        raise ValueError("La señal contiene valores no finitos.")

    entrada = torch.as_tensor(
        np.ascontiguousarray(muestras)[np.newaxis, np.newaxis, :],
        dtype=torch.float32,
        device=dispositivo,
    )

    detector.to(dispositivo)
    detector.eval()
    with torch.no_grad():
        salida_detector = detector(entrada)
        probabilidad = float(torch.sigmoid(salida_detector).item())

    resultado = {
        "es_senoidal": probabilidad >= umbral,
        "probabilidad_senoidal": probabilidad,
    }
    if not resultado["es_senoidal"]:
        return resultado

    estimador.to(dispositivo)
    estimador.eval()
    with torch.no_grad():
        salida_estimador = estimador(entrada).cpu().numpy()

    frecuencia, amplitud, fase = decodificar_predicciones(salida_estimador)[0]
    resultado.update(
        {
            "frecuencia_hz": float(frecuencia),
            "amplitud": float(amplitud),
            "fase_grados": float(fase),
        }
    )
    return resultado


def main():
    """Demuestra la integración con el punto medio de cada rango del YAML."""

    detector, estimador = cargar_sistema()
    frecuencia_real = (FRECUENCIA_MINIMA + FRECUENCIA_MAXIMA) / 2
    amplitud_real = (AMPLITUD_MINIMA + AMPLITUD_MAXIMA) / 2
    fase_real = (FASE_MINIMA + FASE_MAXIMA) / 2
    _, muestras = generar_senoidal(
        frecuencia=frecuencia_real,
        amplitud=amplitud_real,
        fase_grados=fase_real,
    )
    resultado = analizar_senal(muestras, detector, estimador)

    print("Análisis completo")
    print(f"Probabilidad de ser senoidal: {resultado['probabilidad_senoidal']:.2%}")
    if not resultado["es_senoidal"]:
        print("Resultado: la CNN no la considera senoidal.")
        return

    print("Resultado: senoidal")
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
