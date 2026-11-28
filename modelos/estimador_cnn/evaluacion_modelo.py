import sys
from pathlib import Path

import numpy as np
import torch

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelos.clasificador_cnn.division_datos import TAMANO_LOTE
from modelos.estimador_cnn.entrenar_estimador import RUTA_MEJOR_ESTIMADOR
from modelos.estimador_cnn.modelo_estimador import EstimadorParametrosCNN
from modelos.estimador_cnn.preparar_estimador import (
    decodificar_predicciones,
    preparar_datos_estimacion,
)
from modelos.generador_senales import NUMERO_MUESTRAS


def cargar_estimador(ruta_modelo=RUTA_MEJOR_ESTIMADOR, dispositivo=None):
    """Reconstruye la CNN y carga los valores aprendidos."""

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    ruta_modelo = Path(ruta_modelo)
    if not ruta_modelo.is_file():
        raise FileNotFoundError(
            f"No se encontro el estimador: {ruta_modelo}. "
            "Ejecuta primero entrenar_estimador."
        )

    contenido = torch.load(
        ruta_modelo,
        map_location="cpu",
        weights_only=True,
    )
    modelo = EstimadorParametrosCNN(
        numero_muestras=int(contenido["numero_muestras"])
    )
    modelo.load_state_dict(contenido["estado_modelo"])
    modelo.to(dispositivo)
    modelo.eval()
    return modelo, contenido


def diferencia_fase_grados(fase_calculada, fase_real):
    """Calcula la distancia mas corta entre dos angulos."""

    return np.abs((fase_calculada - fase_real + 180.0) % 360.0 - 180.0)


def calcular_metricas(predicciones, parametros_reales):
    """Calcula el error promedio de frecuencia, amplitud y fase."""

    predicciones = np.asarray(predicciones, dtype=np.float32)
    parametros_reales = np.asarray(parametros_reales, dtype=np.float32)
    if predicciones.shape != parametros_reales.shape:
        raise ValueError("Predicciones y parametros reales deben tener la misma forma.")
    if predicciones.ndim != 2 or predicciones.shape[1] != 3:
        raise ValueError("Se esperan frecuencia, amplitud y fase por senal.")
    if len(predicciones) == 0:
        raise ValueError("Se necesita al menos una señal para calcular las métricas.")

    return {
        "error_frecuencia_hz": float(
            np.mean(np.abs(predicciones[:, 0] - parametros_reales[:, 0]))
        ),
        "error_amplitud": float(
            np.mean(np.abs(predicciones[:, 1] - parametros_reales[:, 1]))
        ),
        "error_fase_grados": float(
            np.mean(
                diferencia_fase_grados(
                    predicciones[:, 2],
                    parametros_reales[:, 2],
                )
            )
        ),
        "cantidad_ejemplos": len(predicciones),
    }


def evaluar_estimador(
    modelo,
    muestras,
    parametros_reales,
    dispositivo=None,
    *,
    tamano_lote=TAMANO_LOTE,
):
    """Realiza predicciones sobre senoidales reservadas para prueba."""

    if dispositivo is None:
        dispositivo = torch.device("cpu")
    if not isinstance(tamano_lote, int) or isinstance(tamano_lote, bool):
        raise TypeError("El tamaño del lote debe ser un número entero.")
    if tamano_lote <= 0:
        raise ValueError("El tamaño del lote debe ser mayor que cero.")

    tensor_muestras = torch.as_tensor(muestras, dtype=torch.float32)
    modelo.to(dispositivo)
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for lote in tensor_muestras.split(tamano_lote):
            salidas.append(modelo(lote.to(dispositivo)).cpu())

    predicciones = decodificar_predicciones(torch.cat(salidas).numpy())
    metricas = calcular_metricas(predicciones, parametros_reales)
    return {
        "predicciones": predicciones,
        "parametros_reales": np.asarray(parametros_reales, dtype=np.float32),
        **metricas,
    }


def mostrar_resultado(resultado, informacion_modelo, ruta_modelo):
    """Muestra las metricas y cinco comparaciones comprensibles."""

    print("Evaluacion del estimador terminada")
    print(f"Modelo: {ruta_modelo}")
    print(f"Mejor epoca guardada: {informacion_modelo['epoca']}")
    print(f"Senoidales de prueba: {resultado['cantidad_ejemplos']}")
    print(
        f"Error promedio de frecuencia: "
        f"{resultado['error_frecuencia_hz']:.2f} Hz"
    )
    print(f"Error promedio de amplitud: {resultado['error_amplitud']:.4f}")
    print(
        f"Error promedio de fase: "
        f"{resultado['error_fase_grados']:.2f} grados"
    )

    print("\nPrimeras comparaciones:")
    cantidad = min(5, resultado["cantidad_ejemplos"])
    for indice in range(cantidad):
        real = resultado["parametros_reales"][indice]
        calculado = resultado["predicciones"][indice]
        print(
            f"  Senal {indice + 1}: "
            f"frecuencia {calculado[0]:.2f} Hz (real {real[0]:.2f}), "
            f"amplitud {calculado[1]:.3f} (real {real[1]:.3f}), "
            f"fase {calculado[2]:.2f} grados (real {real[2]:.2f})"
        )


def evaluar_estimador_guardado():
    """Evalúa el modelo guardado y devuelve las muestras usadas en la prueba."""

    dispositivo = torch.device("cpu")
    modelo, informacion = cargar_estimador(dispositivo=dispositivo)
    if modelo.numero_muestras != NUMERO_MUESTRAS:
        raise ValueError(
            f"El estimador espera {modelo.numero_muestras} muestras por señal, "
            f"pero configuracion.yml indica {NUMERO_MUESTRAS}. "
            "Usa la configuración con la que entrenaste el estimador."
        )

    prueba = preparar_datos_estimacion()["prueba"]
    resultado = evaluar_estimador(
        modelo,
        prueba["muestras"],
        prueba["parametros_reales"],
        dispositivo,
    )
    return resultado, informacion, prueba["muestras"]


def main():
    resultado, informacion, _ = evaluar_estimador_guardado()
    mostrar_resultado(resultado, informacion, RUTA_MEJOR_ESTIMADOR)


if __name__ == "__main__":
    main()
