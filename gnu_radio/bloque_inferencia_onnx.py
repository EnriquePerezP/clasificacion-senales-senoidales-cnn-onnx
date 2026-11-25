import sys
from pathlib import Path

import numpy as np
from gnuradio import gr

# Importa el módulo del proyecto desde su carpeta onnx.
CARPETA_ONNX = Path(__file__).resolve().parents[1] / "onnx"
if str(CARPETA_ONNX) not in sys.path:
    sys.path.insert(0, str(CARPETA_ONNX))

from inferencia_onnx import NUMERO_MUESTRAS, analizar_senal_onnx, cargar_sistema_onnx


class BloqueInferenciaONNX(gr.sync_block):
    """Recibe vectores float32 con el muestreo de configuracion.yml."""

    def __init__(self, detector=None, estimador=None):
        gr.sync_block.__init__(
            self,
            name="Detector y estimador ONNX",
            in_sig=[(np.float32, NUMERO_MUESTRAS)],
            out_sig=[
                np.float32,  # Probabilidad de que sea senoidal.
                np.float32,  # Decisión: 1 senoidal, 0 no senoidal.
                np.float32,  # Frecuencia estimada en Hz.
                np.float32,  # Amplitud estimada.
                np.float32,  # Fase estimada en grados.
            ],
        )

        if detector is None or estimador is None:
            detector, estimador = cargar_sistema_onnx()
        self.detector = detector
        self.estimador = estimador

    def work(self, input_items, output_items):
        """Analiza cada vector disponible y escribe cinco valores de salida."""

        vectores = input_items[0]
        for indice, muestras in enumerate(vectores):
            resultado = analizar_senal_onnx(
                muestras,
                self.detector,
                self.estimador,
            )

            output_items[0][indice] = resultado["probabilidad_senoidal"]
            output_items[1][indice] = float(resultado["es_senoidal"])

            # Cero indica que estos parámetros no aplican cuando la señal
            # fue clasificada como no senoidal.
            output_items[2][indice] = resultado.get("frecuencia_hz", 0.0)
            output_items[3][indice] = resultado.get("amplitud", 0.0)
            output_items[4][indice] = resultado.get("fase_grados", 0.0)

        return len(vectores)
