import sys

ruta_proyecto = (
    r"C:\Users\angel\Documents\proyectos\onnx"
    r"\Clasificación de señales senoidales con CNN y ONNX Runtime"
)

if ruta_proyecto not in sys.path:
    sys.path.insert(0, ruta_proyecto)

from gnu_radio import bloque_inferencia_onnx


class blk(bloque_inferencia_onnx.BloqueInferenciaONNX):
    """Conecta el flujo de GNU Radio con el detector y estimador ONNX."""

    def __init__(self):
        super().__init__()