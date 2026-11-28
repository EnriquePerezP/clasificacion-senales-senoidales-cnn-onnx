import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

# Permite ejecutar este archivo directamente desde PyCharm o la terminal.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelos.clasificador_cnn.evaluacion_modelo import cargar_modelo_entrenado
from modelos.clasificador_cnn.prueba_clasificador_ruido.experimento_ruido import (
    RUTA_MODELO_RUIDO,
)
from modelos.estimador_cnn.entrenar_estimador import RUTA_MEJOR_ESTIMADOR
from modelos.estimador_cnn.evaluacion_modelo import cargar_estimador
from modelos.generador_senales import (
    AMPLITUD_MAXIMA,
    AMPLITUD_MINIMA,
    FASE_MAXIMA,
    FASE_MINIMA,
    FRECUENCIA_MAXIMA,
    FRECUENCIA_MINIMA,
    FRECUENCIA_MUESTREO,
    NUMERO_MUESTRAS,
)
from modelos.integracion_modelos import cargar_sistema


CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
RUTA_DETECTOR_ONNX = CARPETA_MODELOS / "detector_senoidal_cnn_ruido.onnx"
RUTA_ESTIMADOR_ONNX = CARPETA_MODELOS / "estimador_parametros_cnn.onnx"
VERSION_OPERADORES_ONNX = 18


class DetectorConProbabilidad(nn.Module):
    """Añade la conversión de la salida del detector a una probabilidad."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, muestras):
        return torch.sigmoid(self.detector(muestras))


def _preparar_ruta(ruta_salida):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    return ruta_salida


def _permitir_mensajes_unicode():
    """Evita fallos si PyTorch muestra símbolos no admitidos por Windows."""

    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")


def _exportar_onnx(modelo, ruta_salida, numero_muestras, nombre_salida):
    _permitir_mensajes_unicode()
    ruta_salida = _preparar_ruta(ruta_salida)
    modelo = modelo.cpu().eval()
    # Una traza con más de una señal permite conservar dinámico el lote.
    entrada_ejemplo = torch.zeros(
        2,
        1,
        numero_muestras,
        dtype=torch.float32,
    )

    torch.onnx.export(
        modelo,
        (entrada_ejemplo,),
        ruta_salida,
        input_names=["muestras"],
        output_names=[nombre_salida],
        dynamic_shapes=({0: torch.export.Dim("cantidad_senales")},),
        opset_version=VERSION_OPERADORES_ONNX,
        dynamo=True,
        external_data=False,
    )
    return ruta_salida


def exportar_detector_onnx(
    detector,
    ruta_salida=RUTA_DETECTOR_ONNX,
    numero_muestras=NUMERO_MUESTRAS,
):
    """Exporta el detector con una probabilidad entre 0 y 1 por señal."""

    return _exportar_onnx(
        DetectorConProbabilidad(detector),
        ruta_salida,
        numero_muestras,
        "probabilidad_senoidal",
    )


def exportar_estimador_onnx(
    estimador,
    ruta_salida=RUTA_ESTIMADOR_ONNX,
    numero_muestras=NUMERO_MUESTRAS,
):
    """Exporta las cuatro salidas internas del estimador de parámetros."""

    return _exportar_onnx(
        estimador,
        ruta_salida,
        numero_muestras,
        "parametros_internos",
    )


def _ejemplos_comprobacion(numero_muestras):
    """Crea señales de prueba con el muestreo y los rangos del YAML."""

    tiempo = np.arange(numero_muestras) / FRECUENCIA_MUESTREO
    fase = np.deg2rad((FASE_MINIMA + FASE_MAXIMA) / 2)
    seno_bajo = np.sin(2 * np.pi * FRECUENCIA_MINIMA * tiempo + fase)
    seno_alto = np.sin(2 * np.pi * FRECUENCIA_MAXIMA * tiempo + fase)
    muestras = np.stack(
        (
            AMPLITUD_MAXIMA * seno_bajo,
            AMPLITUD_MINIMA * seno_alto,
            np.where(seno_bajo >= 0, AMPLITUD_MAXIMA, -AMPLITUD_MAXIMA),
        )
    )
    return muestras[:, np.newaxis, :].astype(np.float32)


def comparar_pytorch_onnx(
    modelo_pytorch,
    ruta_onnx,
    incluir_sigmoid=False,
    numero_muestras=NUMERO_MUESTRAS,
    tolerancia=1e-5,
):
    """Comprueba que ONNX y PyTorch produzcan prácticamente lo mismo."""

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "Faltan ONNX u ONNX Runtime. Instala requirements.txt."
        ) from error

    ruta_onnx = Path(ruta_onnx)
    onnx.checker.check_model(onnx.load(ruta_onnx))

    sesion = ort.InferenceSession(
        str(ruta_onnx),
        providers=["CPUExecutionProvider"],
    )
    muestras = _ejemplos_comprobacion(numero_muestras)
    modelo_pytorch = modelo_pytorch.cpu().eval()
    diferencia_maxima = 0.0
    for lote in (muestras[:1], muestras):
        with torch.inference_mode():
            salida_pytorch = modelo_pytorch(torch.from_numpy(lote))
            if incluir_sigmoid:
                salida_pytorch = torch.sigmoid(salida_pytorch)
            salida_pytorch = salida_pytorch.numpy()

        salida_onnx = sesion.run(None, {"muestras": lote})[0]
        np.testing.assert_allclose(
            salida_onnx,
            salida_pytorch,
            rtol=tolerancia,
            atol=tolerancia,
            equal_nan=False,
        )
        diferencia_maxima = max(
            diferencia_maxima,
            float(np.max(np.abs(salida_onnx - salida_pytorch))),
        )
    return diferencia_maxima


def exportar_modelos_entrenados(
    ruta_detector_pt=RUTA_MODELO_RUIDO,
    ruta_estimador_pt=RUTA_MEJOR_ESTIMADOR,
    ruta_detector_onnx=RUTA_DETECTOR_ONNX,
    ruta_estimador_onnx=RUTA_ESTIMADOR_ONNX,
):
    """Carga, exporta y valida los dos modelos entrenados del proyecto."""

    detector, estimador = cargar_sistema(ruta_detector_pt, ruta_estimador_pt)

    ruta_detector_onnx = _preparar_ruta(ruta_detector_onnx)
    ruta_estimador_onnx = _preparar_ruta(ruta_estimador_onnx)
    if ruta_detector_onnx.resolve() == ruta_estimador_onnx.resolve():
        raise ValueError("El detector y el estimador necesitan rutas ONNX distintas.")

    # Valida ambos archivos antes de reemplazar las exportaciones finales.
    with (
        TemporaryDirectory(dir=ruta_detector_onnx.parent) as carpeta_detector,
        TemporaryDirectory(dir=ruta_estimador_onnx.parent) as carpeta_estimador,
    ):
        detector_temporal = exportar_detector_onnx(
            detector,
            Path(carpeta_detector) / ruta_detector_onnx.name,
        )
        estimador_temporal = exportar_estimador_onnx(
            estimador,
            Path(carpeta_estimador) / ruta_estimador_onnx.name,
        )
        diferencia_detector = comparar_pytorch_onnx(
            detector,
            detector_temporal,
            incluir_sigmoid=True,
        )
        diferencia_estimador = comparar_pytorch_onnx(estimador, estimador_temporal)
        detector_temporal.replace(ruta_detector_onnx)
        estimador_temporal.replace(ruta_estimador_onnx)

    return {
        "ruta_detector": ruta_detector_onnx,
        "ruta_estimador": ruta_estimador_onnx,
        "diferencia_detector": diferencia_detector,
        "diferencia_estimador": diferencia_estimador,
        "numero_muestras": NUMERO_MUESTRAS,
    }


def exportar_checkpoint_onnx(tipo, ruta_checkpoint, ruta_onnx):
    """Exporta un checkpoint de cualquier dispositivo y valida lotes de 1 y 3."""

    if tipo == "detector":
        modelo, _ = cargar_modelo_entrenado(ruta_checkpoint)
        exportar = exportar_detector_onnx
    elif tipo == "estimador":
        modelo, _ = cargar_estimador(ruta_checkpoint)
        exportar = exportar_estimador_onnx
    else:
        raise ValueError(f"Tipo de modelo desconocido: {tipo}.")

    ruta_onnx = _preparar_ruta(ruta_onnx)
    if ruta_onnx.suffix.lower() != ".onnx":
        raise ValueError("La ruta de salida debe terminar en .onnx.")
    if ruta_onnx.resolve() in (RUTA_DETECTOR_ONNX.resolve(), RUTA_ESTIMADOR_ONNX.resolve()):
        raise ValueError("Indica una ruta distinta de los archivos ONNX originales.")
    with TemporaryDirectory(dir=ruta_onnx.parent) as carpeta:
        temporal = Path(carpeta) / ruta_onnx.name
        exportar(modelo, temporal, numero_muestras=modelo.numero_muestras)
        diferencia = comparar_pytorch_onnx(
            modelo, temporal, tipo == "detector", modelo.numero_muestras
        )
        temporal.replace(ruta_onnx)
    return diferencia


def main(argv=None):
    """Exporta ambos modelos y muestra el resultado de la comprobación."""

    parser = argparse.ArgumentParser(description="Exporta y valida modelos en ONNX.")
    parser.add_argument("--tipo", choices=("detector", "estimador"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args(argv)
    if any((args.tipo, args.checkpoint, args.salida)):
        if not all((args.tipo, args.checkpoint, args.salida)):
            parser.error("Proporciona --tipo, --checkpoint y --salida juntos.")
        diferencia = exportar_checkpoint_onnx(args.tipo, args.checkpoint, args.salida)
        print(f"ONNX validado en {args.salida} | diferencia máxima: {diferencia:.3g}")
        return

    resultado = exportar_modelos_entrenados()

    print("Exportación a ONNX terminada")
    print(f"Entrada común: (cantidad, 1, {resultado['numero_muestras']})")
    print(f"Detector: {resultado['ruta_detector']}")
    print(f"Estimador: {resultado['ruta_estimador']}")
    print(
        "Diferencia máxima detector PyTorch/ONNX: "
        f"{resultado['diferencia_detector']:.10f}"
    )
    print(
        "Diferencia máxima estimador PyTorch/ONNX: "
        f"{resultado['diferencia_estimador']:.10f}"
    )
    print("Los dos modelos ONNX fueron comprobados correctamente.")


if __name__ == "__main__":
    main()
