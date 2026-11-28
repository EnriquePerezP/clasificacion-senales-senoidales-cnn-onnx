"""Compara la latencia en ONNX Runtime con las mismas entradas y pesos."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
import torch
import onnxruntime as ort


CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"


def _crear_sesion(modelo, proveedor, hilos_cpu, indice_cuda, prefijo_perfil=None):
    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = hilos_cpu
    if prefijo_perfil is not None:
        opciones.enable_profiling = True
        opciones.profile_file_prefix = str(prefijo_perfil)
    proveedores = [CPU]
    if proveedor == CUDA:
        proveedores.insert(0, (CUDA, {"device_id": indice_cuda, "use_tf32": 0}))
    sesion = ort.InferenceSession(modelo, sess_options=opciones, providers=proveedores)
    sesion.disable_fallback()
    if sesion.get_providers()[0] != proveedor:
        raise RuntimeError(f"ONNX Runtime no pudo inicializar {proveedor}.")
    return sesion


def _verificar_ejecucion_cuda(modelo, entrada, hilos_cpu, indice_cuda):
    with TemporaryDirectory() as temporal:
        sesion = _crear_sesion(modelo, CUDA, hilos_cpu, indice_cuda, Path(temporal) / "ort")
        sesion.run(None, entrada)
        perfil = json.loads(Path(sesion.end_profiling()).read_text(encoding="utf-8"))
    proveedores = Counter(
        evento["args"]["provider"]
        for evento in perfil
        if evento.get("cat") == "Node" and "provider" in evento.get("args", {})
    )
    if not proveedores[CUDA]:
        raise RuntimeError("El perfil ONNX no registró ninguna operación en CUDA.")
    return proveedores


def comparar_inferencia(ruta, lotes, dispositivo, hilos_cpu, repeticiones, calentamiento):
    """Mide sesiones sin perfilado, con entradas y salidas alojadas en CPU."""

    if CUDA not in ort.get_available_providers():
        raise RuntimeError("Instala onnxruntime-gpu compatible con la versión CUDA de PyTorch.")
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    # Las sesiones se construyen con los mismos bytes leídos una sola vez.
    modelo = Path(ruta).read_bytes()
    huella = hashlib.sha256(modelo).hexdigest()
    indice_cuda = dispositivo.index
    cpu = _crear_sesion(modelo, CPU, hilos_cpu, indice_cuda)
    cuda = _crear_sesion(modelo, CUDA, hilos_cpu, indice_cuda)

    resumen = []
    mediciones = []
    for lote in lotes:
        entradas = {"muestras": np.ascontiguousarray(lote, dtype=np.float32)}
        salida_cpu = cpu.run(None, entradas)[0]
        salida_cuda = cuda.run(None, entradas)[0]
        np.testing.assert_allclose(salida_cpu, salida_cuda, rtol=1e-5, atol=1e-5)
        nodos = _verificar_ejecucion_cuda(modelo, entradas, hilos_cpu, indice_cuda)
        diferencia = float(np.max(np.abs(salida_cpu - salida_cuda)))

        for sesion, proveedor in ((cpu, CPU), (cuda, CUDA)):
            for _ in range(calentamiento):
                sesion.run(None, entradas)
            latencias = []
            for _ in range(repeticiones):
                inicio = perf_counter_ns()
                sesion.run(None, entradas)
                latencias.append((perf_counter_ns() - inicio) / 1e6)
            mediana = float(np.median(latencias))
            datos = {
                "sha256_onnx": huella,
                "proveedor": proveedor,
                "lote": len(lote),
            }
            resumen.append({
                **datos,
                "calentamiento": calentamiento,
                "repeticiones": repeticiones,
                "hilos_cpu_ort": hilos_cpu,
                "version_onnxruntime": ort.__version__,
                "latencia_media_ms": float(np.mean(latencias)),
                "latencia_mediana_ms": mediana,
                "latencia_p95_ms": float(np.percentile(latencias, 95)),
                "senales_por_segundo_mediana": len(lote) * 1000 / mediana,
                "diferencia_maxima_cpu_cuda": diferencia,
                "nodos_cuda_sesion_cuda": nodos[CUDA],
                "nodos_cpu_sesion_cuda": nodos[CPU],
            })
            mediciones.extend(
                {**datos, "repeticion": indice, "latencia_ms": latencia}
                for indice, latencia in enumerate(latencias, start=1)
            )
    return resumen, mediciones
