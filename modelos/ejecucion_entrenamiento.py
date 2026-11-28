"""Selección de dispositivo, reproducibilidad y medición del entrenamiento."""

import csv
import hashlib
import os
import platform
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter, process_time

import numpy as np
import torch

from modelos.dataset import SEMILLA


RUTA_RESULTADOS_GPU = Path(__file__).resolve().parents[1] / "experimentos" / "gpu" / "resultados"
MIB = 1024 ** 2


def agregar_argumentos_entrenamiento(parser):
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--salida", type=Path, default=RUTA_RESULTADOS_GPU)
    parser.add_argument("--hilos-cpu", type=int, help="Hilos de PyTorch para operaciones en CPU")


def resolver_dispositivo(solicitado):
    if solicitado not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Dispositivo no admitido: {solicitado}. Usa auto, cpu o cuda.")
    if solicitado == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if solicitado == "auto":
        return torch.device("cpu")
    raise RuntimeError(
        f"CUDA no está disponible para PyTorch {torch.__version__} "
        f"(compilado con CUDA {torch.version.cuda}). Comprueba la instalación y nvidia-smi."
    )


def configurar_entrenamiento(dispositivo, semilla=SEMILLA, hilos_cpu=None):
    if hilos_cpu is not None:
        if hilos_cpu < 1:
            raise ValueError("La cantidad de hilos debe ser mayor que cero.")
        torch.set_num_threads(hilos_cpu)

    # cuBLAS consulta esta variable al inicializar su espacio de trabajo.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    if dispositivo.type == "cuda":
        torch.cuda.manual_seed_all(semilla)
        torch.set_float32_matmul_precision("highest")
        # El exportador de PyTorch 2.14 consulta el indicador global de cuDNN.
        torch.backends.cudnn.allow_tf32 = False


def crear_modelo_inicial(clase_modelo, dispositivo):
    # Ambos dispositivos parten de los mismos tensores inicializados en CPU.
    modelo = clase_modelo()
    huella = hashlib.sha256()
    for nombre, tensor in modelo.state_dict().items():
        huella.update(nombre.encode("utf-8"))
        huella.update(tensor.numpy().tobytes())
    return modelo.to(dispositivo), huella.hexdigest()


def resolver_ruta_checkpoint(ruta_salida, ruta_original, dispositivo):
    if ruta_salida is None:
        ruta_salida = ruta_original
    ruta_salida = Path(ruta_salida)
    if dispositivo.type == "cuda" and ruta_salida.resolve() == ruta_original.resolve():
        return ruta_salida.with_stem(f"{ruta_salida.stem}_cuda")
    return ruta_salida


def guardar_checkpoint(ruta, contenido):
    """Conserva modelo y optimizador como tensores CPU en el archivo."""

    def en_cpu(valor):
        if torch.is_tensor(valor):
            return valor.detach().cpu()
        if isinstance(valor, dict):
            return type(valor)((clave, en_cpu(item)) for clave, item in valor.items())
        if isinstance(valor, (list, tuple)):
            return type(valor)(en_cpu(item) for item in valor)
        return valor

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    estado = contenido["estado_modelo"]
    estado_cpu = en_cpu(estado)
    estado_cpu._metadata = estado._metadata
    with TemporaryDirectory(dir=ruta.parent) as carpeta:
        temporal = Path(carpeta) / ruta.name
        torch.save(
            {
                **contenido,
                "estado_modelo": estado_cpu,
                "estado_optimizador": en_cpu(contenido["estado_optimizador"]),
            },
            temporal,
        )
        temporal.replace(ruta)


def escribir_csv(ruta, filas):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(campo for fila in filas for campo in fila))
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_registro(carpeta, caso, dispositivo, huella, resultado, evaluacion):
    historial = resultado["historial"]
    cargadores = resultado["cargadores"]
    huella_datos = hashlib.sha256()
    for nombre, cargador in cargadores.items():
        huella_datos.update(nombre.encode("utf-8"))
        for tensor in cargador.dataset.tensors:
            huella_datos.update(str(tuple(tensor.shape)).encode("utf-8"))
            huella_datos.update(tensor.numpy().tobytes())
    escribir_csv(carpeta / f"{caso}_epocas.csv", historial)
    escribir_csv(carpeta / f"{caso}_prueba.csv", evaluacion)
    resumen = {
        "caso": caso,
        "dispositivo": dispositivo.type,
        "nombre_dispositivo": (
            torch.cuda.get_device_name(dispositivo) if dispositivo.type == "cuda" else "CPU"
        ),
        "semilla": SEMILLA,
        "sha256_estado_inicial": huella,
        "sha256_datos_divididos": huella_datos.hexdigest(),
        "sistema": platform.platform(),
        "version_pytorch": str(torch.__version__),
        "version_cuda_pytorch": torch.version.cuda,
        "version_cudnn": torch.backends.cudnn.version() if dispositivo.type == "cuda" else None,
        "hilos_cpu_pytorch": torch.get_num_threads(),
        "epocas": len(historial),
        "lote": cargadores["entrenamiento"].batch_size,
        "tasa_aprendizaje": resultado["tasa_aprendizaje"],
        "senales_entrenamiento": len(cargadores["entrenamiento"].dataset),
        "senales_validacion": len(cargadores["validacion"].dataset),
        "senales_prueba": len(cargadores["prueba"].dataset),
        "mejor_epoca": resultado["mejor_epoca"],
        "perdida_minima_validacion": resultado["mejor_perdida_validacion"],
        "tiempo_total_s": resultado["tiempo_total_s"],
        "tiempo_epocas_s": sum(fila["tiempo_epoca_s"] for fila in historial),
        "tiempo_proceso_cpu_s": sum(fila["tiempo_proceso_cpu_s"] for fila in historial),
        "ram_rss_pico_mib": max(fila["ram_rss_pico_mib"] for fila in historial),
        "checkpoint": resultado["ruta_mejor_modelo"].name,
    }
    if dispositivo.type == "cuda":
        for campo in ("vram_asignada_pico_mib", "vram_reservada_pico_mib"):
            resumen[campo] = max(fila[campo] for fila in historial)
    escribir_csv(carpeta / f"{caso}_resumen.csv", [resumen])
    print(f"Tiempo total: {resumen['tiempo_total_s']:.3f} s | Resultados: {carpeta}")


def sincronizar(dispositivo):
    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)


class MedicionEpoca:
    """Mide entrenamiento y validación antes de guardar el checkpoint."""

    def __init__(self, dispositivo):
        import psutil

        self.dispositivo = dispositivo
        self.proceso = psutil.Process()
        self.detener = Event()
        self.hilo = Thread(target=self._muestrear_ram, daemon=True)

    def _muestrear_ram(self):
        while not self.detener.wait(0.01):
            self.pico_ram = max(self.pico_ram, self.proceso.memory_info().rss)

    def __enter__(self):
        sincronizar(self.dispositivo)
        if self.dispositivo.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.dispositivo)
        self.pico_ram = self.proceso.memory_info().rss
        self.hilo.start()
        self.inicio_cpu = process_time()
        self.inicio = perf_counter()
        return self

    def __exit__(self, tipo_error, error, traceback):
        try:
            sincronizar(self.dispositivo)
            duracion = perf_counter() - self.inicio
            tiempo_cpu = process_time() - self.inicio_cpu
        finally:
            self.detener.set()
            self.hilo.join()
        self.datos = {
            "tiempo_epoca_s": duracion,
            "tiempo_proceso_cpu_s": tiempo_cpu,
            "ram_rss_pico_mib": max(self.pico_ram, self.proceso.memory_info().rss) / MIB,
        }
        if self.dispositivo.type == "cuda":
            self.datos["vram_asignada_pico_mib"] = (
                torch.cuda.max_memory_allocated(self.dispositivo) / MIB
            )
            self.datos["vram_reservada_pico_mib"] = (
                torch.cuda.max_memory_reserved(self.dispositivo) / MIB
            )
