"""Entrena en CPU y CUDA, y compara inferencia sobre archivos ONNX idénticos."""

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelos.clasificador_cnn.division_datos import preparar_datos
from modelos.clasificador_cnn.entrenar_modelo import RUTA_MEJOR_MODELO
from modelos.clasificador_cnn.prueba_clasificador_ruido.experimento_ruido import RUTA_MODELO_RUIDO
from modelos.dataset import RUTA_DATASET
from modelos.ejecucion_entrenamiento import (
    RUTA_RESULTADOS_GPU,
    configurar_entrenamiento,
    escribir_csv,
    resolver_dispositivo,
)
from modelos.estimador_cnn.entrenar_estimador import RUTA_MEJOR_ESTIMADOR
from modelos.estimador_cnn.preparar_estimador import preparar_datos_estimacion


RAIZ = Path(__file__).resolve().parents[2]
CASOS = {
    "detector": ("detector", RUTA_MEJOR_MODELO.name, "Detector"),
    "estimador": ("estimador", RUTA_MEJOR_ESTIMADOR.name, "Estimador"),
    "limpio": ("ruido", "detector_senoidal_cnn_limpio.pt", "Detector limpio"),
    "aumentado": ("ruido", RUTA_MODELO_RUIDO.name, "Detector aumentado"),
}
DISPOSITIVOS = {"cpu": ("CPU", "#235a9a"), "cuda": ("CUDA", "#bd4b28")}
SCRIPTS = (
    "modelos/clasificador_cnn/entrenar_modelo.py",
    "modelos/estimador_cnn/entrenar_estimador.py",
    "modelos/clasificador_cnn/prueba_clasificador_ruido/experimento_ruido.py",
)


def _leer_csv(ruta):
    with ruta.open(encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def _comparar_entrenamientos(salida):
    resumenes = {}
    todas_epocas = []
    todas_pruebas = []
    comparacion = salida / "comparacion"
    figura, ejes = plt.subplots(len(CASOS), 3, figsize=(12.8, 9.5), layout="constrained")
    figura.suptitle("Entrenamiento en CPU y CUDA")
    for fila, (caso, (subcarpeta, _, nombre)) in enumerate(CASOS.items()):
        par = {}
        for dispositivo, (etiqueta, color) in DISPOSITIVOS.items():
            carpeta = salida / dispositivo / subcarpeta
            resumen, = _leer_csv(carpeta / f"{caso}_resumen.csv")
            historial = _leer_csv(carpeta / f"{caso}_epocas.csv")
            prueba = _leer_csv(carpeta / f"{caso}_prueba.csv")
            par[dispositivo] = resumen
            datos = {"caso": caso, "dispositivo": dispositivo}
            todas_epocas.extend({**datos, **epoca} for epoca in historial)
            todas_pruebas.extend({**datos, **medicion} for medicion in prueba)
            epocas = [int(epoca["epoca"]) for epoca in historial]
            for eje, campo in zip(
                ejes[fila], ("tiempo_epoca_s", "perdida_entrenamiento", "perdida_validacion")
            ):
                eje.plot(epocas, [float(epoca[campo]) for epoca in historial],
                         color=color, label=etiqueta, marker=".", markersize=3,
                         linestyle="-" if dispositivo == "cpu" else "--")

        for campo in ("sha256_estado_inicial", "sha256_datos_divididos", "semilla", "epocas",
                      "lote", "tasa_aprendizaje", "hilos_cpu_pytorch", "version_pytorch"):
            if par["cpu"][campo] != par["cuda"][campo]:
                raise ValueError(f"{nombre}: los registros CPU y CUDA difieren en {campo}.")
        mejora = float(par["cpu"]["tiempo_total_s"]) / float(par["cuda"]["tiempo_total_s"])
        par["cuda"]["razon_tiempo_total_cpu_cuda"] = mejora
        resumenes[caso] = par
        ejes[fila, 0].set_ylabel(f"{nombre}\nSegundos")
    for eje, titulo in zip(ejes[0], ("Tiempo por época", "Pérdida de entrenamiento", "Pérdida de validación")):
        eje.set_title(titulo)
    for eje in ejes[-1]:
        eje.set_xlabel("Época")
    for eje in ejes.flat:
        eje.grid(alpha=0.2)
    ejes[0, 0].legend()

    escribir_csv(comparacion / "entrenamientos.csv",
                 [resumen for par in resumenes.values() for resumen in par.values()])
    escribir_csv(comparacion / "epocas.csv", todas_epocas)
    escribir_csv(comparacion / "evaluacion.csv", todas_pruebas)
    figura.savefig(comparacion / "curvas_entrenamiento.png", dpi=160)
    plt.close(figura)
    _graficar_recursos(comparacion, resumenes)
    _graficar_ruido(comparacion, todas_pruebas)


def _graficar_recursos(carpeta, resumenes):
    figura, ejes = plt.subplots(2, 2, figsize=(11.5, 7), layout="constrained")
    figura.suptitle("Tiempo y memoria del entrenamiento")
    posiciones = np.arange(len(CASOS))
    etiquetas = [caso[2] for caso in CASOS.values()]
    for eje, campo, titulo, unidad in zip(
        ejes.flat,
        ("tiempo_total_s", "tiempo_proceso_cpu_s", "ram_rss_pico_mib", "vram_asignada_pico_mib"),
        ("Tiempo total", "Tiempo de CPU del proceso", "Pico de RAM del proceso", "Pico de VRAM de PyTorch"),
        ("Segundos", "Segundos de CPU", "MiB", "MiB"),
    ):
        for dispositivo, (etiqueta, color) in DISPOSITIVOS.items():
            if campo.startswith("vram") and dispositivo == "cpu":
                continue
            desplazamiento = 0 if campo.startswith("vram") else (-0.19 if dispositivo == "cpu" else 0.19)
            valores = [float(par[dispositivo][campo]) for par in resumenes.values()]
            eje.bar(posiciones + desplazamiento, valores, 0.36, label=etiqueta, color=color)
        eje.set(title=titulo, ylabel=unidad, xticks=posiciones, xticklabels=etiquetas)
        eje.tick_params(axis="x", labelsize=8)
        eje.grid(axis="y", alpha=0.2)
    ejes[0, 0].legend()
    figura.savefig(carpeta / "recursos_entrenamiento.png", dpi=160)
    plt.close(figura)


def _graficar_ruido(carpeta, pruebas):
    figura, ejes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True, layout="constrained")
    figura.suptitle("Exactitud sobre las mismas señales de prueba con ruido")
    for eje, caso in zip(ejes, ("limpio", "aumentado")):
        for dispositivo, (etiqueta, color) in DISPOSITIVOS.items():
            filas = [fila for fila in pruebas if fila["caso"] == caso and fila["dispositivo"] == dispositivo]
            eje.plot([float(fila["intensidad_ruido"]) * 100 for fila in filas],
                     [float(fila["exactitud"]) * 100 for fila in filas],
                     marker="o" if dispositivo == "cpu" else "x",
                     linestyle="-" if dispositivo == "cpu" else "--",
                     color=color, label=etiqueta)
        eje.set(title=CASOS[caso][2], xlabel="Intensidad del ruido (%)", ylim=(0, 105))
        eje.grid(alpha=0.2)
    ejes[0].set_ylabel("Exactitud (%)")
    ejes[0].legend()
    figura.savefig(carpeta / "ruido_cpu_cuda.png", dpi=160)
    plt.close(figura)


def _exportar_y_medir(salida, dispositivo, hilos, repeticiones, calentamiento, origen):
    from modelos.clasificador_cnn.evaluacion_modelo import cargar_modelo_entrenado
    from modelos.estimador_cnn.evaluacion_modelo import cargar_estimador

    # Los scripts locales comparten nombre de carpeta con el paquete instalado onnx.
    sys.path.insert(0, str(RAIZ / "onnx"))
    from benchmark_onnx import comparar_inferencia
    from exportacion_onnx import exportar_checkpoint_onnx

    muestras_detector = preparar_datos()["prueba"]["muestras"]
    muestras_estimador = preparar_datos_estimacion()["prueba"]["muestras"]
    comparacion = salida / "comparacion"
    verificaciones, resumen, mediciones = [], [], []
    for caso, (subcarpeta, checkpoint, _) in CASOS.items():
        es_estimador = caso == "estimador"
        tipo = "estimador" if es_estimador else "detector"
        cargar = cargar_estimador if es_estimador else cargar_modelo_entrenado
        muestras = muestras_estimador if es_estimador else muestras_detector
        lotes = (muestras[:1], muestras[:32])
        for origen_pt in DISPOSITIVOS:
            ruta_pt = salida / origen_pt / subcarpeta / checkpoint
            ruta_onnx = ruta_pt.with_suffix(".onnx")
            diferencia_onnx = exportar_checkpoint_onnx(tipo, ruta_pt, ruta_onnx)
            modelo_cpu, _ = cargar(ruta_pt)
            modelo_cuda, _ = cargar(ruta_pt, dispositivo)
            with torch.inference_mode():
                entrada = torch.from_numpy(lotes[-1])
                cpu = modelo_cpu(entrada).numpy()
                cuda = modelo_cuda(entrada.to(dispositivo)).cpu().numpy()
            np.testing.assert_allclose(cpu, cuda, rtol=1e-5, atol=1e-5)
            verificaciones.append({
                "caso": caso,
                "dispositivo_entrenamiento": origen_pt,
                "archivo_onnx": ruta_onnx.relative_to(salida).as_posix(),
                "sha256_onnx": hashlib.sha256(ruta_onnx.read_bytes()).hexdigest(),
                "diferencia_maxima_pytorch_onnx_cpu": diferencia_onnx,
                "diferencia_maxima_checkpoint_cpu_cuda": float(np.max(np.abs(cpu - cuda))),
            })
            print(f"Checkpoint portable y ONNX verificado: {ruta_onnx}")
    escribir_csv(comparacion / "verificacion_onnx.csv", verificaciones)

    for caso, (subcarpeta, checkpoint, _) in CASOS.items():
        ruta = (salida / origen / subcarpeta / checkpoint).with_suffix(".onnx")
        muestras = muestras_estimador if caso == "estimador" else muestras_detector
        resultados, latencias = comparar_inferencia(
            ruta, (muestras[:1], muestras[:32]), dispositivo, hilos, repeticiones, calentamiento
        )
        datos = {"caso": caso, "archivo_onnx": ruta.relative_to(salida).as_posix()}
        resumen.extend({**datos, **fila} for fila in resultados)
        mediciones.extend({**datos, **fila} for fila in latencias)
    escribir_csv(comparacion / "inferencia_onnx.csv", resumen)
    escribir_csv(comparacion / "latencias_onnx.csv", mediciones)
    _graficar_inferencia(comparacion, resumen)


def _graficar_inferencia(carpeta, resumen):
    figura, ejes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    figura.suptitle("Inferencia en ONNX Runtime con los mismos archivos")
    posiciones = np.arange(len(CASOS))
    for eje, lote in zip(ejes, (1, 32)):
        for dispositivo, (etiqueta, color) in DISPOSITIVOS.items():
            proveedor = f"{etiqueta}ExecutionProvider"
            latencias = [
                fila["latencia_mediana_ms"]
                for fila in resumen if fila["lote"] == lote and fila["proveedor"] == proveedor
            ]
            desplazamiento = -0.19 if dispositivo == "cpu" else 0.19
            eje.bar(posiciones + desplazamiento, latencias, 0.36, label=etiqueta, color=color)
        eje.set(title=f"Lote de {lote}", ylabel="Mediana (ms)", xticks=posiciones,
                xticklabels=[caso[2] for caso in CASOS.values()])
        eje.tick_params(axis="x", labelrotation=15, labelsize=8)
        eje.grid(axis="y", alpha=0.2)
    ejes[0].legend()
    figura.savefig(carpeta / "latencia_onnx.png", dpi=160)
    plt.close(figura)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RUTA_RESULTADOS_GPU)
    parser.add_argument("--fase", choices=("todo", "entrenamiento", "onnx"), default="todo")
    parser.add_argument("--hilos-cpu", type=int, default=torch.get_num_threads())
    parser.add_argument("--repeticiones", type=int, default=200)
    parser.add_argument("--calentamiento", type=int, default=30)
    parser.add_argument("--origen-onnx", choices=DISPOSITIVOS, default="cuda")
    args = parser.parse_args(argv)
    if min(args.hilos_cpu, args.repeticiones, args.calentamiento) < 1:
        parser.error("Usa valores mayores que cero para hilos, repeticiones y calentamiento.")
    salida = args.salida.resolve()
    dispositivo = resolver_dispositivo("cuda")
    configurar_entrenamiento(dispositivo, hilos_cpu=args.hilos_cpu)
    # Fuerza una operación real antes de comenzar una comparación de entrenamiento.
    torch.ones(1, device=dispositivo).add_(1)
    torch.cuda.synchronize(dispositivo)

    if args.fase in ("todo", "entrenamiento"):
        inicio = datetime.now(timezone.utc).isoformat()
        for nombre in DISPOSITIVOS:
            for script in SCRIPTS:
                subprocess.run(
                    [sys.executable, "-u", str(RAIZ / script), "--dispositivo", nombre,
                     "--hilos-cpu", str(args.hilos_cpu), "--salida", str(salida)],
                    cwd=RAIZ, env={**os.environ, "MPLBACKEND": "Agg"}, check=True,
                )
        _comparar_entrenamientos(salida)
        escribir_csv(salida / "comparacion" / "ejecucion.csv", [{
            "inicio_utc": inicio,
            "fin_entrenamientos_utc": datetime.now(timezone.utc).isoformat(),
            "sha256_configuracion": hashlib.sha256((RAIZ / "configuracion.yml").read_bytes()).hexdigest(),
            "sha256_dataset": hashlib.sha256(RUTA_DATASET.read_bytes()).hexdigest(),
            "python": sys.version.split()[0],
        }])
    if args.fase in ("todo", "onnx"):
        _exportar_y_medir(
            salida, dispositivo, args.hilos_cpu, args.repeticiones,
            args.calentamiento, args.origen_onnx,
        )
    print(f"Comparación guardada en {salida / 'comparacion'}")


if __name__ == "__main__":
    main()
