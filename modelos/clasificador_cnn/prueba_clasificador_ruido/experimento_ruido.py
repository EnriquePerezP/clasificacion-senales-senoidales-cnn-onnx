import argparse
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Permite ejecutar el archivo directamente desde PyCharm o la terminal.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelos.clasificador_cnn.division_datos import (
    TAMANO_LOTE,
    crear_cargadores,
    preparar_datos,
)
from modelos.clasificador_cnn.entrenar_modelo import (
    crear_componentes_entrenamiento,
    entrenar_varias_epocas,
)
from modelos.clasificador_cnn.evaluacion_modelo import (
    cargar_modelo_entrenado,
    evaluar_modelo,
)
from modelos.clasificador_cnn.modelo_cnn import DetectorSenoidalCNN
from modelos.dataset import (
    INTENSIDAD_RUIDO_MAXIMA,
    INTENSIDAD_RUIDO_MINIMA,
    SEMILLA,
    crear_dataset,
)
from modelos.ejecucion_entrenamiento import (
    agregar_argumentos_entrenamiento,
    configurar_entrenamiento,
    crear_modelo_inicial,
    guardar_registro,
    resolver_dispositivo,
    resolver_ruta_checkpoint,
)
from modelos.ruido import (
    RUIDO_ALTO,
    RUIDO_BAJO,
    RUIDO_MEDIO,
    RUIDO_MUY_BAJO,
    agregar_ruido,
)


# El cero es la referencia limpia. Los demás niveles vienen de configuracion.yml.
NIVELES_RUIDO = (0.0, RUIDO_MUY_BAJO, RUIDO_BAJO, RUIDO_MEDIO, RUIDO_ALTO)

# El modelo entrenado con aumento de ruido se guarda dentro del experimento.
CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
RUTA_MODELO_RUIDO = CARPETA_MODELOS / "detector_senoidal_cnn_ruido.pt"

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
RUTA_RESULTADOS_CSV = CARPETA_RESULTADOS / "comparacion_ruido.csv"
RUTA_GRAFICA_RUIDO = CARPETA_RESULTADOS / "comparacion_ruido.png"


def agregar_ruido_a_senales(muestras, intensidad, semilla=SEMILLA):
    """Añade ruido independiente a un grupo de señales sin modificarlo."""

    muestras = np.asarray(muestras, dtype=np.float32)
    if muestras.ndim != 3 or muestras.shape[1] != 1:
        raise ValueError(
            "Las muestras deben tener la forma "
            "(cantidad_de_señales, 1, numero_de_muestras)."
        )

    if not np.isfinite(intensidad) or intensidad < 0:
        raise ValueError("La intensidad del ruido debe ser finita y no negativa.")

    if intensidad == 0:
        return muestras.copy()

    generador = np.random.default_rng(semilla)
    resultado = np.empty_like(muestras)

    for indice, señal in enumerate(muestras[:, 0, :]):
        semilla_señal = int(
            generador.integers(0, np.iinfo(np.uint32).max)
        )
        resultado[indice, 0, :] = agregar_ruido(
            señal,
            intensidad=intensidad,
            activado=True,
            semilla=semilla_señal,
        )

    return resultado


def _crear_copias_con_ruido_variable(
    muestras,
    intensidad_minima=INTENSIDAD_RUIDO_MINIMA,
    intensidad_maxima=INTENSIDAD_RUIDO_MAXIMA,
    semilla=SEMILLA,
):
    """Crea una copia de cada señal con una intensidad elegida al azar."""

    if not np.isfinite(intensidad_minima) or intensidad_minima < 0:
        raise ValueError("La intensidad mínima debe ser finita y no negativa.")
    if not np.isfinite(intensidad_maxima) or intensidad_maxima <= 0:
        raise ValueError("La intensidad máxima debe ser finita y mayor que cero.")
    if intensidad_maxima < intensidad_minima:
        raise ValueError(
            "La intensidad máxima no puede ser menor que la mínima."
        )

    muestras = np.asarray(muestras, dtype=np.float32)
    generador = np.random.default_rng(semilla)
    resultado = np.empty_like(muestras)

    for indice, señal in enumerate(muestras[:, 0, :]):
        intensidad = float(
            generador.uniform(intensidad_minima, intensidad_maxima)
        )
        semilla_señal = int(
            generador.integers(0, np.iinfo(np.uint32).max)
        )
        resultado[indice, 0, :] = agregar_ruido(
            señal,
            intensidad=intensidad,
            activado=True,
            semilla=semilla_señal,
        )

    return resultado


def crear_divisiones_aumentadas(divisiones, semilla=SEMILLA):
    """
    Duplica entrenamiento y validación con copias ruidosas.

    El grupo de prueba no se aumenta ni se mezcla con el entrenamiento. Así
    continúa siendo una prueba justa con señales nunca usadas para aprender.
    """

    resultado = {}
    desplazamientos_semilla = {"entrenamiento": 0, "validacion": 1}

    for nombre in ("entrenamiento", "validacion"):
        grupo = divisiones[nombre]
        muestras_limpias = np.asarray(grupo["muestras"], dtype=np.float32)
        clases = np.asarray(grupo["clases"])
        muestras_ruidosas = _crear_copias_con_ruido_variable(
            muestras_limpias,
            semilla=semilla + desplazamientos_semilla[nombre],
        )

        grupo_aumentado = {
            "muestras": np.concatenate(
                (muestras_limpias.copy(), muestras_ruidosas), axis=0
            ),
            "clases": np.concatenate((clases.copy(), clases.copy()), axis=0),
        }
        if "tipos" in grupo:
            tipos = np.asarray(grupo["tipos"])
            grupo_aumentado["tipos"] = np.concatenate(
                (tipos.copy(), tipos.copy()), axis=0
            )
        resultado[nombre] = grupo_aumentado

    resultado["prueba"] = {
        clave: np.asarray(valor).copy()
        for clave, valor in divisiones["prueba"].items()
    }
    return resultado


def _crear_cargador_prueba(muestras, clases):
    tensor_muestras = torch.from_numpy(
        np.asarray(muestras, dtype=np.float32).copy()
    )
    tensor_clases = torch.from_numpy(
        np.asarray(clases, dtype=np.float32).copy()
    ).unsqueeze(dim=1)
    return DataLoader(
        TensorDataset(tensor_muestras, tensor_clases),
        batch_size=TAMANO_LOTE,
        shuffle=False,
    )


def evaluar_niveles_de_ruido(
    modelo,
    grupo_prueba,
    niveles=NIVELES_RUIDO,
    semilla=SEMILLA + 2,
    dispositivo=None,
):
    """Evalúa cada nivel con ruido reproducible e independiente del aumento."""

    resultados = []
    for posicion, intensidad in enumerate(niveles):
        muestras = agregar_ruido_a_senales(
            grupo_prueba["muestras"],
            intensidad=float(intensidad),
            semilla=semilla + posicion,
        )
        cargador = _crear_cargador_prueba(muestras, grupo_prueba["clases"])
        resultado = evaluar_modelo(modelo, cargador, dispositivo)
        resultado["intensidad_ruido"] = float(intensidad)
        resultados.append(resultado)

    return resultados


def _entrenar_modelo(divisiones, ruta_modelo, mostrar_progreso, dispositivo):
    configurar_entrenamiento(dispositivo)
    cargadores = crear_cargadores(divisiones, semilla=SEMILLA)

    modelo, huella = crear_modelo_inicial(DetectorSenoidalCNN, dispositivo)
    funcion_perdida, optimizador = crear_componentes_entrenamiento(modelo)
    resultado = entrenar_varias_epocas(
        modelo=modelo,
        cargadores=cargadores,
        funcion_perdida=funcion_perdida,
        optimizador=optimizador,
        ruta_mejor_modelo=ruta_modelo,
        dispositivo=dispositivo,
        mostrar_progreso=mostrar_progreso,
    )
    resultado["sha256_estado_inicial"] = huella
    return resultado


def entrenar_modelo_con_ruido(
    divisiones, mostrar_progreso=True, dispositivo=None, ruta_modelo=None
):
    """Entrena una nueva CNN con señales limpias y copias ruidosas."""

    dispositivo = torch.device(dispositivo or "cpu")
    ruta_modelo = resolver_ruta_checkpoint(ruta_modelo, RUTA_MODELO_RUIDO, dispositivo)
    divisiones_aumentadas = crear_divisiones_aumentadas(divisiones)
    resultado = _entrenar_modelo(
        divisiones_aumentadas,
        ruta_modelo,
        mostrar_progreso,
        dispositivo,
    )
    resultado["cantidad_entrenamiento"] = len(
        divisiones_aumentadas["entrenamiento"]["clases"]
    )
    resultado["cantidad_validacion"] = len(
        divisiones_aumentadas["validacion"]["clases"]
    )
    return resultado


def guardar_comparacion_csv(
    resultados_limpio, resultados_robusto, ruta_salida=RUTA_RESULTADOS_CSV
):
    """Guarda los números de la comparación para poder revisarlos después."""

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            [
                "intensidad_ruido",
                "exactitud_modelo_limpio",
                "exactitud_modelo_robusto",
                "aciertos_modelo_limpio",
                "aciertos_modelo_robusto",
                "total_senales",
            ]
        )
        for limpio, robusto in zip(resultados_limpio, resultados_robusto):
            escritor.writerow(
                [
                    limpio["intensidad_ruido"],
                    f"{limpio['exactitud']:.6f}",
                    f"{robusto['exactitud']:.6f}",
                    limpio["cantidad_aciertos"],
                    robusto["cantidad_aciertos"],
                    limpio["cantidad_ejemplos"],
                ]
            )


def guardar_grafica(resultados_limpio, resultados_robusto, ruta_salida=RUTA_GRAFICA_RUIDO):
    """Dibuja cuánto cambia la exactitud al aumentar el ruido."""

    niveles = [resultado["intensidad_ruido"] for resultado in resultados_limpio]
    exactitud_limpio = [resultado["exactitud"] * 100 for resultado in resultados_limpio]
    exactitud_robusto = [resultado["exactitud"] * 100 for resultado in resultados_robusto]

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura, grafica = plt.subplots(figsize=(9, 5.5))
    grafica.plot(
        niveles,
        exactitud_limpio,
        marker="o",
        linewidth=2,
        label="Entrenado solo con señales limpias",
    )
    grafica.plot(
        niveles,
        exactitud_robusto,
        marker="o",
        linewidth=2,
        label="Entrenado con señales limpias y ruidosas",
    )
    grafica.set_title("Efecto del ruido en la clasificación")
    grafica.set_xlabel("Intensidad del ruido respecto a la amplitud")
    grafica.set_ylabel("Exactitud (%)")
    grafica.set_xticks(niveles)
    grafica.set_ylim(0, 105)
    grafica.grid(True, alpha=0.3)
    grafica.legend()
    figura.tight_layout()
    figura.savefig(ruta_salida, dpi=160)
    plt.close(figura)


def mostrar_tabla(resultados_limpio, resultados_robusto):
    cantidad_prueba = resultados_limpio[0]["cantidad_ejemplos"]
    print()
    print(f"Comparación final sobre las mismas {cantidad_prueba} señales de prueba")
    print(f"{'Ruido':>7}{'Modelo limpio':>19}{'Modelo aumentado':>22}")
    for limpio, robusto in zip(resultados_limpio, resultados_robusto):
        print(
            f"{limpio['intensidad_ruido']:>7.2%}"
            f"{limpio['exactitud']:>19.2%}"
            f"{robusto['exactitud']:>22.2%}"
        )


def main(argv=None):
    """Compara el entrenamiento con señales limpias y con aumento de ruido."""

    parser = argparse.ArgumentParser(description="Compara el entrenamiento con y sin aumento de ruido.")
    agregar_argumentos_entrenamiento(parser)
    args = parser.parse_args(argv)
    dispositivo = resolver_dispositivo(args.dispositivo)
    configurar_entrenamiento(dispositivo, hilos_cpu=args.hilos_cpu)
    carpeta = args.salida / dispositivo.type / "ruido"
    ruta_modelo_limpio = carpeta / "detector_senoidal_cnn_limpio.pt"
    ruta_modelo_robusto = carpeta / RUTA_MODELO_RUIDO.name

    print(f"Experimento de ruido para el clasificador CNN en {dispositivo}")
    with TemporaryDirectory() as carpeta_temporal:
        ruta_dataset_limpio = Path(carpeta_temporal) / "dataset_senales_limpias.csv"

        print("1. Generando y dividiendo un dataset limpio temporal...")
        crear_dataset(ruta_salida=ruta_dataset_limpio, ruido_activado=False)
        divisiones = preparar_datos(ruta_dataset=ruta_dataset_limpio)

    print("2. Entrenando la CNN con señales limpias...")
    entrenamiento_limpio = _entrenar_modelo(
        divisiones, ruta_modelo_limpio, True, dispositivo
    )
    print(f"   Mejor época: {entrenamiento_limpio['mejor_epoca']}")

    print("3. Entrenando otra CNN con señales limpias y ruidosas...")
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()
    entrenamiento = entrenar_modelo_con_ruido(
        divisiones, dispositivo=dispositivo, ruta_modelo=ruta_modelo_robusto
    )
    print(
        "   Señales usadas para entrenar: "
        f"{entrenamiento['cantidad_entrenamiento']}"
    )
    print(f"   Mejor época: {entrenamiento['mejor_epoca']}")

    modelo_limpio, _ = cargar_modelo_entrenado(ruta_modelo_limpio, dispositivo)
    modelo_robusto, _ = cargar_modelo_entrenado(ruta_modelo_robusto, dispositivo)
    print("4. Evaluando ambos modelos con las mismas señales y el mismo ruido...")
    resultados_limpio = evaluar_niveles_de_ruido(
        modelo_limpio, divisiones["prueba"], dispositivo=dispositivo
    )
    resultados_robusto = evaluar_niveles_de_ruido(
        modelo_robusto, divisiones["prueba"], dispositivo=dispositivo
    )

    for caso, resultado, evaluacion in (
        ("limpio", entrenamiento_limpio, resultados_limpio),
        ("aumentado", entrenamiento, resultados_robusto),
    ):
        for fila in evaluacion:
            fila.pop("ejemplos")
        guardar_registro(
            carpeta, caso, dispositivo, resultado["sha256_estado_inicial"], resultado, evaluacion
        )
    guardar_comparacion_csv(resultados_limpio, resultados_robusto, carpeta / RUTA_RESULTADOS_CSV.name)
    guardar_grafica(resultados_limpio, resultados_robusto, carpeta / RUTA_GRAFICA_RUIDO.name)
    mostrar_tabla(resultados_limpio, resultados_robusto)

    print()
    print(f"Modelo con aumento de ruido guardado en: {ruta_modelo_robusto}")
    print(f"Resultados numéricos guardados en: {carpeta / RUTA_RESULTADOS_CSV.name}")
    print(f"Gráfica guardada en: {carpeta / RUTA_GRAFICA_RUIDO.name}")


if __name__ == "__main__":
    main()
