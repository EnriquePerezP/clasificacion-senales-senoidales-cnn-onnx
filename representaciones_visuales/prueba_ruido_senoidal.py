from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasificador_senales.generador_senales import generar_senoidal
from clasificador_senales.ruido import (
    CONFIGURACION,
    RUIDO_ACTIVADO,
    RUIDO_ALTO,
    RUIDO_BAJO,
    RUIDO_MEDIO,
    RUIDO_MUY_BAJO,
    agregar_ruido,
)


FRECUENCIA_EJEMPLO = 200
AMPLITUD_EJEMPLO = 0.75
FASE_EJEMPLO = 0
TIEMPO_VISIBLE = 0.030
SEMILLA_RUIDO = CONFIGURACION["Semilla"]

# Cada pareja contiene el nombre mostrado y la intensidad utilizada.
NIVELES_RUIDO = [
    ("Sin ruido", 0.00),
    ("Ruido muy bajo", RUIDO_MUY_BAJO),
    ("Ruido bajo", RUIDO_BAJO),
    ("Ruido medio", RUIDO_MEDIO),
    ("Ruido alto", RUIDO_ALTO),
]


def mostrar_comparacion():
    """Compara una misma senoidal limpia con distintos niveles de ruido."""

    tiempo, señal_original = generar_senoidal(
        frecuencia=FRECUENCIA_EJEMPLO,
        amplitud=AMPLITUD_EJEMPLO,
        fase_grados=FASE_EJEMPLO,
    )

    figura, graficas = plt.subplots(
        len(NIVELES_RUIDO),
        1,
        figsize=(11, 2.4 * len(NIVELES_RUIDO)),
        sharex=True,
    )

    muestras_visibles = tiempo <= TIEMPO_VISIBLE

    for grafica, (nombre, intensidad) in zip(graficas, NIVELES_RUIDO):
        señal_resultante = agregar_ruido(
            muestras=señal_original,
            intensidad=intensidad,
            activado=RUIDO_ACTIVADO,
            semilla=SEMILLA_RUIDO,
        )

        grafica.plot(
            tiempo[muestras_visibles] * 1000,
            señal_resultante[muestras_visibles],
            label=nombre,
        )

        # La original se dibuja como referencia cuando existe ruido.
        if RUIDO_ACTIVADO and intensidad > 0:
            grafica.plot(
                tiempo[muestras_visibles] * 1000,
                señal_original[muestras_visibles],
                linestyle="--",
                alpha=0.65,
                label="Señal original",
            )

        estado = "activado" if RUIDO_ACTIVADO else "desactivado"
        grafica.set_title(
            f"{nombre} | intensidad {intensidad:.2f} | ruido {estado}"
        )
        grafica.set_ylabel("Amplitud")
        grafica.grid(True)
        grafica.legend(loc="upper right")

        diferencia_maxima = np.max(
            np.abs(señal_resultante - señal_original)
        )
        print(
            f"{nombre}: intensidad={intensidad:.2f}, "
            f"diferencia máxima={diferencia_maxima:.4f}"
        )

    graficas[-1].set_xlabel("Tiempo en milisegundos")

    figura.suptitle(
        "Comparación de una onda senoidal con ruido opcional",
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


if __name__ == "__main__":
    mostrar_comparacion()
