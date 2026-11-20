import sys
from numbers import Integral
from pathlib import Path

import torch
from torch import nn

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clasificador_senales.generador_senales import NUMERO_MUESTRAS


class DetectorSenoidalCNN(nn.Module):
    """
    CNN 1D que distingue entre una señal senoidal y una no senoidal.

    La entrada debe tener esta forma:
        (cantidad_de_señales, 1, numero_de_muestras)

    Por defecto, el número de muestras se toma de configuracion.yml.

    El modelo devuelve un valor por señal. Ese valor todavía no es un
    porcentaje; durante el entrenamiento se utilizará directamente para
    calcular el error de la red.
    """

    def __init__(self, numero_muestras=NUMERO_MUESTRAS):
        super().__init__()

        if not isinstance(numero_muestras, Integral) or isinstance(numero_muestras, bool):
            raise TypeError("El número de muestras debe ser un entero.")
        if numero_muestras < 4:
            raise ValueError(
                "Se necesitan al menos 4 muestras para los dos MaxPool1d."
            )

        self.numero_muestras = numero_muestras

        # Primer bloque: aprende patrones pequeños directamente de las muestras.
        self.bloque_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Segundo bloque: combina los patrones encontrados por el primero.
        self.bloque_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Tercer bloque: aprende patrones más completos de cada señal.
        self.bloque_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Resume cada uno de los 64 patrones aprendidos en un solo valor.
        self.resumen = nn.AdaptiveAvgPool1d(output_size=1)

        # Produce una sola respuesta: senoidal o no senoidal.
        self.clasificador = nn.Linear(in_features=64, out_features=1)

    def extraer_caracteristicas(self, señales):
        """Devuelve los 64 resúmenes que la CNN aprendió de cada señal."""

        if señales.ndim != 3:
            raise ValueError(
                "La entrada debe tener tres dimensiones: "
                "(cantidad_de_señales, 1, numero_de_muestras)."
            )

        if señales.shape[1] != 1:
            raise ValueError("Cada señal debe tener exactamente un canal.")

        if señales.shape[2] != self.numero_muestras:
            raise ValueError(
                f"Cada señal debe contener {self.numero_muestras} muestras."
            )

        resultado = self.bloque_1(señales)
        resultado = self.bloque_2(resultado)
        resultado = self.bloque_3(resultado)
        resultado = self.resumen(resultado)

        # Cambia (cantidad, 64, 1) por (cantidad, 64).
        return resultado.squeeze(dim=-1)

    def forward(self, señales):
        """Realiza el recorrido completo y devuelve la clasificación."""

        caracteristicas = self.extraer_caracteristicas(señales)
        return self.clasificador(caracteristicas)


def contar_parametros_entrenables(modelo):
    """Cuenta los valores internos que la CNN aprenderá al entrenarse."""

    return sum(parametro.numel() for parametro in modelo.parameters() if parametro.requires_grad)


def main():
    """Muestra la estructura y comprueba el recorrido con una entrada vacía."""

    modelo = DetectorSenoidalCNN()
    entrada_de_ejemplo = torch.zeros(1, 1, modelo.numero_muestras, dtype=torch.float32)

    modelo.eval()
    with torch.no_grad():
        salida = modelo(entrada_de_ejemplo)

    print("CNN 1D creada correctamente")
    print(modelo)
    print(f"Forma de la entrada: {tuple(entrada_de_ejemplo.shape)}")
    print(f"Forma de la salida: {tuple(salida.shape)}")
    print(f"Parámetros que aprenderá: {contar_parametros_entrenables(modelo):,}")


if __name__ == "__main__":
    main()
