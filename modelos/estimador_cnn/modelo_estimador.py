from numbers import Integral

import torch
from torch import nn

from modelos.generador_senales import NUMERO_MUESTRAS


class EstimadorParametrosCNN(nn.Module):
    """
    CNN 1D que estima la frecuencia, amplitud y fase de una senoidal.

    Recibe lotes con forma (cantidad de senales, 1, numero de muestras).
    Devuelve frecuencia y amplitud normalizadas, seguidas del seno y
    coseno de la fase.
    """

    CANTIDAD_SALIDAS = 4

    def __init__(self, numero_muestras=NUMERO_MUESTRAS):
        super().__init__()

        if not isinstance(numero_muestras, Integral) or isinstance(numero_muestras, bool):
            raise TypeError("El numero de muestras debe ser un entero.")
        if numero_muestras < 4:
            raise ValueError("Se necesitan al menos 4 muestras para los dos MaxPool1d.")

        self.numero_muestras = numero_muestras
        self.bloque_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.bloque_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.bloque_3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # El resumen conserva ocho posiciones de la señal, que aportan
        # información sobre el inicio de la onda para estimar la fase.
        self.resumen = nn.AdaptiveAvgPool1d(output_size=8)
        self.capas_compartidas = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8, out_features=128),
            nn.ReLU(),
        )

        self.salida_frecuencia = nn.Linear(128, 1)
        self.salida_amplitud = nn.Linear(128, 1)
        self.salida_fase = nn.Linear(128, 2)

    def extraer_caracteristicas(self, senales):
        """Extrae un resumen que conserva información de la posición temporal."""

        if senales.ndim != 3:
            raise ValueError(
                "La entrada debe tener tres dimensiones: "
                "(cantidad de senales, 1, numero de muestras)."
            )
        if senales.shape[1] != 1:
            raise ValueError("Cada senal debe tener exactamente un canal.")
        if senales.shape[2] != self.numero_muestras:
            raise ValueError(f"Cada senal debe contener {self.numero_muestras} muestras.")

        resultado = self.bloque_1(senales)
        resultado = self.bloque_2(resultado)
        resultado = self.bloque_3(resultado)
        resultado = self.resumen(resultado)
        return self.capas_compartidas(resultado)

    def forward(self, senales):
        """Devuelve los cuatro valores internos estimados para cada señal."""

        caracteristicas = self.extraer_caracteristicas(senales)
        frecuencia = self.salida_frecuencia(caracteristicas)
        amplitud = self.salida_amplitud(caracteristicas)
        fase = self.salida_fase(caracteristicas)
        return torch.cat((frecuencia, amplitud, fase), dim=1)
