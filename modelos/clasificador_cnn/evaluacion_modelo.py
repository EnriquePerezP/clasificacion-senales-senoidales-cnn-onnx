import sys
from numbers import Real
from pathlib import Path

import torch
from torch import nn

# Permite ejecutar este archivo directamente desde cualquier carpeta.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelos.clasificador_cnn.division_datos import (
    crear_cargadores,
    preparar_datos,
)
from modelos.clasificador_cnn.entrenar_modelo import RUTA_MEJOR_MODELO
from modelos.clasificador_cnn.modelo_cnn import DetectorSenoidalCNN
from modelos.dataset import SEMILLA
from modelos.generador_senales import CONFIGURACION, NUMERO_MUESTRAS


PARAMETROS_EVALUACION = CONFIGURACION["Evaluacion"]
UMBRAL_CLASIFICACION = PARAMETROS_EVALUACION["Umbral de clasificacion"]


def cargar_modelo_entrenado(ruta_modelo=RUTA_MEJOR_MODELO, dispositivo=None):
    """Reconstruye la CNN y carga los valores aprendidos durante el entrenamiento."""

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    ruta_modelo = Path(ruta_modelo)
    if not ruta_modelo.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado: {ruta_modelo}. "
            "Primero ejecuta entrenar_modelo.py."
        )

    contenido = torch.load(
        ruta_modelo,
        map_location=dispositivo,
        weights_only=True,
    )

    campos_necesarios = {
        "estado_modelo",
        "numero_muestras",
        "epoca",
        "perdida_validacion",
        "exactitud_validacion",
    }
    campos_faltantes = campos_necesarios.difference(contenido)
    if campos_faltantes:
        raise ValueError(
            "El archivo del modelo está incompleto. Faltan: "
            + ", ".join(sorted(campos_faltantes))
        )

    modelo = DetectorSenoidalCNN(
        numero_muestras=int(contenido["numero_muestras"])
    )
    modelo.load_state_dict(contenido["estado_modelo"])
    modelo.to(dispositivo)
    modelo.eval()

    return modelo, contenido


def _division_segura(numerador, denominador):
    """Evita una división entre cero cuando una clase no aparece."""

    return numerador / denominador if denominador else 0.0


def evaluar_modelo(
    modelo,
    cargador_prueba,
    dispositivo=None,
    umbral=UMBRAL_CLASIFICACION,
):
    """
    Evalúa la CNN sin modificar sus valores aprendidos.

    La probabilidad igual o mayor al umbral se interpreta como senoidal.
    Una probabilidad menor se interpreta como no senoidal.
    """

    if dispositivo is None:
        dispositivo = torch.device("cpu")

    if not isinstance(umbral, Real) or isinstance(umbral, bool):
        raise TypeError("El umbral de clasificación debe ser un número.")
    if not 0 < umbral < 1:
        raise ValueError("El umbral debe ser mayor que 0 y menor que 1.")

    modelo.to(dispositivo)
    modelo.eval()
    funcion_perdida = nn.BCEWithLogitsLoss()

    perdida_acumulada = 0.0
    cantidad_ejemplos = 0
    verdaderos_positivos = 0
    verdaderos_negativos = 0
    falsos_positivos = 0
    falsos_negativos = 0
    confianza_acumulada = 0.0
    ejemplos = {}

    with torch.no_grad():
        for muestras, clases_correctas in cargador_prueba:
            muestras = muestras.to(dispositivo)
            clases_correctas = clases_correctas.to(dispositivo)

            salidas = modelo(muestras)
            probabilidades = torch.sigmoid(salidas)
            predicciones = (
                probabilidades >= umbral
            ).to(dtype=clases_correctas.dtype)

            cantidad_lote = len(clases_correctas)
            perdida = funcion_perdida(salidas, clases_correctas)
            perdida_acumulada += perdida.item() * cantidad_lote
            cantidad_ejemplos += cantidad_lote

            verdaderos_positivos += int(
                ((predicciones == 1) & (clases_correctas == 1)).sum().item()
            )
            verdaderos_negativos += int(
                ((predicciones == 0) & (clases_correctas == 0)).sum().item()
            )
            falsos_positivos += int(
                ((predicciones == 1) & (clases_correctas == 0)).sum().item()
            )
            falsos_negativos += int(
                ((predicciones == 0) & (clases_correctas == 1)).sum().item()
            )

            # Si predice clase 1 usamos su probabilidad; si predice clase 0,
            # usamos la probabilidad contraria. Es la seguridad de su decisión.
            confianza_prediccion = torch.where(
                predicciones == 1,
                probabilidades,
                1 - probabilidades,
            )
            confianza_acumulada += float(confianza_prediccion.sum().item())

            # Conservamos como máximo un ejemplo de cada resultado posible.
            # Después se utilizarán para explicar visualmente los aciertos
            # y errores sin guardar nuevamente todo el grupo de prueba.
            for indice in range(cantidad_lote):
                clase_real = int(clases_correctas[indice].item())
                clase_predicha = int(predicciones[indice].item())

                if clase_real == 1 and clase_predicha == 1:
                    categoria = "senoidal_correcta"
                elif clase_real == 0 and clase_predicha == 0:
                    categoria = "no_senoidal_correcta"
                elif clase_real == 0 and clase_predicha == 1:
                    categoria = "falso_positivo"
                else:
                    categoria = "falso_negativo"

                if categoria not in ejemplos:
                    ejemplos[categoria] = {
                        "muestras": muestras[indice, 0].detach().cpu().numpy().copy(),
                        "clase_real": clase_real,
                        "clase_predicha": clase_predicha,
                        "probabilidad_senoidal": float(
                            probabilidades[indice].item()
                        ),
                    }

    if cantidad_ejemplos == 0:
        raise ValueError("El cargador de prueba está vacío.")

    cantidad_aciertos = verdaderos_positivos + verdaderos_negativos
    cantidad_errores = falsos_positivos + falsos_negativos

    return {
        "umbral": umbral,
        "cantidad_ejemplos": cantidad_ejemplos,
        "cantidad_aciertos": cantidad_aciertos,
        "cantidad_errores": cantidad_errores,
        "perdida": perdida_acumulada / cantidad_ejemplos,
        "exactitud": cantidad_aciertos / cantidad_ejemplos,
        "confianza_promedio": confianza_acumulada / cantidad_ejemplos,
        "verdaderos_positivos": verdaderos_positivos,
        "verdaderos_negativos": verdaderos_negativos,
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
        "precision_senoidal": _division_segura(
            verdaderos_positivos,
            verdaderos_positivos + falsos_positivos,
        ),
        "deteccion_senoidales": _division_segura(
            verdaderos_positivos,
            verdaderos_positivos + falsos_negativos,
        ),
        "ejemplos": ejemplos,
    }


def mostrar_resultado(resultado, informacion_modelo, ruta_modelo):
    """Presenta las métricas usando nombres fáciles de relacionar."""

    print("Evaluación final de la CNN 1D")
    print(f"Modelo cargado: {ruta_modelo}")
    print(f"Mejor época guardada: {informacion_modelo['epoca']}")
    print(
        "Exactitud que tuvo en validación: "
        f"{informacion_modelo['exactitud_validacion']:.2%}"
    )
    print()
    print(f"Señales de prueba: {resultado['cantidad_ejemplos']}")
    print(f"Umbral de clasificación: {resultado['umbral']:.2%}")
    print(f"Clasificaciones correctas: {resultado['cantidad_aciertos']}")
    print(f"Clasificaciones incorrectas: {resultado['cantidad_errores']}")
    print(f"Exactitud final: {resultado['exactitud']:.2%}")
    print(f"Seguridad promedio del modelo: {resultado['confianza_promedio']:.2%}")
    print()
    print("Matriz de confusión")
    print("                         Predijo NO     Predijo SÍ")
    print(
        "Realmente no senoidal "
        f"{resultado['verdaderos_negativos']:11d} "
        f"{resultado['falsos_positivos']:14d}"
    )
    print(
        "Realmente senoidal    "
        f"{resultado['falsos_negativos']:11d} "
        f"{resultado['verdaderos_positivos']:14d}"
    )
    print()
    print(
        "Cuando dijo 'senoidal', acertó: "
        f"{resultado['precision_senoidal']:.2%}"
    )
    print(
        "De todas las senoidales, detectó: "
        f"{resultado['deteccion_senoidales']:.2%}"
    )


def main(mostrar=True):
    """Evalúa el modelo guardado y genera las gráficas del grupo de prueba."""

    torch.manual_seed(SEMILLA)
    dispositivo = torch.device("cpu")

    modelo, informacion_modelo = cargar_modelo_entrenado(
        ruta_modelo=RUTA_MEJOR_MODELO,
        dispositivo=dispositivo,
    )
    if modelo.numero_muestras != NUMERO_MUESTRAS:
        raise ValueError(
            f"El modelo espera {modelo.numero_muestras} muestras por señal, "
            f"pero configuracion.yml indica {NUMERO_MUESTRAS}. "
            "Usa la configuración de muestreo con la que entrenaste el modelo."
        )

    divisiones = preparar_datos(semilla=SEMILLA)
    cargadores = crear_cargadores(divisiones, semilla=SEMILLA)
    resultado = evaluar_modelo(
        modelo=modelo,
        cargador_prueba=cargadores["prueba"],
        dispositivo=dispositivo,
    )
    mostrar_resultado(
        resultado=resultado,
        informacion_modelo=informacion_modelo,
        ruta_modelo=RUTA_MEJOR_MODELO,
    )

    from representaciones_visuales.resultados_modelo_clasificador import (
        crear_visualizaciones,
    )

    crear_visualizaciones(
        resultado,
        modelo=modelo,
        cargador_prueba=cargadores["prueba"],
        tipos=divisiones["prueba"]["tipos"],
        mostrar=mostrar,
    )


if __name__ == "__main__":
    main()
