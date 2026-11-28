# Clasificación de señales senoidales con CNN y ONNX Runtime

## Entrenamiento experimental en CPU y CUDA

La rama `experimento/gpu` admite `--dispositivo cpu`, `--dispositivo cuda` y
`--dispositivo auto`. La selección automática usa CUDA cuando PyTorch la
detecta. Una solicitud explícita de CUDA falla si el dispositivo no está
disponible.

En Windows, este entorno de Python 3.11 usa PyTorch 2.14.1 con CUDA 13.0 y
ONNX Runtime GPU 1.32.0. Para instalarlo en un entorno separado, ejecuta lo
siguiente desde la raíz con Python 3.11 activo.

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv-gpu\Scripts\python.exe -m pip install numpy PyYAML matplotlib onnx onnxscript psutil onnxruntime-gpu==1.32.0
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device='cuda')); print(torch.cuda.get_device_name())"
```

Para un entorno de CPU puedes instalar `requirements.txt`. Mantén una sola
distribución de ONNX Runtime en cada entorno. Consulta las matrices de
compatibilidad al usar otras versiones. [PyTorch](https://pytorch.org/get-started/previous-versions/) · [ONNX Runtime CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

### Ejecutar los entrenamientos

```powershell
.\.venv-gpu\Scripts\python.exe modelos/clasificador_cnn/entrenar_modelo.py --dispositivo auto
.\.venv-gpu\Scripts\python.exe modelos/estimador_cnn/entrenar_estimador.py --dispositivo auto
.\.venv-gpu\Scripts\python.exe modelos/clasificador_cnn/prueba_clasificador_ruido/experimento_ruido.py --dispositivo auto
```

Cada script escribe sus modelos, épocas y métricas de prueba en
`experimentos/gpu/resultados/<cpu|cuda>/`. Usa `--salida` para dirigir otra
ejecución a una carpeta propia. Los checkpoints guardan el estado del modelo
y del optimizador en CPU y se cargan en el dispositivo elegido. Los modelos y
los archivos ONNX que usa el flujo original tienen sus propias rutas.

### Comparar CPU con CUDA

```powershell
.\.venv-gpu\Scripts\python.exe experimentos/gpu/comparar.py --hilos-cpu 12
```

El comparador usa el dataset, las divisiones, la semilla 42 y los
hiperparámetros del YAML actual. Ejecuta cada script en un proceso propio.
Comprueba las huellas SHA-256 de los pesos iniciales y de las particiones
antes de cruzar los resultados. El experimento de ruido conserva las dos
condiciones originales y aplica las mismas realizaciones de ruido en prueba.
El estimador conserva sus ocho ventanas de promedio adaptativo. En CUDA, el
entrenamiento calcula esas ventanas con reducciones deterministas porque el
gradiente CUDA de la implementación estándar no ofrece ese modo.

- El tiempo por época incluye entrenamiento y validación. El total abarca las
  épocas y el guardado de los mejores checkpoints. Los tiempos incluyen la
  primera época y sincronizan CUDA en los límites de medición.
- El tiempo de CPU del proceso suma el trabajo de sus hilos. La RAM corresponde
  al pico de memoria residente observado con muestreo cada 10 ms. La VRAM es
  el pico de memoria asignada y reservada por el asignador de PyTorch.
- Se exportan y verifican ocho checkpoints. Las comparaciones ONNX contra
  PyTorch usan lotes de 1 y 3. Para comprobar la portabilidad, se cargan los
  mismos pesos en PyTorch CPU y CUDA y se comparan sus salidas.
- Para cada modelo y tamaño de lote (1 y 32), las dos sesiones de ONNX Runtime
  leen el mismo archivo y reciben las mismas entradas. Se comprueba la
  ejecución CUDA con un perfil separado de la medición. Tras 30 llamadas de
  calentamiento se miden 200 llamadas por proveedor. El tiempo incluye las
  transferencias que requiera una llamada con entradas y salidas en CPU.
  `--origen-onnx cpu` permite usar como referencia los archivos del
  entrenamiento en CPU. La referencia predeterminada es `cuda`.

Los CSV y las gráficas agregadas se guardan en
[`experimentos/gpu/resultados/comparacion/`](experimentos/gpu/resultados/comparacion/).
`--fase entrenamiento` ejecuta solo los entrenamientos y
`--fase onnx` reutiliza los checkpoints existentes para exportar y medir.
Para exportar un checkpoint individual, usa `onnx/exportacion_onnx.py` con
`--tipo detector|estimador --checkpoint RUTA.pt --salida RUTA.onnx`.

### Resultados del 27 de noviembre de 2026

Se midieron 20 épocas por caso con un Intel Core i5-12500H y una NVIDIA
GeForce RTX 3050 Laptop GPU de 4 GiB. Se usaron 12 hilos de PyTorch para
las operaciones en CPU y 12 hilos de ONNX Runtime por sesión. Se repitieron
los ocho entrenamientos en otra carpeta de salida. Para cada dispositivo,
coincidieron exactamente las métricas por época y de prueba y los tensores
de los checkpoints. [Registro de reproducibilidad](experimentos/gpu/resultados/comparacion/reproducibilidad.csv)

| Entrenamiento | CPU | CUDA | Tiempo CPU dividido por tiempo CUDA |
| --- | ---: | ---: | ---: |
| Detector | 3.482 s | 1.615 s | 2.16× |
| Estimador | 1.915 s | 1.165 s | 1.64× |
| Detector limpio | 3.667 s | 1.580 s | 2.32× |
| Detector con aumento de ruido | 6.626 s | 2.389 s | 2.77× |

El detector obtuvo 87.78 % de exactitud de prueba al entrenar en CPU y
86.67 % al entrenar en CUDA. El estimador obtuvo un error absoluto medio de
frecuencia de 33.81 Hz y 37.39 Hz, respectivamente. En la prueba con 20 %
de ruido, el detector aumentado obtuvo 91.11 % y 92.22 %, respectivamente.

En las sesiones ONNX, CPU tuvo menor latencia mediana para lotes de una
señal y CUDA tuvo menor latencia para lotes de 32 en los cuatro modelos.
Para el detector, las medianas CPU/CUDA fueron 0.076/0.405 ms con una señal
y 0.949/0.457 ms con 32 señales. Estos tiempos corresponden al equipo y a
las ejecuciones registrados en los CSV.
