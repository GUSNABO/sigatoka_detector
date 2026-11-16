# Detector de Sigatoka Negra con Deep Learning 🍌🦠

Este proyecto implementa un sistema de Inteligencia Artificial basado en **PyTorch**, **SqueezeNet** y **SVM** para la detección y clasificación de la enfermedad de la **Sigatoka Negra** en hojas de banano. 

La arquitectura es híbrida: **SqueezeNet** actúa como extractor de características profundas (Deep Feature Extractor) y un **SVM** realiza la clasificación final. Esta combinación conserva la eficiencia para Edge AI y prioriza la sensibilidad (Recall) por fase.

---

## 📁 Estructura del Código

El sistema está dividido en dos partes principales que cumplen roles muy distintos: el "Cerebro" que aprende y la "Interfaz" que interactúa con el usuario.

### 1. El Entrenamiento (`train_sigatoka.py`)
Este es el motor de aprendizaje híbrido del proyecto. Sus responsabilidades incluyen:
*   **Procesamiento de Datos:** Carga las imágenes, aplica Data Augmentation suave y redimensiona al estándar requerido (227x227 píxeles).
*   **Extracción de Características:** Usa SqueezeNet preentrenada para obtener embeddings profundos de cada imagen.
*   **Clasificación con SVM:** Entrena un SVM con ponderación de clases para priorizar el Recall por fase.
*   **Evaluación y Guardado:** Reporta Recall y F1 Macro, genera matrices de confusión y guarda el modelo final en `sigatoka_svm.joblib`.

### 2. La Aplicación Web (`app_sigatoka.py`)
Esta es la interfaz gráfica, construida con la librería Streamlit. Está diseñada para ser amigable con un usuario final (como un agricultor o agrónomo). Sus tareas son:
*   **Carga del Conocimiento:** Lee el archivo `sigatoka_svm.joblib` generado en el entrenamiento.
*   **Interacción:** Muestra una página web donde el usuario puede cargar o arrastrar una fotografía de una hoja de banano.
*   **Diagnóstico:** Extrae características con SqueezeNet y clasifica con SVM para devolver un diagnóstico instantáneo de la fase de infección.

### 3. Descarga de No_hoja (`download_no_hoja.py`)
Script auxiliar para descargar imágenes negativas (objetos que no son hojas de banano) y crear la clase `No_hoja` de forma automática.

---

## ⚙️ Instalación de Dependencias

Antes de ejecutar cualquier script, asegúrate de tener instaladas todas las librerías necesarias. En tu terminal, ejecuta:

```bash
pip install -r requirements.txt
```

---

## 🔄 Flujo de SqueezeNet + SVM en orden

El proyecto no entrena una red completa para clasificar directamente. La arquitectura es híbrida y trabaja en dos etapas:

1. **SqueezeNet como extractor de características**
   - En `train_sigatoka.py`, dentro de `build_feature_extractor()`, se carga una SqueezeNet preentrenada:
   ```python
   model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
   ```
   - Esta red no decide la clase final. Su trabajo es analizar la imagen y producir un vector de características (embeddings) de baja dimensión.
   - En `extract_features()`, cada imagen pasa por `model.features(inputs)` y luego se aplica `adaptive_avg_pool2d` para convertir la salida en un vector compacto.

2. **SVM como clasificador final**
   - Una vez que SqueezeNet genera los embeddings, se entrena el clasificador:
   ```python
   classifier = Pipeline([
       ("scaler", StandardScaler()),
       ("svm", SVC(kernel="rbf", class_weight="balanced"))
   ])
   classifier.fit(X_train, y_train)
   ```
   - El SVM recibe los vectores de características y aprende a separar las clases: sano, fase inicial, fase media, fase final y no hoja.

### Orden real de ejecución

El flujo del entrenamiento es:

```text
Imágenes del dataset
        ↓
SqueezeNet extrae embeddings
        ↓
StandardScaler normaliza los embeddings
        ↓
SVM entrena con esos vectores
        ↓
Se evalúa en validación y prueba
        ↓
Se guarda el modelo final en sigatoka_svm.joblib
```

### ¿Qué hace cada componente?

- **SqueezeNet:** detecta patrones visuales complejos (textura, manchas, color, forma) y los resume en un vector representativo.
- **SVM:** toma ese vector y define el límite de decisión entre las clases.

### En la app

Cuando se usa la aplicación web (`app_sigatoka.py`), el flujo es el mismo pero en inferencia:

```text
Imagen subida por el usuario
        ↓
SqueezeNet extrae features
        ↓
SVM predice la clase
        ↓
Se muestra el diagnóstico final
```

Esto hace que el sistema sea más eficiente que entrenar una red completa desde cero, y además conserva la capacidad de usar un clasificador robusto sobre descriptores profundos.

---

## 🧭 Plan de mejora paso a paso (con cantidades)

### Paso 1: Crear la clase `No_hoja`
1. Crea la carpeta `dataset_sigatoka/4_No_hoja/`.
2. Descarga automática con icrawler:

```bash
python download_no_hoja.py --total-per-category 100
```

**Resultado esperado (exacto):** 4 categorías x 100 imágenes = **400 imágenes** en total.
Cada categoría se distribuye entre 4 palabras clave (25 imágenes por palabra clave).

Carpetas creadas:

```
dataset_sigatoka/4_No_hoja/
	animales/ (gato, perro, vaca, caballo)
	objetos/ (martillo, silla, telefono, bicicleta)
	fondos/ (suelo, cielo, pared, carretera)
	otras_plantas/ (hoja_de_mango, hoja_de_cafe, hoja_de_palma, hoja_de_papaya)
```

> Nota: revisa y elimina imágenes corruptas o irrelevantes. La calidad del dataset afecta directamente la precisión.

### Paso 2: Limpieza de dataset (corruptos y duplicados)
Primero revisa sin modificar archivos:

```bash
python clean_dataset.py --root dataset_sigatoka
```

Si quieres aplicar cambios, usa cuarentena (recomendado):

```bash
python clean_dataset.py --root dataset_sigatoka --action quarantine
```

Los archivos se moverán a `dataset_sigatoka/_quarantine`.

### Paso 3: Verificar conteo
```bash
python - <<'PY'
import os

root = "dataset_sigatoka"
class_counts = {}
no_hoja_counts = {}

for class_name in sorted(os.listdir(root)):
	class_path = os.path.join(root, class_name)
	if not os.path.isdir(class_path):
		continue
	count = 0
	for dirpath, _, filenames in os.walk(class_path):
		for name in filenames:
			if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
				count += 1
	class_counts[class_name] = count

	if "no" in class_name.lower():
		sub_counts = {}
		for sub in sorted(os.listdir(class_path)):
			sub_path = os.path.join(class_path, sub)
			if not os.path.isdir(sub_path):
				continue
			sub_count = 0
			for dirpath, _, filenames in os.walk(sub_path):
				for name in filenames:
					if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
						sub_count += 1
			sub_counts[sub] = sub_count
		no_hoja_counts = sub_counts

print("Clases y conteos:")
for k, v in class_counts.items():
	print(f"- {k}: {v}")

if no_hoja_counts:
	print("\nNo_hoja subcarpetas:")
	for k, v in no_hoja_counts.items():
		print(f"- {k}: {v}")
PY
```

Si alguna categoria de `No_hoja` queda por debajo de 100, vuelve a ejecutar:

```bash
python download_no_hoja.py --total-per-category 7 --categories fondos otras_plantas
```

### Paso 4: Balancear las clases (cantidad exacta)
Objetivo final: **400 imágenes por clase**.

*   0_Sano: 400
*   1_Fase_Inicial: 400
*   2_Fase_Media: 400
*   3_Fase_Final: 400
*   4_No_hoja: 400

Si no tienes 400 por clase, usa **300 por clase en todas** (sin excepción) para mantener el balance.

Para aplicar el balanceo automatico (mueve sobrantes a respaldo):

```bash
python balance_dataset.py --root dataset_sigatoka --target 300 --action apply
```

### Paso 5: Reentrenar el modelo
```bash
python train_sigatoka.py
```

### Paso 6: Recargar el modelo en la app
Si la app estaba abierta, haz clic en **Recargar modelo** o reinicia Streamlit para cargar el nuevo `sigatoka_svm.joblib`.

### Paso 7: Evaluar negativos (No_hoja)
Para una evaluacion rapida:

```bash
python evaluate_no_hoja.py --neg-dir dataset_sigatoka/4_No_hoja
```

1. Crea una carpeta de prueba con **60 imágenes** que no sean hojas de banano.
2. Ejecuta la app y verifica que no emita diagnóstico en esas imágenes.

Si aún da falsos positivos, sube el `ood_percentile` en `train_sigatoka.py` a 10, reentrena y prueba. Si persiste, sube a 15.

---

## 🚀 ¿Cómo iniciar y probar el proyecto?

Para ver la aplicación funcionando correctamente, **es obligatorio seguir un orden estricto**. No puedes abrir la interfaz web sin antes haber creado y entrenado el "cerebro" (el modelo).

### Paso 1: Crear el Cerebro (Ejecutar Primero)

Lo primero que debes hacer es entrenar tu modelo de Inteligencia Artificial. Abre tu terminal, asegúrate de estar en la carpeta del proyecto y ejecuta el siguiente comando:

```bash
python train_sigatoka.py
```
> **Nota:** Este proceso requiere que la IA analice cientos de imágenes. Puede tardar varios minutos u horas dependiendo del poder de procesamiento de tu computadora. Al finalizar, el script guardará automáticamente el modelo entrenado en `sigatoka_svm.joblib`.

### Paso 2: Iniciar la Interfaz Visual (Ejecutar Después)

**Solo cuando el entrenamiento haya terminado con éxito**, podrás encender la interfaz gráfica para usar tu modelo. En la misma terminal escribe:

```bash
streamlit run app_sigatoka.py
```
> **Resultado:** Este comando levantará un servidor local y abrirá automáticamente una pestaña en tu navegador web de internet (usualmente en la dirección `http://localhost:8501`). Allí verás el título de tu aplicación, el botón para subir imágenes y podrás probar el detector en vivo.

---

## ✅ Filtro anti-falsos positivos (Fuera de Dominio)

Este modelo solo conoce hojas de banano. Si subes un objeto diferente (por ejemplo, un animal o un paisaje), el clasificador puede forzar una fase de Sigatoka que no corresponde. Para reducir ese error se agregó un filtro **OOD (Out-Of-Distribution)** basado en similitud de embeddings:

*   Durante el entrenamiento se calcula un **centroide por clase** y un **umbral de similitud** (percentil 5 del set de validación).
*   En la app, si la **similitud máxima** con los centroides es menor al umbral, se muestra una **advertencia de baja confianza**.
*   Si el modelo predice `No_hoja`, se verifica un **umbral por clase**. Si es bajo, se reporta la mejor fase con baja confianza; si es alto, **no se emite diagnóstico**.

> **Importante:** después de actualizar el código, debes reentrenar para generar un nuevo `sigatoka_svm.joblib` con los campos `centroids` y `ood_threshold`.

---

## 🎯 Consejos para mejorar la precisión

*   Agrega **datos negativos** (otras plantas, suelo, frutos, animales) o crea una clase explícita `No_hoja`.
*   Aumenta la diversidad de iluminación, ángulos y distancias en el dataset.
*   Balancea las clases para evitar sesgos hacia una fase específica.
*   Ajusta el `ood_percentile` en `train_sigatoka.py`: un valor más alto hace el filtro más estricto (menos falsos positivos), pero puede rechazar hojas válidas.
