# Detector de Sigatoka Negra con Deep Learning 🍌🦠

Este proyecto implementa un sistema de Inteligencia Artificial basado en **PyTorch** y **Streamlit** para la detección y clasificación de la enfermedad de la **Sigatoka Negra** en hojas de banano. 

Utiliza una arquitectura de red neuronal **SqueezeNet** pre-entrenada y modificada. Este modelo fue elegido específicamente por ser liviano y eficiente, ideal para ejecutarse en entornos con pocos recursos computacionales (Edge AI), manteniendo una alta precisión en el diagnóstico.

---

## 📁 Estructura del Código

El sistema está dividido en dos partes principales que cumplen roles muy distintos: el "Cerebro" que aprende y la "Interfaz" que interactúa con el usuario.

### 1. El Entrenamiento (`train_sigatoka.py`)
Este es el motor de aprendizaje profundo del proyecto. Sus responsabilidades incluyen:
*   **Procesamiento de Datos:** Carga las imágenes de las carpetas de entrenamiento, las recorta, ajusta sus colores (Data Augmentation) y las redimensiona al estándar requerido por la red neuronal (227x227 píxeles).
*   **Entrenamiento:** Utiliza el modelo SqueezeNet para aprender a diferenciar las características visuales de 4 categorías distintas:
    *   `0_Sano`
    *   `1_Fase_Inicial`
    *   `2_Fase_Media`
    *   `3_Fase_Final`
*   **Evaluación y Guardado:** Valida qué tan preciso es el modelo y, al terminar, guarda el "cerebro entrenado" en un archivo (generalmente con extensión `.pth`). Este archivo contiene todo el conocimiento adquirido por la IA.

### 2. La Aplicación Web (`app_sigatoka.py`)
Esta es la interfaz gráfica, construida con la librería Streamlit. Está diseñada para ser amigable con un usuario final (como un agricultor o agrónomo). Sus tareas son:
*   **Carga del Conocimiento:** Lee el archivo `.pth` generado en el entrenamiento para activar la red neuronal pre-entrenada.
*   **Interacción:** Muestra una página web donde el usuario puede cargar o arrastrar una fotografía de una hoja de banano.
*   **Diagnóstico:** Toma esa foto subida por el usuario, la procesa rápidamente y la pasa por la Inteligencia Artificial para devolver un diagnóstico instantáneo de la fase de infección.

---

## ⚙️ Instalación de Dependencias

Antes de ejecutar cualquier script, asegúrate de tener instaladas todas las librerías necesarias. En tu terminal, ejecuta:

```bash
pip install -r requirements.txt
```

---

## 🚀 ¿Cómo iniciar y probar el proyecto?

Para ver la aplicación funcionando correctamente, **es obligatorio seguir un orden estricto**. No puedes abrir la interfaz web sin antes haber creado y entrenado el "cerebro" (el modelo).

### Paso 1: Crear el Cerebro (Ejecutar Primero)

Lo primero que debes hacer es entrenar tu modelo de Inteligencia Artificial. Abre tu terminal, asegúrate de estar en la carpeta del proyecto y ejecuta el siguiente comando:

```bash
python train_sigatoka.py
```
> **Nota:** Este proceso requiere que la IA analice cientos de imágenes. Puede tardar varios minutos u horas dependiendo del poder de procesamiento de tu computadora (si tienes tarjeta gráfica o solo procesador). Al finalizar, el script guardará automáticamente el modelo entrenado.

### Paso 2: Iniciar la Interfaz Visual (Ejecutar Después)

**Solo cuando el entrenamiento haya terminado con éxito**, podrás encender la interfaz gráfica para usar tu modelo. En la misma terminal escribe:

```bash
streamlit run app_sigatoka.py
```
> **Resultado:** Este comando levantará un servidor local y abrirá automáticamente una pestaña en tu navegador web de internet (usualmente en la dirección `http://localhost:8501`). Allí verás el título de tu aplicación, el botón para subir imágenes y podrás probar el detector en vivo.
