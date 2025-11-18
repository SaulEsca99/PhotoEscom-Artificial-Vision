# 📸 PhotoEscom - Editor de Fotos para Visión Artificial

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/Librerías-Tkinter%20%7C%20Pillow%20%7C%20NumPy%20%7C%20SciPy%20%7C%20OpenCV%20%7C%20scikit--learn-green.svg)](https://pypi.org/)

**PhotoEscom** es un editor de imágenes avanzado construido con **Python** y **Tkinter**, diseñado como proyecto para el curso de **Visión Artificial**.

Además de las herramientas típicas de edición (rotar, filtros, brillo, zoom, historial), incluye un conjunto completo de **métodos de visión artificial**:

- Detección de bordes (gradiente, Sobel, Prewitt, Roberts, Kirsch, Robinson, Frei‑Chen, Canny, LoG).
- Umbralización de Otsu (manual y OpenCV).
- Detección de esquinas de Harris.
- Esqueletonización.
- Análisis de perímetro de objetos.
- Segmentación de imágenes (threshold, K‑means, watershed, region growing).
- Template matching (manual, OpenCV y multi‑escala).

---

## 🖼️ Vistazo a la Interfaz

![Demo de PhotoEscom](demo.png)

La interfaz se organiza en:

- **Barra superior**: Cargar, guardar, deshacer/rehacer, restaurar y control de zoom (Zoom In, Zoom Out, Ajustar).
- **Panel izquierdo (pestañas)**:
  - Herramientas básicas
  - Transformar
  - Ajustes
  - Filtros
  - Detección de Bordes
  - Otsu
  - Harris
  - Esqueleto
  - Perímetro
  - Segmentación
  - Template
- **Panel central**: Lienzo con scroll y zoom para visualizar la imagen procesada.

---

## ✨ Características Principales

### 1. Edición Básica

- **Cargar y Guardar** imágenes en múltiples formatos (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`).
- **Historial de cambios** con:
  - Deshacer (Undo)
  - Rehacer (Redo)
  - Restaurar a la imagen original
- **Zoom**:
  - Zoom In / Zoom Out
  - Ajustar al lienzo (auto-fit)
  - Scroll vertical y horizontal sobre la imagen.
- **Transformaciones**:
  - Rotación rápida: ±90°
  - Rotación libre con slider
  - Volteo horizontal y vertical
- **Filtros (sobre PIL)**:
  - Original
  - Escala de grises
  - Sepia
  - Invertir colores
  - Desenfoque
  - Detalle
- **Ajustes de Imagen**:
  - Brillo
  - Contraste
  - Saturación
  - Nitidez (sharpen)

---

## 🧠 Módulo de Visión Artificial

### 2. Detección de Bordes

Implementación fiel a los operadores vistos en clase (1ª y 2ª derivada, brújula, base vectorial y Canny).

#### 2.1 Operadores de Primera Derivada

- **Gradiente Básico** (diferencias finitas centradas).
- **Sobel**:
  - Máscara clásica 3×3.
  - Versión extendida (Sobel extendido) para tamaños 5×5 y 7×7.
- **Prewitt** 3×3.
- **Roberts**:
  - Forma: \(\sqrt{D_1^2 + D_2^2}\)
  - Forma: \(|D_1| + |D_2|\)

#### 2.2 Operadores de Brújula (Compass)

- **Kirsch** (8 direcciones).
- **Robinson** (8 direcciones).

#### 2.3 Operador de Base Vectorial

- **Frei‑Chen** usando el subespacio de 9 máscaras:
  - Cálculo de proyecciones.
  - Cálculo de magnitud y relación \(M/S\).

#### 2.4 Operadores de Segunda Derivada

- **Laplaciano de la Gaussiana (LoG)**:
  - Suavizado Gaussiano con parámetro \(\sigma\).
  - Cálculo del laplaciano.
  - Detección de cruce por cero (zero‑crossings).

#### 2.5 Operador de Canny (Óptimo)

Implementación completa del **algoritmo de Canny**:

- Suavizado Gaussiano con \(\sigma\) configurable.
- Cálculo de gradiente (Sobel).
- **Supresión no máxima**.
- **Histéresis de doble umbral** con:
  - Umbral bajo
  - Umbral alto

En la pestaña de Detección de Bordes puedes ajustar:

- Umbral T (para binarizar la magnitud de gradiente).
- \(\sigma\) del suavizado Gaussiano.
- Modo de visualización:
  - Magnitud de gradiente.
  - Ángulo.
  - Borde binario (por umbral).

Incluye botón de **Vista Previa** y de **Aplicar**.

---

### 3. Umbralización de Otsu

Pestaña **Otsu**:

- **Otsu Manual**:
  - Implementación propia del cálculo del umbral óptimo.
  - Muestra:
    - Umbral encontrado.
    - Varianza máxima entre clases.
- **Otsu OpenCV**:
  - Uso de `cv2.threshold` con flag `THRESH_OTSU`.
  - Muestra el umbral encontrado por OpenCV.

Ideal para **binarización automática** de imágenes (separación objeto/fondo).

---

### 4. Detección de esquinas (Harris)

Pestaña **Harris**:

- **Métodos disponibles**:
  - Implementación **manual**.
  - Implementación con **OpenCV**.
- Permite visualizar:
  - Esquinas marcadas sobre la imagen.
  - Conteo total de esquinas detectadas.

Parámetros típicos del método Harris (como \(k\) y umbral) se pueden ajustar internamente.

---

### 5. Esqueletonización

Pestaña **Esqueleto**:

- **Métodos soportados**:
  - Esqueleto morfológico con OpenCV.
  - Esqueleto morfológico manual.
  - **Zhang–Suen** (adelgazamiento iterativo clásico).
- Muestra:
  - Imagen reducida a un esqueleto de 1 píxel de grosor.
  - Número de iteraciones (cuando aplica).

Útil para análisis estructural de objetos binarizados.

---

### 6. Análisis de Perímetro de Objetos

Pestaña **Perímetro**:

- **Métodos**:
  - OpenCV (contornos).
  - **Chain Code**.
  - Métodos **morfológicos**.
- Calcula y muestra:
  - Número de objetos detectados.
  - Perímetro y área (para los primeros objetos).
  - Otros datos descriptivos según el método.

---

### 7. Segmentación de Imágenes

Pestaña **Segmentación**:

- **Threshold (Umbralización clásica)**:
  - Modos: binary, binary_inv, truncate, tozero, tozero_inv.
- **K‑means Clustering**:
  - Segmentación en **K regiones** de color.
  - Parámetro: número de clusters.
  - Devuelve información por cluster:
    - Color centro.
    - Número de píxeles.
    - Porcentaje del área.
- **Watershed**:
  - Basado en topografía de la imagen:
    - Otsu + operaciones morfológicas.
    - Etiquetado de marcadores.
  - Visualización:
    - Bordes en rojo.
    - Regiones coloreadas.
    - Mezcla con la imagen original.
- **Region Growing** (interno para ciertos casos):
  - Inicio desde una semilla (por defecto el centro).
  - Umbral de similitud de intensidad.

---

### 8. Template Matching

Pestaña **Template**:

- Permite **cargar una imagen plantilla (template)** y buscarla en la imagen actual.

#### 8.1 Métodos manuales

- Implementación **manual**:
  - SSD (Sum of Squared Differences).
  - NCC (Normalized Cross‑Correlation).
- Devuelve:
  - Ubicación del mejor match.
  - Score de similitud.
  - Imagen con:
    - Rectángulo verde delimitando el template.
    - Centro marcado.
    - Texto con el score.

#### 8.2 Métodos OpenCV

- Template matching con `cv2.matchTemplate`:
  - Métodos: `sqdiff`, `sqdiff_normed`, `ccorr`, `ccorr_normed`, `ccoeff`, `ccoeff_normed`.
  - Por defecto se usa `ccoeff_normed`.
- Soporta también **multi‑escala**:
  - Búsqueda del template a diferentes escalas para encontrar el mejor tamaño/ubicación.

---

## ⚙️ Dependencias

Este proyecto utiliza:

- **Python 3.10+**
- **Tkinter** (interfaz gráfica)
- **Pillow (PIL)** – manipulación y carga de imágenes
- **NumPy** – operaciones numéricas
- **SciPy** – filtros y convoluciones (`ndimage`, `gaussian_filter`, etc.)
- **OpenCV-Python (`cv2`)** – Canny, Otsu, watershed, template matching, etc.
- **scikit-learn** – `KMeans` para segmentación por clustering
- **Otros**:
  - `matplotlib` (opcional, para ejemplos/demos)
  - `networkx`, `nltk`, etc. pueden estar presentes pero no son obligatorias para ejecutar la GUI básica.

### Instalación con `pip`

Puedes instalar las dependencias principales con:


Si el repositorio incluye un archivo `requirements.txt`, también puedes usar:

---

## 🚀 Cómo Ejecutar

1. Clona o descarga este repositorio:

   ```bash
   git clone https://tu-repositorio.git
   cd PhotoEscom-Artificial-Vision
   ```

2. Asegúrate de tener todas las dependencias instaladas (ver sección anterior).

3. Ejecuta la aplicación principal:

   ```bash
   python Photoescom.py
   ```

4. Se abrirá la ventana de **PhotoEscom**.  
   - Usa el botón **📁 Cargar** para abrir una imagen.
   - Navega por las pestañas del panel izquierdo para aplicar las diferentes técnicas de edición y visión artificial.
   - Usa **Deshacer/Rehacer** para explorar diferentes combinaciones de procesamiento.

---

## 📂 Archivos Relevantes del Proyecto

- `Photoescom.py` – Aplicación principal (GUI con Tkinter).
- `vision_methods.py` – Métodos de visión (Otsu, Harris, etc.).
- `skeleton_perimeter.py` – Esqueletonización y análisis de perímetro.
- `segmentation_template.py` – Métodos de segmentación y template matching.
- `demo_examples.py` – Ejemplos de uso de los métodos.
- `integration_guide.py` – Guía para integrar los métodos en otros proyectos.
- `demo.png` – Captura de la interfaz.
- `requirements.txt` – Lista de dependencias.

---

## 🧪 Uso como Biblioteca (Integración)

Además de la GUI, los módulos de visión (`vision_methods.py`, `skeleton_perimeter.py`, `segmentation_template.py`, etc.) están organizados para poder ser reutilizados desde otros scripts de Python, permitiendo:

- Integrar **Otsu**, **Harris**, **segmentación** y **template matching** en pipelines propios.
- Probar métodos sin abrir la interfaz, usando directamente funciones de los módulos.

Para detalles, consulta `integration_guide.py`.

---
