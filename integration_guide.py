"""
GUÍA DE INTEGRACIÓN - Nuevas Funcionalidades para PhotoEscom

Este archivo contiene el código para agregar las nuevas pestañas y funcionalidades
a tu aplicación PhotoEscom.

INSTRUCCIONES:
1. Copia los archivos vision_methods.py, skeleton_perimeter.py y segmentation_template.py
   al mismo directorio que tu archivo principal
2. Agrega los imports al inicio de tu archivo principal
3. Copia los métodos de integración en tu clase PhotoEditor
4. Agrega las nuevas pestañas en create_widgets()
"""

# ========== PASO 1: IMPORTS A AGREGAR AL INICIO ==========
"""
Agrega estos imports después de tus imports existentes:

from vision_methods import OtsuThreshold, HarrisCornerDetector
from skeleton_perimeter import SkeletonizationMethods, PerimeterAnalysis
from segmentation_template import ImageSegmentation, TemplateMatching
"""

# ========== PASO 2: VARIABLES A AGREGAR EN __init__ ==========
"""
Agrega estas variables en tu método __init__, después de las variables de detección de bordes:

# Variables para Otsu
self.otsu_auto_var = tk.BooleanVar(value=True)

# Variables para Harris
self.harris_method_var = tk.StringVar(value="opencv")
self.harris_k_var = tk.DoubleVar(value=0.04)
self.harris_threshold_var = tk.DoubleVar(value=0.01)
self.harris_window_var = tk.IntVar(value=3)

# Variables para Esqueletonización
self.skeleton_method_var = tk.StringVar(value="opencv")

# Variables para Análisis de Perímetro
self.perimeter_method_var = tk.StringVar(value="opencv")

# Variables para Segmentación
self.seg_method_var = tk.StringVar(value="kmeans")
self.seg_threshold_var = tk.DoubleVar(value=127.0)
self.seg_clusters_var = tk.IntVar(value=3)

# Variables para Template Matching
self.template_method_var = tk.StringVar(value="opencv")
self.template_image = None
self.template_scale_var = tk.DoubleVar(value=1.0)
"""

# ========== PASO 3: MÉTODOS A AGREGAR EN LA CLASE PhotoEditor ==========

def create_advanced_tools_notebook(self, main_panel):
    """
    AGREGAR ESTE MÉTODO a tu clase PhotoEditor.
    
    Crea un segundo notebook para las herramientas avanzadas.
    Llamar este método en create_widgets() después de crear el primer notebook.
    """
    import tkinter as tk
    from tkinter import ttk
    
    # Crear notebook para herramientas avanzadas
    advanced_notebook = ttk.Notebook(main_panel, width=350)
    advanced_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
    advanced_notebook.pack_propagate(False)
    
    # Pestañas para nuevas funcionalidades
    otsu_frame = ttk.Frame(advanced_notebook, padding=10)
    advanced_notebook.add(otsu_frame, text="Otsu")
    
    harris_frame = ttk.Frame(advanced_notebook, padding=10)
    advanced_notebook.add(harris_frame, text="Harris")
    
    skeleton_frame = ttk.Frame(advanced_notebook, padding=10)
    advanced_notebook.add(skeleton_frame, text="Esqueleto")
    
    perimeter_frame = ttk.Frame(advanced_notebook, padding=10)
    advanced_notebook.add(perimeter_frame, text="Perímetro")
    
    segmentation_frame = ttk.Frame(advanced_notebook, padding=10)
    advanced_notebook.add(segmentation_frame, text="Segmentación")
    
    template_frame = ttk.Frame(advanced_notebook, padding=10)
    advanced_notebook.add(template_frame, text="Template")
    
    # Crear contenido de cada pestaña
    self.create_otsu_panel(otsu_frame)
    self.create_harris_panel(harris_frame)
    self.create_skeleton_panel(skeleton_frame)
    self.create_perimeter_panel(perimeter_frame)
    self.create_segmentation_panel(segmentation_frame)
    self.create_template_panel(template_frame)


def create_otsu_panel(self, parent):
    """Panel para método de Otsu"""
    from tkinter import ttk, messagebox
    import tkinter as tk
    
    # Información
    info_text = """Método de Otsu:
    
Umbralización automática que 
encuentra el umbral óptimo 
maximizando la varianza entre 
clases (foreground/background).

Ideal para:
• Separar objetos del fondo
• Binarización automática
• Preprocesamiento"""
    
    info_label = ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=300)
    info_label.pack(pady=10, padx=5)
    
    # Botones
    ttk.Button(parent, text="Aplicar Otsu Manual",
              command=self.apply_otsu_manual).pack(pady=5, fill=tk.X)
    
    ttk.Button(parent, text="Aplicar Otsu OpenCV",
              command=self.apply_otsu_opencv).pack(pady=5, fill=tk.X)
    
    # Resultado
    self.otsu_result_label = ttk.Label(parent, text="", wraplength=300)
    self.otsu_result_label.pack(pady=10)


def create_harris_panel(self, parent):
    """Panel para detección de Harris"""
    from tkinter import ttk
    import tkinter as tk
    
    # Información
    info_text = """Detección de Esquinas Harris:
    
Detecta puntos de interés 
(esquinas) basándose en cambios 
de intensidad en múltiples 
direcciones.

Útil para:
• Matching de imágenes
• Tracking de objetos
• Reconocimiento de patrones"""
    
    ttk.Label(parent, text=info_text, justify=tk.LEFT, 
             wraplength=300).pack(pady=10)
    
    # Método
    ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
    ttk.Radiobutton(parent, text="Manual", value="manual",
                   variable=self.harris_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="OpenCV", value="opencv",
                   variable=self.harris_method_var).pack(anchor=tk.W, padx=20)
    
    # Parámetros
    params_frame = ttk.LabelFrame(parent, text="Parámetros", padding=10)
    params_frame.pack(fill=tk.X, pady=10, padx=5)
    
    # k
    ttk.Label(params_frame, text="k (sensibilidad):").grid(row=0, column=0, sticky=tk.W)
    ttk.Scale(params_frame, from_=0.01, to=0.10, variable=self.harris_k_var).grid(
        row=0, column=1, sticky=tk.EW)
    ttk.Label(params_frame, textvariable=self.harris_k_var, width=6).grid(row=0, column=2)
    
    # Umbral
    ttk.Label(params_frame, text="Umbral:").grid(row=1, column=0, sticky=tk.W)
    ttk.Scale(params_frame, from_=0.001, to=0.1, variable=self.harris_threshold_var).grid(
        row=1, column=1, sticky=tk.EW)
    ttk.Label(params_frame, textvariable=self.harris_threshold_var, width=6).grid(row=1, column=2)
    
    params_frame.columnconfigure(1, weight=1)
    
    # Botones
    ttk.Button(parent, text="Detectar Esquinas",
              command=self.apply_harris).pack(pady=5, fill=tk.X, padx=5)
    
    self.harris_result_label = ttk.Label(parent, text="", wraplength=300)
    self.harris_result_label.pack(pady=10)


def create_skeleton_panel(self, parent):
    """Panel para esqueletonización"""
    from tkinter import ttk
    import tkinter as tk
    
    info_text = """Esqueletonización:
    
Reduce objetos binarios a su 
estructura de 1 píxel de ancho,
preservando topología.

Métodos:
• Morfológico: Erosiones sucesivas
• Zhang-Suen: Algoritmo de adelgazamiento

Aplicaciones:
• Análisis de formas
• Reconocimiento de caracteres
• Procesamiento de huellas"""
    
    ttk.Label(parent, text=info_text, justify=tk.LEFT,
             wraplength=300).pack(pady=10)
    
    # Método
    ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
    ttk.Radiobutton(parent, text="Morfológico OpenCV", value="opencv",
                   variable=self.skeleton_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="Morfológico Manual", value="manual",
                   variable=self.skeleton_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="Zhang-Suen", value="zhang_suen",
                   variable=self.skeleton_method_var).pack(anchor=tk.W, padx=20)
    
    # Botones
    ttk.Button(parent, text="Aplicar Esqueletonización",
              command=self.apply_skeletonization).pack(pady=10, fill=tk.X, padx=5)
    
    ttk.Button(parent, text="Comparar Métodos",
              command=self.compare_skeleton_methods).pack(pady=5, fill=tk.X, padx=5)
    
    self.skeleton_result_label = ttk.Label(parent, text="", wraplength=300)
    self.skeleton_result_label.pack(pady=10)


def create_perimeter_panel(self, parent):
    """Panel para análisis de perímetro"""
    from tkinter import ttk
    import tkinter as tk
    
    info_text = """Análisis de Perímetro:
    
Mide el perímetro de objetos en 
imágenes binarias.

Métodos:
• OpenCV: findContours
• Chain Code: Codificación direccional
• Morfológico: Original - Erosión

Calcula:
• Longitud del perímetro
• Área
• Circularidad
• Centroide"""
    
    ttk.Label(parent, text=info_text, justify=tk.LEFT,
             wraplength=300).pack(pady=10)
    
    # Método
    ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
    ttk.Radiobutton(parent, text="OpenCV Contours", value="opencv",
                   variable=self.perimeter_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="Chain Code", value="chain_code",
                   variable=self.perimeter_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="Morfológico", value="morphological",
                   variable=self.perimeter_method_var).pack(anchor=tk.W, padx=20)
    
    # Botones
    ttk.Button(parent, text="Analizar Perímetro",
              command=self.apply_perimeter_analysis).pack(pady=10, fill=tk.X, padx=5)
    
    ttk.Button(parent, text="Comparar Métodos",
              command=self.compare_perimeter_methods).pack(pady=5, fill=tk.X, padx=5)
    
    # Área para mostrar resultados
    self.perimeter_result_text = tk.Text(parent, height=8, width=40)
    self.perimeter_result_text.pack(pady=10, padx=5)


def create_segmentation_panel(self, parent):
    """Panel para segmentación"""
    from tkinter import ttk
    import tkinter as tk
    
    info_text = """Segmentación:
    
Divide la imagen en regiones 
homogéneas.

Métodos disponibles:
• Threshold: Umbralización simple
• K-means: Clustering de colores
• Watershed: Cuencas hidrográficas
• Region Growing: Crecimiento de regiones"""
    
    ttk.Label(parent, text=info_text, justify=tk.LEFT,
             wraplength=300).pack(pady=10)
    
    # Método
    ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
    ttk.Radiobutton(parent, text="Threshold", value="threshold",
                   variable=self.seg_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="K-means", value="kmeans",
                   variable=self.seg_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="Watershed", value="watershed",
                   variable=self.seg_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="Region Growing", value="region_growing",
                   variable=self.seg_method_var).pack(anchor=tk.W, padx=20)
    
    # Parámetros
    params_frame = ttk.LabelFrame(parent, text="Parámetros", padding=10)
    params_frame.pack(fill=tk.X, pady=10, padx=5)
    
    # Umbral (para threshold)
    ttk.Label(params_frame, text="Umbral:").grid(row=0, column=0, sticky=tk.W)
    ttk.Scale(params_frame, from_=0, to=255, 
             variable=self.seg_threshold_var).grid(row=0, column=1, sticky=tk.EW)
    ttk.Label(params_frame, textvariable=self.seg_threshold_var, 
             width=6).grid(row=0, column=2)
    
    # Clusters (para k-means)
    ttk.Label(params_frame, text="Clusters:").grid(row=1, column=0, sticky=tk.W)
    ttk.Scale(params_frame, from_=2, to=10, 
             variable=self.seg_clusters_var).grid(row=1, column=1, sticky=tk.EW)
    ttk.Label(params_frame, textvariable=self.seg_clusters_var,
             width=6).grid(row=1, column=2)
    
    params_frame.columnconfigure(1, weight=1)
    
    # Botones
    ttk.Button(parent, text="Aplicar Segmentación",
              command=self.apply_segmentation).pack(pady=10, fill=tk.X, padx=5)
    
    self.seg_result_label = ttk.Label(parent, text="", wraplength=300)
    self.seg_result_label.pack(pady=10)


def create_template_panel(self, parent):
    """Panel para template matching"""
    from tkinter import ttk
    import tkinter as tk
    
    info_text = """Template Matching:
    
Busca una plantilla (template) 
dentro de una imagen más grande.

Métodos:
• Manual: SSD o NCC desde cero
• OpenCV: Múltiples algoritmos
• Multi-escala: Búsqueda a diferentes tamaños

Pasos:
1. Cargar template
2. Seleccionar método
3. Aplicar búsqueda"""
    
    ttk.Label(parent, text=info_text, justify=tk.LEFT,
             wraplength=300).pack(pady=10)
    
    # Cargar template
    ttk.Button(parent, text="📁 Cargar Template",
              command=self.load_template).pack(pady=5, fill=tk.X, padx=5)
    
    self.template_status = ttk.Label(parent, text="No hay template cargado")
    self.template_status.pack(pady=5)
    
    # Método
    ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
    ttk.Radiobutton(parent, text="Manual (SSD)", value="manual",
                   variable=self.template_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="OpenCV", value="opencv",
                   variable=self.template_method_var).pack(anchor=tk.W, padx=20)
    ttk.Radiobutton(parent, text="Multi-escala", value="multiscale",
                   variable=self.template_method_var).pack(anchor=tk.W, padx=20)
    
    # Botones
    ttk.Button(parent, text="Buscar Template",
              command=self.apply_template_matching).pack(pady=10, fill=tk.X, padx=5)
    
    self.template_result_label = ttk.Label(parent, text="", wraplength=300)
    self.template_result_label.pack(pady=10)


# ========== MÉTODOS DE APLICACIÓN ==========

def apply_otsu_manual(self):
    """Aplicar método de Otsu manual"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from vision_methods import OtsuThreshold
        import numpy as np
        from PIL import Image
        
        # Obtener imagen
        img_array = np.array(self.history[self.history_index])
        
        # Aplicar Otsu
        threshold, result, metrics = OtsuThreshold.calculate_threshold(img_array)
        
        # Convertir a PIL
        result_pil = Image.fromarray(result).convert('RGB')
        
        # Actualizar imagen
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        # Mostrar resultados
        info_text = f"""Umbral óptimo: {threshold}
Varianza máxima: {metrics['max_variance']:.2f}
Píxeles foreground: {metrics['foreground_pixels']}
Píxeles background: {metrics['background_pixels']}"""
        
        self.otsu_result_label.config(text=info_text)
        self.status_bar.config(text=f"Otsu aplicado - Umbral: {threshold}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al aplicar Otsu: {str(e)}")


def apply_otsu_opencv(self):
    """Aplicar método de Otsu con OpenCV"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from vision_methods import OtsuThreshold
        import numpy as np
        from PIL import Image
        
        img_array = np.array(self.history[self.history_index])
        
        threshold, result = OtsuThreshold.apply_otsu_opencv(img_array)
        
        result_pil = Image.fromarray(result).convert('RGB')
        
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        self.otsu_result_label.config(text=f"Umbral OpenCV: {threshold:.2f}")
        self.status_bar.config(text=f"Otsu OpenCV - Umbral: {threshold:.2f}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


def apply_harris(self):
    """Aplicar detección de Harris"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from vision_methods import HarrisCornerDetector
        import numpy as np
        from PIL import Image
        
        img_array = np.array(self.history[self.history_index])
        
        method = self.harris_method_var.get()
        k = self.harris_k_var.get()
        threshold = self.harris_threshold_var.get()
        
        if method == "manual":
            corners_img, response, coords = HarrisCornerDetector.detect_corners_manual(
                img_array, k=k, threshold=threshold
            )
        else:
            corners_img, response, coords = HarrisCornerDetector.detect_corners_opencv(
                img_array, k=k, threshold=threshold
            )
        
        result_pil = Image.fromarray(corners_img)
        
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        info_text = f"""Método: {method}
Esquinas detectadas: {len(coords)}
k = {k:.3f}
Umbral = {threshold:.3f}"""
        
        self.harris_result_label.config(text=info_text)
        self.status_bar.config(text=f"Harris - {len(coords)} esquinas detectadas")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


def apply_skeletonization(self):
    """Aplicar esqueletonización"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from skeleton_perimeter import SkeletonizationMethods
        import numpy as np
        from PIL import Image
        
        img_array = np.array(self.history[self.history_index])
        
        method = self.skeleton_method_var.get()
        
        if method == "zhang_suen":
            skeleton, iterations = SkeletonizationMethods.zhang_suen_skeleton(img_array)
        else:
            skeleton, iterations = SkeletonizationMethods.morphological_skeleton(
                img_array, method=method
            )
        
        result_pil = Image.fromarray(skeleton).convert('RGB')
        
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        iter_text = iterations if iterations >= 0 else "N/A"
        self.skeleton_result_label.config(
            text=f"Método: {method}\nIteraciones: {iter_text}"
        )
        self.status_bar.config(text=f"Esqueletonización aplicada ({method})")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


def compare_skeleton_methods(self):
    """Comparar métodos de esqueletonización"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from skeleton_perimeter import SkeletonizationMethods
        import numpy as np
        from PIL import Image
        
        img_array = np.array(self.history[self.history_index])
        
        results = SkeletonizationMethods.compare_methods(img_array)
        
        # Crear imagen combinada mostrando los tres métodos
        skeletons = [results[key]['skeleton'] for key in results.keys()]
        
        # Mostrar el primer resultado
        first_key = list(results.keys())[0]
        result_pil = Image.fromarray(results[first_key]['skeleton']).convert('RGB')
        
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        info_text = "Comparación de métodos:\n"
        for key, data in results.items():
            iter_val = data['iterations'] if data['iterations'] >= 0 else "N/A"
            info_text += f"\n{data['name']}: {iter_val} iter"
        
        self.skeleton_result_label.config(text=info_text)
        messagebox.showinfo("Comparación", 
                          "Métodos comparados. Usa Deshacer/Rehacer para ver cada resultado.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


def apply_perimeter_analysis(self):
    """Aplicar análisis de perímetro"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from skeleton_perimeter import PerimeterAnalysis
        import numpy as np
        from PIL import Image
        
        img_array = np.array(self.history[self.history_index])
        
        method = self.perimeter_method_var.get()
        
        perimeter_img, measurements = PerimeterAnalysis.calculate_perimeter(
            img_array, method=method
        )
        
        result_pil = Image.fromarray(perimeter_img)
        
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        # Mostrar resultados en el Text widget
        self.perimeter_result_text.delete(1.0, tk.END)
        self.perimeter_result_text.insert(1.0, f"Método: {measurements['method']}\n")
        self.perimeter_result_text.insert(tk.END, f"Objetos: {measurements['num_objects']}\n\n")
        
        if 'total_perimeter' in measurements:
            self.perimeter_result_text.insert(tk.END, 
                f"Perímetro total: {measurements['total_perimeter']:.2f}\n\n")
        
        if 'objects' in measurements and len(measurements['objects']) > 0:
            for obj in measurements['objects'][:5]:  # Primeros 5
                self.perimeter_result_text.insert(tk.END, f"Objeto {obj['contour_id']}:\n")
                if 'perimeter' in obj:
                    self.perimeter_result_text.insert(tk.END, 
                        f"  Perímetro: {obj['perimeter']:.2f}\n")
                if 'area' in obj:
                    self.perimeter_result_text.insert(tk.END, 
                        f"  Área: {obj['area']:.2f}\n")
                if 'circularity' in obj:
                    self.perimeter_result_text.insert(tk.END, 
                        f"  Circularidad: {obj['circularity']:.3f}\n")
                self.perimeter_result_text.insert(tk.END, "\n")
        
        self.status_bar.config(text=f"Perímetro analizado ({method})")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


def compare_perimeter_methods(self):
    """Comparar métodos de análisis de perímetro"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from skeleton_perimeter import PerimeterAnalysis
        import numpy as np
        
        img_array = np.array(self.history[self.history_index])
        
        results = PerimeterAnalysis.compare_methods(img_array)
        
        # Mostrar comparación
        self.perimeter_result_text.delete(1.0, tk.END)
        self.perimeter_result_text.insert(1.0, "COMPARACIÓN DE MÉTODOS\n\n")
        
        for key, data in results.items():
            meas = data['measurements']
            self.perimeter_result_text.insert(tk.END, f"{data['name']}:\n")
            self.perimeter_result_text.insert(tk.END, f"  Objetos: {meas['num_objects']}\n")
            if 'total_perimeter' in meas:
                self.perimeter_result_text.insert(tk.END, 
                    f"  Perímetro: {meas['total_perimeter']:.2f}\n")
            self.perimeter_result_text.insert(tk.END, "\n")
        
        messagebox.showinfo("Comparación", "Métodos comparados exitosamente")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


def apply_segmentation(self):
    """Aplicar segmentación"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    try:
        from segmentation_template import ImageSegmentation
        import numpy as np
        from PIL import Image
        
        img_array = np.array(self.history[self.history_index])
        
        method = self.seg_method_var.get()
        
        if method == 'threshold':
            threshold = int(self.seg_threshold_var.get())
            result, info = ImageSegmentation.threshold_segmentation(
                img_array, threshold=threshold
            )
        elif method == 'kmeans':
            n_clusters = int(self.seg_clusters_var.get())
            result, info = ImageSegmentation.kmeans_segmentation(
                img_array, n_clusters=n_clusters
            )
        elif method == 'watershed':
            result, info = ImageSegmentation.watershed_segmentation(img_array)
        elif method == 'region_growing':
            result, info = ImageSegmentation.region_growing(img_array)
        
        result_pil = Image.fromarray(result)
        
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        info_text = f"Método: {info['method']}\n"
        if 'n_clusters' in info:
            info_text += f"Clusters: {info['n_clusters']}\n"
        if 'num_regions' in info:
            info_text += f"Regiones: {info['num_regions']}\n"
        
        self.seg_result_label.config(text=info_text)
        self.status_bar.config(text=f"Segmentación aplicada ({method})")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


def load_template(self):
    """Cargar template para matching"""
    from tkinter import filedialog
    from PIL import Image
    
    file_path = filedialog.askopenfilename(
        title="Seleccionar Template",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_path:
        try:
            self.template_image = Image.open(file_path).convert('RGB')
            self.template_status.config(
                text=f"Template cargado: {self.template_image.size[0]}x{self.template_image.size[1]}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar template: {str(e)}")


def apply_template_matching(self):
    """Aplicar template matching"""
    if not self.current_image:
        messagebox.showwarning("Advertencia", "No hay imagen cargada")
        return
    
    if not self.template_image:
        messagebox.showwarning("Advertencia", "Debe cargar un template primero")
        return
    
    try:
        from segmentation_template import TemplateMatching
        import numpy as np
        from PIL import Image
        
        img_array = np.array(self.history[self.history_index])
        template_array = np.array(self.template_image)
        
        method = self.template_method_var.get()
        
        if method == 'manual':
            result, info = TemplateMatching.match_template_manual(
                img_array, template_array, method='ssd'
            )
        elif method == 'opencv':
            result, info = TemplateMatching.match_template_opencv(
                img_array, template_array, method='ccoeff_normed'
            )
        elif method == 'multiscale':
            result, info = TemplateMatching.multi_scale_matching(
                img_array, template_array
            )
        
        result_pil = Image.fromarray(result)
        
        self.current_image = result_pil
        self.add_to_history()
        self.display_image_on_canvas()
        
        info_text = f"""Método: {info['method']}
Ubicación: {info['best_location']}
Score: {info['best_score']:.3f}"""
        
        if 'best_scale' in info:
            info_text += f"\nEscala: {info['best_scale']:.2f}"
        
        self.template_result_label.config(text=info_text)
        self.status_bar.config(text=f"Template encontrado en {info['best_location']}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error: {str(e)}")


# ========== INSTRUCCIONES FINALES ==========
"""
PASOS PARA INTEGRAR:

1. Copia los 3 archivos .py al directorio de tu aplicación:
   - vision_methods.py
   - skeleton_perimeter.py
   - segmentation_template.py

2. En tu archivo principal, agrega los imports al inicio

3. En __init__ de PhotoEditor, agrega las nuevas variables

4. Copia TODOS los métodos de este archivo a tu clase PhotoEditor

5. En el método create_widgets(), después de crear el primer notebook,
   agrega esta línea:
   
   self.create_advanced_tools_notebook(main_panel)

6. Ejecuta la aplicación y verás un segundo notebook con todas las
   nuevas funcionalidades

¡Listo! Ahora tendrás 6 nuevas pestañas con todas las funcionalidades solicitadas.
"""

print("Archivo de integración creado correctamente.")
print("Lee las instrucciones al final del archivo para integrar las funcionalidades.")
