from inference_sdk import InferenceHTTPClient
from vision_methods import OtsuThreshold, HarrisCornerDetector
from skeleton_perimeter import SkeletonizationMethods, PerimeterAnalysis
from segmentation_template import ImageSegmentation, TemplateMatching
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label, binary_dilation


class PhotoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("PhotoEscom - Editor de Fotos Profesional")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2e2e2e')
        self.root.minsize(1200, 700)

        # Variables para manejar la imagen
        self.original_image = None
        self.current_image = None
        self.display_image = None
        self.history = []
        self.history_index = -1
        self.filename = None
        self.zoom_factor = 1.0

        # Variables para controles básicos
        self.rotate_var = tk.DoubleVar(value=0)
        self.scale_x_var = tk.DoubleVar(value=1.0)
        self.scale_y_var = tk.DoubleVar(value=1.0)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.sharpen_var = tk.DoubleVar(value=1.0)
        self.filter_var = tk.StringVar(value="original")

        # Variables para detección de bordes (basadas en PDF)
        self.edge_operator_var = tk.StringVar(value="sobel")
        self.threshold_var = tk.DoubleVar(value=30.0)
        self.sigma_var = tk.DoubleVar(value=1.0)
        self.canny_low_var = tk.DoubleVar(value=50.0)
        self.canny_high_var = tk.DoubleVar(value=150.0)

        # Variables adicionales del PDF
        self.roberts_form_var = tk.StringVar(value="sqrt")
        self.show_magnitude_var = tk.BooleanVar(value=False)
        self.show_angle_var = tk.BooleanVar(value=False)
        self.extended_size_var = tk.IntVar(value=3)

        # Variables para nuevas funcionalidades
        self.otsu_auto_var = tk.BooleanVar(value=True)
        self.harris_method_var = tk.StringVar(value="opencv")
        self.harris_k_var = tk.DoubleVar(value=0.04)
        self.harris_threshold_var = tk.DoubleVar(value=0.01)
        self.harris_window_var = tk.IntVar(value=3)
        self.skeleton_method_var = tk.StringVar(value="opencv")
        self.perimeter_method_var = tk.StringVar(value="opencv")
        self.seg_method_var = tk.StringVar(value="kmeans")
        self.seg_threshold_var = tk.DoubleVar(value=127.0)
        self.seg_clusters_var = tk.IntVar(value=3)
        self.template_method_var = tk.StringVar(value="opencv")
        self.template_image = None

        # Para reconocimiento
        self.clf = None
        self.scaler = None
        self.conf_threshold_var = tk.DoubleVar(value=0.1)  # Nuevo: umbral ajustable
        self.preprocess_var = tk.BooleanVar(value=True)  # Nuevo: preprocesar imagen

        # Traces para previews
        self.brightness_var.trace('w', self.preview_adjustments)
        self.contrast_var.trace('w', self.preview_adjustments)
        self.saturation_var.trace('w', self.preview_adjustments)
        self.sharpen_var.trace('w', self.preview_adjustments)

        self.rotate_var.trace('w', self.preview_transforms)
        self.scale_x_var.trace('w', self.preview_transforms)
        self.scale_y_var.trace('w', self.preview_transforms)

        # Configurar estilo
        self.setup_styles()

        # Crear interfaz
        self.create_widgets()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configurar colores
        self.bg_color = '#2e2e2e'
        self.frame_bg = '#3c3c3c'
        self.button_bg = '#4a4a4a'
        self.accent_color = '#007acc'
        self.text_color = '#ffffff'
        self.highlight_color = '#4a76cf'

        # Configurar estilos
        self.style.configure('TFrame', background=self.frame_bg)
        self.style.configure('TLabel', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TButton', background=self.button_bg, foreground=self.text_color,
                             borderwidth=1, focuscolor=self.accent_color)
        self.style.configure('TScale', background=self.frame_bg, troughcolor=self.accent_color)
        self.style.configure('TCheckbutton', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TRadiobutton', background=self.frame_bg, foreground=self.text_color)
        self.style.configure('TNotebook', background=self.bg_color)
        self.style.configure('TNotebook.Tab', background=self.button_bg, foreground=self.text_color,
                             padding=[10, 5])
        self.style.map('TNotebook.Tab', background=[('selected', self.accent_color)])
        self.style.map('TButton', background=[('active', self.highlight_color)])

    def create_widgets(self):
        # Barra de herramientas superior
        self.create_top_toolbar()

        # Panel principal
        main_panel = ttk.Frame(self.root)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Panel de herramientas izquierdo (ÚNICO)
        # Ajustamos el ancho a 300 para que no sea tan grande
        tools_notebook = ttk.Notebook(main_panel, width=300)
        tools_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tools_notebook.pack_propagate(False)

        # --- Pestañas originales ---
        basic_tools_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(basic_tools_frame, text="Herramientas")

        transform_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(transform_frame, text="Transformar")

        adjust_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(adjust_frame, text="Ajustes")

        filter_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(filter_frame, text="Filtros")

        edge_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(edge_frame, text="Detección Bordes")

        # --- Pestañas de nuevas funcionalidades (añadidas al mismo notebook) ---
        otsu_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(otsu_frame, text="Otsu")

        harris_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(harris_frame, text="Harris")

        skeleton_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(skeleton_frame, text="Esqueleto")

        perimeter_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(perimeter_frame, text="Perímetro")

        segmentation_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(segmentation_frame, text="Segmentación")

        template_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(template_frame, text="Template")

        recognition_frame = ttk.Frame(tools_notebook, padding=10)
        tools_notebook.add(recognition_frame, text="Reconocimiento")

        # --- Panel de visualización ---
        self.image_frame = ttk.Frame(main_panel)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Crear lienzo
        self.create_image_canvas()

        # --- Llenar todas las pestañas ---
        # Originales
        self.create_basic_tools_panel(basic_tools_frame)
        self.create_transform_panel(transform_frame)
        self.create_adjustments_panel(adjust_frame)
        self.create_filters_panel(filter_frame)
        self.create_edge_detection_panel(edge_frame)

        # Nuevas
        self.create_otsu_panel(otsu_frame)
        self.create_harris_panel(harris_frame)
        self.create_skeleton_panel(skeleton_frame)
        self.create_perimeter_panel(perimeter_frame)
        self.create_segmentation_panel(segmentation_frame)
        self.create_template_panel(template_frame)
        self.create_recognition_panel(recognition_frame)

        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="PhotoEscom - Listo. Cargue una imagen para comenzar")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_recognition_panel(self, parent):
        """Panel para reconocimiento de billetes"""
        info_text = """Reconocimiento de Billetes:

Usa IA para identificar billetes mexicanos."""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=250).pack(pady=10)

        # Umbral de confianza ajustable
        ttk.Label(parent, text="Umbral de Confianza:").pack(anchor=tk.W, padx=5)
        ttk.Scale(parent, from_=0.0, to=1.0, variable=self.conf_threshold_var).pack(fill=tk.X, padx=5)
        self.conf_label = ttk.Label(parent, text=f"{self.conf_threshold_var.get():.2f}")
        self.conf_label.pack(pady=5)
        self.conf_threshold_var.trace('w', self.update_conf_label)

        # Checkbox para preprocesar
        ttk.Checkbutton(parent, text="Preprocesar imagen (mejorar contraste)", variable=self.preprocess_var).pack(anchor=tk.W, pady=5)

        ttk.Button(parent, text="Reconocer Billete",
                   command=self.apply_recognition).pack(pady=10, fill=tk.X, padx=5)

        self.recognition_result_label = ttk.Label(parent, text="", wraplength=250)
        self.recognition_result_label.pack(pady=10)

    def update_conf_label(self, *args):
        self.conf_label.config(text=f"{self.conf_threshold_var.get():.2f}")

    def apply_recognition(self):
        """Aplicar reconocimiento de billetes MEJORADO"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        try:
            import pickle
            from inference_sdk import InferenceHTTPClient

            # Cargar modelo entrenado (opcional)
            model_path = 'models/bill_recognizer.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print(f"✓ Modelo SVM cargado (Test Acc: {model_data.get('test_accuracy', 0):.2%})")
            else:
                print("⚠️  Modelo SVM no encontrado, usando solo Roboflow")

            # Guardar imagen temporal
            temp_path = os.path.abspath("temp_recognition.jpg")

            # Asegurarse de guardar la imagen correctamente
            try:
                # Si la imagen actual es muy grande, redimensionar
                img_to_save = self.current_image.copy()
                max_size = 1920
                if max(img_to_save.size) > max_size:
                    ratio = max_size / max(img_to_save.size)
                    new_size = (int(img_to_save.size[0] * ratio), int(img_to_save.size[1] * ratio))
                    img_to_save = img_to_save.resize(new_size, Image.Resampling.LANCZOS)
                    print(f"✓ Imagen redimensionada a {new_size}")

                img_to_save.save(temp_path, quality=95)
                print(f"✓ Imagen guardada en: {temp_path}")

                # Verificar que se guardó
                if not os.path.exists(temp_path):
                    raise Exception("No se pudo guardar la imagen temporal")

                file_size = os.path.getsize(temp_path) / 1024
                print(f"✓ Tamaño: {file_size:.2f} KB")

            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar imagen: {e}")
                return

            # Preprocesado opcional
            if self.preprocess_var.get():
                img_cv = cv2.imread(temp_path)
                lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                cv2.imwrite(temp_path, img_cv)
                print("✓ Preprocesamiento aplicado")

            # Inferencia con Roboflow
            try:
                CLIENT = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key="tkhi9LqnfsZxo8AK0ULH"
                )

                # Usar versión 1 que funciona mejor
                print("🔍 Conectando con Roboflow API...")
                result = CLIENT.infer(temp_path, model_id="billetes-mexicanos-9s5an/1")
                print(f"✓ Respuesta recibida en {result.get('time', 0):.3f}s")

            except Exception as e:
                messagebox.showerror("Error", f"Error con Roboflow API:\n{str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return

            print("\n" + "=" * 60)
            print("🔍 RESULTADOS DE DETECCIÓN")
            print("=" * 60)

            # Procesar resultados
            img = cv2.imread(temp_path)
            if img is None:
                messagebox.showerror("Error", "No se pudo leer la imagen temporal")
                os.remove(temp_path)
                return

            h_img, w_img = img.shape[:2]
            detections = []

            if 'predictions' in result and result['predictions']:
                predictions = result['predictions']
                conf_threshold = self.conf_threshold_var.get()

                print(f"Predicciones totales: {len(predictions)}")
                print(f"Umbral de confianza: {conf_threshold:.0%}")

                for i, pred in enumerate(predictions, 1):
                    confidence = pred['confidence']
                    class_name = pred['class']

                    print(f"\nPredicción #{i}:")
                    print(f"  Clase: ${class_name}")
                    print(f"  Confianza: {confidence:.2%}")

                    if confidence >= conf_threshold:
                        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                        x1 = max(0, int(x - w / 2))
                        y1 = max(0, int(y - h / 2))
                        x2 = min(w_img, int(x + w / 2))
                        y2 = min(h_img, int(y + h / 2))

                        # Color según confianza
                        if confidence >= 0.7:
                            color = (0, 255, 0)  # Verde
                            label_bg = (0, 180, 0)
                        elif confidence >= 0.5:
                            color = (0, 255, 255)  # Amarillo
                            label_bg = (0, 180, 180)
                        else:
                            color = (0, 165, 255)  # Naranja
                            label_bg = (0, 120, 200)

                        # Dibujar rectángulo
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

                        # Preparar texto
                        label = f"${class_name} ({confidence:.0%})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.9
                        thickness = 2

                        # Calcular tamaño del texto
                        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                        # Dibujar fondo del texto
                        cv2.rectangle(img,
                                      (x1, y1 - th - baseline - 12),
                                      (x1 + tw + 12, y1),
                                      label_bg, -1)

                        # Dibujar borde del texto
                        cv2.rectangle(img,
                                      (x1, y1 - th - baseline - 12),
                                      (x1 + tw + 12, y1),
                                      color, 2)

                        # Dibujar texto
                        cv2.putText(img, label,
                                    (x1 + 6, y1 - baseline - 6),
                                    font, font_scale, (255, 255, 255), thickness)

                        detections.append(f"${class_name} ({confidence:.0%})")
                        print(f"  ✓ DETECTADO")
                    else:
                        print(f"  ✗ Rechazado (< {conf_threshold:.0%})")

                # Actualizar interfaz
                if detections:
                    result_text = f"✓ Detectados:\n\n" + "\n".join(detections)
                    self.recognition_result_label.config(text=result_text)
                    self.status_bar.config(text=f"✓ {len(detections)} billetes detectados")
                    print(f"\n✅ Total detectados: {len(detections)}")
                else:
                    # Mostrar las mejores predicciones aunque no pasen el umbral
                    top_preds = sorted(predictions, key=lambda p: p['confidence'], reverse=True)[:3]
                    result_text = f"⚠️ Sin detecciones\n(umbral={conf_threshold:.0%})\n\n"
                    result_text += "Mejores predicciones:\n"
                    for p in top_preds:
                        result_text += f"${p['class']}: {p['confidence']:.0%}\n"

                    self.recognition_result_label.config(text=result_text)
                    self.status_bar.config(text="Baja el umbral para ver más")
                    print(f"\n⚠️ Ninguna predicción supera {conf_threshold:.0%}")

                # Convertir y mostrar imagen con detecciones
                result_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self.current_image = result_pil
                self.add_to_history()
                self.display_image_on_canvas()

            else:
                result_text = "❌ No hay predicciones\n\nVerifica:\n• La imagen tiene billetes\n• Los billetes son visibles\n• Intenta preprocesar"
                self.recognition_result_label.config(text=result_text)
                self.status_bar.config(text="Sin predicciones")
                print("❌ La API no devolvió predicciones")

            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"✓ Archivo temporal eliminado")

            print("=" * 60 + "\n")

        except Exception as e:
            error_msg = f"Error en reconocimiento:\n{str(e)}"
            messagebox.showerror("Error", error_msg)
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

            # Limpiar
            if os.path.exists("temp_recognition.jpg"):
                os.remove("temp_recognition.jpg")

    def create_otsu_panel(self, parent):
        """Panel para método de Otsu"""
        info_text = """Método de Otsu:

Umbralización automática que 
encuentra el umbral óptimo.

Ideal para separar objetos 
del fondo."""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=250).pack(pady=10, padx=5)

        ttk.Button(parent, text="Aplicar Otsu Manual",
                   command=self.apply_otsu_manual).pack(pady=5, fill=tk.X, padx=5)

        ttk.Button(parent, text="Aplicar Otsu OpenCV",
                   command=self.apply_otsu_opencv).pack(pady=5, fill=tk.X, padx=5)

        self.otsu_result_label = ttk.Label(parent, text="", wraplength=250)
        self.otsu_result_label.pack(pady=10)

    def create_harris_panel(self, parent):
        """Panel para detección de Harris"""
        info_text = """Detección de Harris:

Detecta esquinas y puntos
de interés en la imagen."""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=250).pack(pady=10)

        ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(parent, text="Manual", value="manual",
                        variable=self.harris_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="OpenCV", value="opencv",
                        variable=self.harris_method_var).pack(anchor=tk.W, padx=20)

        # Parámetros
        params_frame = ttk.LabelFrame(parent, text="Parámetros", padding=5)
        params_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(params_frame, text="k:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(params_frame, from_=0.01, to=0.2, variable=self.harris_k_var).grid(row=0, column=1, sticky=tk.EW)

        ttk.Label(params_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(params_frame, from_=0.001, to=0.1, variable=self.harris_threshold_var).grid(row=1, column=1, sticky=tk.EW)

        params_frame.columnconfigure(1, weight=1)

        ttk.Button(parent, text="Detectar Esquinas",
                   command=self.apply_harris).pack(pady=10, fill=tk.X, padx=5)

        self.harris_result_label = ttk.Label(parent, text="", wraplength=250)
        self.harris_result_label.pack(pady=10)

    def create_skeleton_panel(self, parent):
        """Panel para esqueletonización"""
        info_text = """Esqueletonización:

Reduce objetos a 1 píxel
de ancho."""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=250).pack(pady=10)

        ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(parent, text="Morfológico OpenCV", value="opencv",
                        variable=self.skeleton_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="Morfológico Manual", value="manual",
                        variable=self.skeleton_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="Zhang-Suen", value="zhang_suen",
                        variable=self.skeleton_method_var).pack(anchor=tk.W, padx=20)

        ttk.Button(parent, text="Aplicar",
                   command=self.apply_skeletonization).pack(pady=10, fill=tk.X, padx=5)

        self.skeleton_result_label = ttk.Label(parent, text="", wraplength=250)
        self.skeleton_result_label.pack(pady=10)

    def create_perimeter_panel(self, parent):
        """Panel para análisis de perímetro"""
        info_text = """Análisis de Perímetro:

Mide contornos de objetos."""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=250).pack(pady=10)

        ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(parent, text="OpenCV", value="opencv",
                        variable=self.perimeter_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="Chain Code", value="chain_code",
                        variable=self.perimeter_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="Morfológico", value="morphological",
                        variable=self.perimeter_method_var).pack(anchor=tk.W, padx=20)

        ttk.Button(parent, text="Analizar",
                   command=self.apply_perimeter_analysis).pack(pady=10, fill=tk.X, padx=5)

        self.perimeter_result_text = tk.Text(parent, height=10, width=30)
        self.perimeter_result_text.pack(pady=10, padx=5)

    def create_segmentation_panel(self, parent):
        """Panel para segmentación"""
        info_text = """Segmentación:

Divide la imagen en regiones."""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=250).pack(pady=10)

        ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(parent, text="Threshold", value="threshold",
                        variable=self.seg_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="K-means", value="kmeans",
                        variable=self.seg_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="Watershed", value="watershed",
                        variable=self.seg_method_var).pack(anchor=tk.W, padx=20)

        # Parámetros
        params_frame = ttk.LabelFrame(parent, text="Parámetros", padding=5)
        params_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(params_frame, text="Clusters:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(params_frame, from_=2, to=10,
                  variable=self.seg_clusters_var).grid(row=0, column=1, sticky=tk.EW)

        ttk.Label(params_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(params_frame, from_=0, to=255,
                  variable=self.seg_threshold_var).grid(row=1, column=1, sticky=tk.EW)

        params_frame.columnconfigure(1, weight=1)

        ttk.Button(parent, text="Aplicar",
                   command=self.apply_segmentation).pack(pady=10, fill=tk.X, padx=5)

        self.seg_result_label = ttk.Label(parent, text="", wraplength=250)
        self.seg_result_label.pack(pady=10)

    def create_template_panel(self, parent):
        """Panel para template matching"""
        info_text = """Template Matching:

Busca una plantilla en
la imagen."""

        ttk.Label(parent, text=info_text, justify=tk.LEFT, wraplength=250).pack(pady=10)

        ttk.Button(parent, text="📁 Cargar Template",
                   command=self.load_template).pack(pady=5, fill=tk.X, padx=5)

        self.template_status = ttk.Label(parent, text="No hay template", wraplength=250)
        self.template_status.pack(pady=5)

        ttk.Label(parent, text="Método:").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(parent, text="Manual", value="manual",
                        variable=self.template_method_var).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(parent, text="OpenCV", value="opencv",
                        variable=self.template_method_var).pack(anchor=tk.W, padx=20)

        ttk.Button(parent, text="Buscar Template",
                   command=self.apply_template_matching).pack(pady=10, fill=tk.X, padx=5)

        self.template_result_label = ttk.Label(parent, text="", wraplength=250)
        self.template_result_label.pack(pady=10)

    # ========== MÉTODOS DE APLICACIÓN ==========

    def apply_otsu_manual(self):
        """Aplicar Otsu manual"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        try:
            img_array = np.array(self.history[self.history_index])
            threshold, result, metrics = OtsuThreshold.calculate_threshold(img_array)

            result_pil = Image.fromarray(result).convert('RGB')
            self.current_image = result_pil
            self.add_to_history()
            self.display_image_on_canvas()

            self.otsu_result_label.config(text=f"Umbral: {threshold}\nVarianza: {metrics['max_variance']:.2f}")
            self.status_bar.config(text=f"Otsu aplicado - Umbral: {threshold}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_otsu_opencv(self):
        """Aplicar Otsu OpenCV"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        try:
            img_array = np.array(self.history[self.history_index])
            threshold, result = OtsuThreshold.apply_otsu_opencv(img_array)

            result_pil = Image.fromarray(result).convert('RGB')
            self.current_image = result_pil
            self.add_to_history()
            self.display_image_on_canvas()

            self.otsu_result_label.config(text=f"Umbral OpenCV: {threshold:.2f}")
            self.status_bar.config(text=f"Otsu OpenCV - {threshold:.2f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_harris(self):
        """Aplicar Harris"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        try:
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

            self.harris_result_label.config(text=f"Método: {method}\nEsquinas: {len(coords)}")
            self.status_bar.config(text=f"Harris - {len(coords)} esquinas")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_skeletonization(self):
        """Aplicar esqueletonización"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        try:
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
            self.skeleton_result_label.config(text=f"Método: {method}\nIter: {iter_text}")
            self.status_bar.config(text=f"Esqueleto ({method})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_perimeter_analysis(self):
        """Aplicar análisis de perímetro"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        try:
            img_array = np.array(self.history[self.history_index])
            method = self.perimeter_method_var.get()

            perimeter_img, measurements = PerimeterAnalysis.calculate_perimeter(
                img_array, method=method
            )

            result_pil = Image.fromarray(perimeter_img)
            self.current_image = result_pil
            self.add_to_history()
            self.display_image_on_canvas()

            self.perimeter_result_text.delete(1.0, tk.END)
            self.perimeter_result_text.insert(1.0, f"Método: {measurements['method']}\n")
            self.perimeter_result_text.insert(tk.END, f"Objetos: {measurements['num_objects']}\n")

            if 'objects' in measurements and measurements['objects']:
                for obj in measurements['objects'][:3]:
                    self.perimeter_result_text.insert(tk.END, f"\nObj {obj['contour_id']}:\n")
                    if 'perimeter' in obj:
                        self.perimeter_result_text.insert(tk.END, f"P: {obj['perimeter']:.2f}\n")
                    if 'area' in obj:
                        self.perimeter_result_text.insert(tk.END, f"A: {obj['area']:.2f}\n")

            self.status_bar.config(text=f"Perímetro ({method})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_segmentation(self):
        """Aplicar segmentación"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        try:
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
            else:
                result, info = ImageSegmentation.region_growing(img_array)

            result_pil = Image.fromarray(result)
            self.current_image = result_pil
            self.add_to_history()
            self.display_image_on_canvas()

            info_text = f"Método: {info['method']}\n"
            if 'n_clusters' in info:
                info_text += f"Clusters: {info['n_clusters']}"

            self.seg_result_label.config(text=info_text)
            self.status_bar.config(text=f"Segmentación ({method})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_template(self):
        """Cargar template"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Template",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                self.template_image = Image.open(file_path).convert('RGB')
                size = self.template_image.size
                self.template_status.config(text=f"Template: {size[0]}x{size[1]}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def apply_template_matching(self):
        """Aplicar template matching"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen")
            return

        if not self.template_image:
            messagebox.showwarning("Advertencia", "Cargar template primero")
            return

        try:
            img_array = np.array(self.history[self.history_index])
            template_array = np.array(self.template_image)
            method = self.template_method_var.get()

            if method == 'manual':
                result, info = TemplateMatching.match_template_manual(
                    img_array, template_array, method='ssd'
                )
            else:
                result, info = TemplateMatching.match_template_opencv(
                    img_array, template_array, method='ccoeff_normed'
                )

            result_pil = Image.fromarray(result)
            self.current_image = result_pil
            self.add_to_history()
            self.display_image_on_canvas()

            self.template_result_label.config(
                text=f"Método: {method}\nPos: {info['best_location']}\nScore: {info['best_score']:.3f}"
            )
            self.status_bar.config(text=f"Template en {info['best_location']}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ============= MÉTODOS DE DETECCIÓN DE BORDES SEGÚN PDF =============
    # (Todo tu código original de detección de bordes aquí...)

    def normalize_image(self, image):
        """Normalizar imagen a rango [0, 255] usando min-max normalization"""
        if image.dtype != np.float64:
            image = image.astype(np.float64)

        min_val = np.min(image)
        max_val = np.max(image)

        if max_val == min_val:
            return np.zeros_like(image, dtype=np.uint8)

        normalized = (image - min_val) / (max_val - min_val)
        return (normalized * 255).astype(np.uint8)

    def gradient_operator(self, image_array):
        """Gradiente básico - Diferencias finitas"""
        grad_x = np.zeros_like(image_array, dtype=np.float64)
        grad_y = np.zeros_like(image_array, dtype=np.float64)

        grad_x[:, 1:-1] = (image_array[:, 2:] - image_array[:, :-2]) / 2.0
        grad_y[1:-1, :] = (image_array[2:, :] - image_array[:-2, :]) / 2.0

        grad_x[:, 0] = image_array[:, 1] - image_array[:, 0]
        grad_x[:, -1] = image_array[:, -1] - image_array[:, -2]
        grad_y[0, :] = image_array[1, :] - image_array[0, :]
        grad_y[-1, :] = image_array[-1, :] - image_array[-2, :]

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def sobel_operator(self, image_array):
        """Operador de Sobel"""
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def prewitt_operator(self, image_array):
        """Operador de Prewitt"""
        gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def roberts_operator(self, image_array):
        """Operador de Roberts"""
        h, w = image_array.shape
        grad_x = np.zeros_like(image_array, dtype=np.float64)
        grad_y = np.zeros_like(image_array, dtype=np.float64)

        grad_x[1:, 1:] = image_array[1:, 1:] - image_array[:-1, :-1]
        grad_y[1:, :-1] = image_array[1:, :-1] - image_array[:-1, 1:]

        if self.roberts_form_var.get() == "sqrt":
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        else:
            magnitude = np.abs(grad_x) + np.abs(grad_y)

        angle = np.arctan2(grad_y, grad_x)
        return magnitude, angle, grad_x, grad_y

    def kirsch_operator(self, image_array):
        """Máscaras de Kirsch"""
        masks = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float64),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float64),
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float64),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float64),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float64),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float64),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float64),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float64)
        ]

        responses = [ndimage.convolve(image_array.astype(np.float64), mask) for mask in masks]
        magnitude = np.maximum.reduce(responses)
        responses_stack = np.stack(responses, axis=-1)
        angle_indices = np.argmax(responses_stack, axis=-1)
        angle = angle_indices * 45.0

        return magnitude, np.deg2rad(angle), None, None

    def robinson_operator(self, image_array):
        """Máscaras de Robinson"""
        masks = [
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
            np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float64),
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64),
            np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float64),
            np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64),
            np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float64),
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
            np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float64)
        ]

        responses = [ndimage.convolve(image_array.astype(np.float64), mask) for mask in masks]
        magnitude = np.maximum.reduce(responses)
        responses_stack = np.stack(responses, axis=-1)
        angle_indices = np.argmax(responses_stack, axis=-1)
        angle = angle_indices * 45.0

        return magnitude, np.deg2rad(angle), None, None

    def frei_chen_operator(self, image_array):
        """Máscaras de Frei-Chen"""
        sqrt2 = np.sqrt(2)
        masks = [
            (1 / (2 * sqrt2)) * np.array([[1, sqrt2, 1], [0, 0, 0], [-1, -sqrt2, -1]], dtype=np.float64),
            (1 / (2 * sqrt2)) * np.array([[1, 0, -1], [sqrt2, 0, -sqrt2], [1, 0, -1]], dtype=np.float64),
            (1 / (2 * sqrt2)) * np.array([[0, -1, sqrt2], [1, 0, -1], [-sqrt2, 1, 0]], dtype=np.float64),
            (1 / (2 * sqrt2)) * np.array([[sqrt2, -1, 0], [-1, 0, 1], [0, 1, -sqrt2]], dtype=np.float64),
            0.5 * np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]], dtype=np.float64),
            0.5 * np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float64),
            (1 / 6) * np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64),
            (1 / 6) * np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]], dtype=np.float64),
            (1 / 3) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64)
        ]

        projections = [ndimage.convolve(image_array.astype(np.float64), mask) for mask in masks]
        edge_responses = [p ** 2 for p in projections[:4]]
        all_responses = [p ** 2 for p in projections]

        M = np.sum(edge_responses, axis=0)
        S = np.sum(all_responses, axis=0)
        S_safe = np.where(S > 0, S, 1)
        cos_theta = np.sqrt(np.clip(M / S_safe, 0, 1))
        magnitude = np.sqrt(M)

        return magnitude, cos_theta, None, None

    def extended_sobel_operator(self, image_array, size=7):
        """Operador de Sobel Extendido"""
        if size == 3:
            return self.sobel_operator(image_array)

        if size == 5:
            gx = np.array([[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3],
                           [-2, -3, 0, 3, 2], [-1, -2, 0, 2, 1]], dtype=np.float64)
        elif size == 7:
            gx = np.array([[-1, -1, -1, 0, 1, 1, 1], [-1, -2, -2, 0, 2, 2, 1],
                           [-1, -2, -3, 0, 3, 2, 1], [-1, -2, -3, 0, 3, 2, 1],
                           [-1, -2, -3, 0, 3, 2, 1], [-1, -2, -2, 0, 2, 2, 1],
                           [-1, -1, -1, 0, 1, 1, 1]], dtype=np.float64)
        else:
            return self.sobel_operator(image_array)

        gy = gx.T
        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def canny_operator(self, image_array):
        """Algoritmo de Canny"""
        try:
            img_uint8 = image_array.astype(np.uint8)
            sigma = self.sigma_var.get()
            ksize = int(2 * np.ceil(2 * sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)
            low_threshold = int(self.canny_low_var.get())
            high_threshold = int(self.canny_high_var.get())
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            return edges, None, None, None
        except:
            return self.canny_manual(image_array)

    def canny_manual(self, image_array):
        """Implementación manual de Canny"""
        sigma = self.sigma_var.get()
        smoothed = gaussian_filter(image_array.astype(np.float64), sigma=sigma)
        magnitude, angle, grad_x, grad_y = self.sobel_operator(smoothed)
        suppressed = self.non_maximum_suppression(magnitude, angle)
        low_threshold = self.canny_low_var.get()
        high_threshold = self.canny_high_var.get()
        edges = self.hysteresis_threshold(suppressed, low_threshold, high_threshold)
        return (edges * 255).astype(np.uint8), angle, grad_x, grad_y

    def non_maximum_suppression(self, magnitude, angle):
        """Supresión no máxima"""
        rows, cols = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        angle_deg = (np.rad2deg(angle) + 180) % 180

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                current = magnitude[i, j]
                ang = angle_deg[i, j]

                if (ang >= 0 and ang < 22.5) or (ang >= 157.5 and ang < 180):
                    neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
                elif ang >= 22.5 and ang < 67.5:
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
                elif ang >= 67.5 and ang < 112.5:
                    neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
                else:
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

                if current >= max(neighbors):
                    suppressed[i, j] = current

        return suppressed

    def hysteresis_threshold(self, image, low_threshold, high_threshold):
        """Histéresis de umbral"""
        strong_edges = image > high_threshold
        weak_edges = (image >= low_threshold) & (image <= high_threshold)
        strong_dilated = binary_dilation(strong_edges, structure=np.ones((3, 3)))
        connected_weak = weak_edges & strong_dilated
        edges = strong_edges | connected_weak
        return edges.astype(np.float32)

    def laplacian_operator(self, image_array):
        """Laplaciano de la Gaussiana"""
        sigma = self.sigma_var.get()
        smoothed = gaussian_filter(image_array.astype(np.float64), sigma=sigma)
        laplacian = ndimage.laplace(smoothed)
        zero_crossings = self.detect_zero_crossings(laplacian)
        return zero_crossings, None, None, None

    def detect_zero_crossings(self, laplacian):
        """Detectar zero-crossings"""
        zc = np.zeros_like(laplacian, dtype=np.uint8)
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            rolled = np.roll(np.roll(laplacian, dx, axis=0), dy, axis=1)
            zc |= ((laplacian * rolled) < 0).astype(np.uint8)
        return zc * 255

    def apply_edge_detection(self):
        """Aplicar detección de bordes"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return

        try:
            gray_image = self.history[self.history_index].convert('L')
            image_array = np.array(gray_image, dtype=np.float32)
            operator = self.edge_operator_var.get()
            size = self.extended_size_var.get()

            if operator == "gradient":
                magnitude, angle, grad_x, grad_y = self.gradient_operator(image_array)
            elif operator == "sobel":
                if size > 3:
                    magnitude, angle, grad_x, grad_y = self.extended_sobel_operator(image_array, size)
                else:
                    magnitude, angle, grad_x, grad_y = self.sobel_operator(image_array)
            elif operator == "prewitt":
                magnitude, angle, grad_x, grad_y = self.prewitt_operator(image_array)
            elif operator == "roberts":
                magnitude, angle, grad_x, grad_y = self.roberts_operator(image_array)
            elif operator == "kirsch":
                magnitude, angle, _, _ = self.kirsch_operator(image_array)
            elif operator == "robinson":
                magnitude, angle, _, _ = self.robinson_operator(image_array)
            elif operator == "frei_chen":
                magnitude, cos_theta, _, _ = self.frei_chen_operator(image_array)
                angle = np.arccos(np.clip(cos_theta, 0, 1))
            elif operator == "canny":
                result, angle, grad_x, grad_y = self.canny_operator(image_array)
                edge_image = Image.fromarray(result).convert('RGB')
                self.current_image = edge_image
                self.add_to_history()
                self.display_image_on_canvas()
                self.status_bar.config(text=f"PhotoEscom - Operador {operator} aplicado")
                return
            elif operator == "laplacian":
                result, _, _, _ = self.laplacian_operator(image_array)
                edge_image = Image.fromarray(result).convert('RGB')
                self.current_image = edge_image
                self.add_to_history()
                self.display_image_on_canvas()
                self.status_bar.config(text=f"PhotoEscom - Laplaciano aplicado")
                return
            else:
                messagebox.showerror("Error", f"Operador no reconocido: {operator}")
                return

            if self.show_magnitude_var.get():
                result = self.normalize_image(magnitude)
            elif self.show_angle_var.get() and angle is not None:
                angle_normalized = ((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                result = angle_normalized
            else:
                threshold = self.threshold_var.get()
                magnitude_norm = self.normalize_image(magnitude)
                result = np.where(magnitude_norm > threshold, 255, 0).astype(np.uint8)

            edge_image = Image.fromarray(result).convert('RGB')
            self.current_image = edge_image
            self.add_to_history()
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotoEscom - Operador {operator} aplicado")

        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")

    def preview_edge_detection(self):
        """Vista previa rápida"""
        if not self.current_image:
            return

        try:
            original_img = self.history[self.history_index]
            if max(original_img.size) > 800:
                ratio = 800 / max(original_img.size)
                new_size = (int(original_img.size[0] * ratio), int(original_img.size[1] * ratio))
                preview_img = original_img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                preview_img = original_img

            gray_image = preview_img.convert('L')
            image_array = np.array(gray_image, dtype=np.float32)
            operator = self.edge_operator_var.get()

            if operator == "canny":
                result, _, _, _ = self.canny_operator(image_array)
            else:
                if operator == "sobel":
                    magnitude, _, _, _ = self.sobel_operator(image_array)
                elif operator == "prewitt":
                    magnitude, _, _, _ = self.prewitt_operator(image_array)
                elif operator == "roberts":
                    magnitude, _, _, _ = self.roberts_operator(image_array)
                elif operator == "gradient":
                    magnitude, _, _, _ = self.gradient_operator(image_array)
                elif operator == "kirsch":
                    magnitude, _, _, _ = self.kirsch_operator(image_array)
                elif operator == "robinson":
                    magnitude, _, _, _ = self.robinson_operator(image_array)
                elif operator == "frei_chen":
                    magnitude, _, _, _ = self.frei_chen_operator(image_array)
                elif operator == "laplacian":
                    result, _, _, _ = self.laplacian_operator(image_array)
                    magnitude = None
                else:
                    return

                if magnitude is not None:
                    threshold = self.threshold_var.get()
                    magnitude_norm = self.normalize_image(magnitude)
                    result = np.where(magnitude_norm > threshold, 255, 0).astype(np.uint8)

            if max(original_img.size) > 800:
                result_pil = Image.fromarray(result)
                result_pil = result_pil.resize(original_img.size, Image.Resampling.NEAREST)
                result = np.array(result_pil)

            edge_image = Image.fromarray(result).convert('RGB')
            self.current_image = edge_image
            self.display_image_on_canvas()
            self.status_bar.config(text=f"PhotoEscom - Vista previa: {operator}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_edge_detection_panel(self, parent):
        """Panel de detección de bordes"""
        canvas = tk.Canvas(parent, bg=self.frame_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Primera Derivada
        first_deriv_frame = ttk.LabelFrame(scrollable_frame, text="Primera Derivada", padding=10)
        first_deriv_frame.pack(fill=tk.X, pady=(0, 10))

        operators_first = [("Gradiente Básico", "gradient"), ("Sobel", "sobel"),
                           ("Prewitt", "prewitt"), ("Roberts", "roberts")]

        for i, (text, value) in enumerate(operators_first):
            ttk.Radiobutton(first_deriv_frame, text=text, value=value,
                            variable=self.edge_operator_var).grid(row=i // 2, column=i % 2, sticky=tk.W, pady=2)

        # Compass
        compass_frame = ttk.LabelFrame(scrollable_frame, text="Compass", padding=10)
        compass_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(compass_frame, text="Kirsch (8 dir)", value="kirsch",
                        variable=self.edge_operator_var).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(compass_frame, text="Robinson (8 dir)", value="robinson",
                        variable=self.edge_operator_var).grid(row=0, column=1, sticky=tk.W, pady=2)

        # Frei-Chen
        frei_frame = ttk.LabelFrame(scrollable_frame, text="Frei-Chen", padding=10)
        frei_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(frei_frame, text="Frei-Chen (9 máscaras)", value="frei_chen",
                        variable=self.edge_operator_var).pack(anchor=tk.W)

        # Canny
        canny_frame = ttk.LabelFrame(scrollable_frame, text="Canny", padding=10)
        canny_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(canny_frame, text="Canny (Óptimo)", value="canny",
                        variable=self.edge_operator_var).pack(anchor=tk.W)

        # Segunda Derivada
        second_deriv_frame = ttk.LabelFrame(scrollable_frame, text="Segunda Derivada", padding=10)
        second_deriv_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(second_deriv_frame, text="Laplaciano (LoG)", value="laplacian",
                        variable=self.edge_operator_var).pack(anchor=tk.W)

        # Parámetros
        params_frame = ttk.LabelFrame(scrollable_frame, text="Parámetros", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(params_frame, text="Umbral T:").grid(row=0, column=0, sticky=tk.W, pady=5)
        threshold_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.threshold_var)
        threshold_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        threshold_value = ttk.Label(params_frame, text="30", width=5)
        threshold_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        ttk.Label(params_frame, text="Sigma (σ):").grid(row=1, column=0, sticky=tk.W, pady=5)
        sigma_scale = ttk.Scale(params_frame, from_=0.5, to=3.0, variable=self.sigma_var)
        sigma_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        sigma_value = ttk.Label(params_frame, text="1.0", width=5)
        sigma_value.grid(row=1, column=2, padx=(5, 0), pady=5)

        ttk.Label(params_frame, text="Canny Low:").grid(row=2, column=0, sticky=tk.W, pady=5)
        canny_low_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.canny_low_var)
        canny_low_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        canny_low_value = ttk.Label(params_frame, text="50", width=5)
        canny_low_value.grid(row=2, column=2, padx=(5, 0), pady=5)

        ttk.Label(params_frame, text="Canny High:").grid(row=3, column=0, sticky=tk.W, pady=5)
        canny_high_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.canny_high_var)
        canny_high_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        canny_high_value = ttk.Label(params_frame, text="150", width=5)
        canny_high_value.grid(row=3, column=2, padx=(5, 0), pady=5)

        ttk.Label(params_frame, text="Roberts Form:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(params_frame, text="Sqrt", value="sqrt", variable=self.roberts_form_var).grid(row=4, column=1, sticky=tk.W)
        ttk.Radiobutton(params_frame, text="Abs", value="abs", variable=self.roberts_form_var).grid(row=4, column=2, sticky=tk.W)

        ttk.Checkbutton(params_frame, text="Mostrar Magnitud", variable=self.show_magnitude_var).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5)

        ttk.Checkbutton(params_frame, text="Mostrar Ángulo", variable=self.show_angle_var).grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)

        ttk.Label(params_frame, text="Sobel Extendido:").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(params_frame, text="3x3", value=3, variable=self.extended_size_var).grid(row=7, column=1, sticky=tk.W)
        ttk.Radiobutton(params_frame, text="5x5", value=5, variable=self.extended_size_var).grid(row=7, column=2, sticky=tk.W)
        ttk.Radiobutton(params_frame, text="7x7", value=7, variable=self.extended_size_var).grid(row=8, column=1, sticky=tk.W)

        params_frame.columnconfigure(1, weight=1)

        # Botones
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="Vista Previa",
                   command=self.preview_edge_detection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Aplicar",
                   command=self.apply_edge_detection).pack(side=tk.LEFT)

        def update_param_values(*args):
            threshold_value.config(text=f"{self.threshold_var.get():.0f}")
            sigma_value.config(text=f"{self.sigma_var.get():.1f}")
            canny_low_value.config(text=f"{self.canny_low_var.get():.0f}")
            canny_high_value.config(text=f"{self.canny_high_var.get():.0f}")

        self.threshold_var.trace('w', update_param_values)
        self.sigma_var.trace('w', update_param_values)
        self.canny_low_var.trace('w', update_param_values)
        self.canny_high_var.trace('w', update_param_values)

    def update_operator_info(self, *args):
        """Actualizar información del operador"""
        operator = self.edge_operator_var.get()
        info_texts = {
            "gradient": "Gradiente Básico",
            "sobel": "Sobel",
            "prewitt": "Prewitt",
            "roberts": "Roberts",
            "kirsch": "Kirsch",
            "robinson": "Robinson",
            "frei_chen": "Frei-Chen",
            "canny": "Canny",
            "laplacian": "Laplaciano"
        }
        if hasattr(self, 'operator_info'):
            self.operator_info.config(text=info_texts.get(operator, ""))

    def create_top_toolbar(self):
        """Barra de herramientas superior"""
        toolbar = ttk.Frame(self.root, height=50)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 5))
        toolbar.pack_propagate(False)

        title_label = ttk.Label(toolbar, text="PhotoEscom", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=(10, 20))

        ttk.Button(toolbar, text="📁 Cargar", command=self.load_image, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="💾 Guardar", command=self.save_image, width=12).pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Button(toolbar, text="↶ Deshacer", command=self.undo, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="↷ Rehacer", command=self.redo, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="🔄 Restaurar", command=self.reset_image, width=12).pack(side=tk.LEFT, padx=5)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Button(toolbar, text="➕ Zoom In", command=self.zoom_in, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="➖ Zoom Out", command=self.zoom_out, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="📏 Ajustar", command=self.zoom_fit, width=10).pack(side=tk.LEFT, padx=5)

    def create_image_canvas(self):
        """Crear canvas de imagen"""
        canvas_container = ttk.Frame(self.image_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)

        self.canvas = tk.Canvas(canvas_container, bg='#1e1e1e', highlightthickness=0,
                                yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)

        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        v_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        h_scrollbar.grid(row=1, column=0, sticky=tk.EW)

        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.image_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")

        self.image_container.bind("<Configure>", self.on_container_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def create_basic_tools_panel(self, parent):
        """Panel de herramientas básicas"""
        info_frame = ttk.LabelFrame(parent, text="Información", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_label = ttk.Label(info_frame, text="Sin imagen cargada", justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)

        tools_frame = ttk.LabelFrame(parent, text="Herramientas", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 10))

        flip_frame = ttk.Frame(tools_frame)
        flip_frame.pack(fill=tk.X, pady=5)
        ttk.Button(flip_frame, text="↔ Horizontal", command=lambda: self.apply_flip('horizontal')).pack(
            side=tk.LEFT, expand=True, padx=2)
        ttk.Button(flip_frame, text="↕ Vertical", command=lambda: self.apply_flip('vertical')).pack(
            side=tk.LEFT, expand=True, padx=2)

        rotate_frame = ttk.Frame(tools_frame)
        rotate_frame.pack(fill=tk.X, pady=5)
        ttk.Button(rotate_frame, text="↺ 90°", command=lambda: self.quick_rotate(-90)).pack(side=tk.LEFT, expand=True,
                                                                                            padx=2)
        ttk.Button(rotate_frame, text="↻ 90°", command=lambda: self.quick_rotate(90)).pack(side=tk.LEFT, expand=True,
                                                                                           padx=2)

    def create_transform_panel(self, parent):
        """Panel de transformación"""
        ttk.Label(parent, text="Rotación:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Scale(parent, from_=-180, to=180, variable=self.rotate_var).grid(row=0, column=1, sticky=tk.EW, pady=5)
        ttk.Label(parent, text="Escala X:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_x_var).grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Label(parent, text="Escala Y:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_y_var).grid(row=2, column=1, sticky=tk.EW, pady=5)
        ttk.Button(parent, text="Aplicar", command=self.apply_transforms).grid(row=3, column=0, columnspan=2, pady=10)
        parent.columnconfigure(1, weight=1)

    def create_adjustments_panel(self, parent):
        """Panel de ajustes"""
        adjustments = [("Brillo:", self.brightness_var, 0.1, 3.0),
                       ("Contraste:", self.contrast_var, 0.1, 3.0),
                       ("Saturación:", self.saturation_var, 0.0, 3.0),
                       ("Nitidez:", self.sharpen_var, 1.0, 5.0)]

        for i, (label, var, min_val, max_val) in enumerate(adjustments):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            ttk.Scale(parent, from_=min_val, to=max_val, variable=var).grid(row=i, column=1, sticky=tk.EW, pady=5)

        ttk.Button(parent, text="Aplicar", command=self.finalize_adjustments).grid(row=len(adjustments), column=0,
                                                                                   columnspan=2, pady=10)
        parent.columnconfigure(1, weight=1)

    def create_filters_panel(self, parent):
        """Panel de filtros"""
        filters = [("Original", "original"), ("Escala de Grises", "grayscale"),
                   ("Sepia", "sepia"), ("Invertir", "invert"),
                   ("Desenfoque", "blur"), ("Detalle", "detail")]

        for i, (text, mode) in enumerate(filters):
            ttk.Radiobutton(parent, text=text, value=mode,
                            variable=self.filter_var, command=self.apply_filter).grid(row=i, column=0, sticky=tk.W,
                                                                                      pady=2)

    def load_image(self):
        """Cargar imagen"""
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_path:
            try:
                self.original_image = Image.open(file_path).convert('RGB')
                self.current_image = self.original_image.copy()
                self.filename = os.path.basename(file_path)
                self.history = [self.current_image.copy()]
                self.history_index = 0

                self.reset_controls()
                self.zoom_fit()
                self.display_image_on_canvas()
                self.update_info()
                self.status_bar.config(text=f"PhotoEscom - Imagen cargada: {self.filename}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar: {str(e)}")

    def reset_controls(self):
        """Resetear controles"""
        self.rotate_var.set(0)
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.sharpen_var.set(1.0)
        self.filter_var.set("original")

    def update_info(self):
        """Actualizar información"""
        if self.current_image:
            width, height = self.current_image.size
            info_text = f"Archivo: {self.filename or 'N/A'}\nDimensiones: {width} x {height}\nHistorial: {self.history_index + 1}/{len(self.history)}"
            self.info_label.config(text=info_text)
        else:
            self.info_label.config(text="Sin imagen cargada")

    def display_image_on_canvas(self):
        """Mostrar imagen en canvas"""
        if self.current_image:
            width, height = self.current_image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)

            if new_width <= 0: new_width = 1
            if new_height <= 0: new_height = 1

            img = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(img)

            if hasattr(self, 'image_label'):
                self.image_label.config(image=self.display_image)
            else:
                self.image_label = ttk.Label(self.image_container, image=self.display_image)
                self.image_label.pack()

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def save_image(self):
        """Guardar imagen"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            try:
                self.current_image.save(file_path)
                messagebox.showinfo("Éxito", "Imagen guardada")
                self.status_bar.config(text=f"PhotoEscom - Guardado: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def add_to_history(self):
        """Agregar a historial"""
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(self.current_image.copy())
        self.history_index = len(self.history) - 1
        if len(self.history) > 20:
            self.history.pop(0)
            self.history_index -= 1
        self.update_info()

    def undo(self):
        """Deshacer"""
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.update_info()

    def redo(self):
        """Rehacer"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.update_info()

    def reset_image(self):
        """Resetear imagen"""
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.add_to_history()
            self.reset_controls()
            self.zoom_fit()
            self.display_image_on_canvas()

    def quick_rotate(self, angle):
        """Rotación rápida"""
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        img = img.rotate(angle, expand=True)
        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()

    def preview_transforms(self, *args):
        """Vista previa de transformaciones"""
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        angle = self.rotate_var.get()
        if angle != 0:
            img = img.rotate(angle, expand=True)
        scale_x = self.scale_x_var.get()
        scale_y = self.scale_y_var.get()
        if scale_x != 1.0 or scale_y != 1.0:
            new_size = (int(img.width * scale_x), int(img.height * scale_y))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        self.current_image = img
        self.display_image_on_canvas()

    def apply_transforms(self):
        """Aplicar transformaciones"""
        if not self.current_image:
            return
        self.add_to_history()
        self.rotate_var.set(0)
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)

    def apply_flip(self, direction):
        """Voltear imagen"""
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        if direction == 'horizontal':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()

    def preview_adjustments(self, *args):
        """Vista previa de ajustes"""
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        brightness = self.brightness_var.get()
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        contrast = self.contrast_var.get()
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        saturation = self.saturation_var.get()
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)
        sharpen = self.sharpen_var.get()
        if sharpen != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpen)
        self.current_image = img
        self.display_image_on_canvas()

    def finalize_adjustments(self):
        """Finalizar ajustes"""
        if not self.current_image:
            return
        self.add_to_history()
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.sharpen_var.set(1.0)

    def apply_filter(self):
        """Aplicar filtro"""
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        filter_type = self.filter_var.get()
        if filter_type == "grayscale":
            img = img.convert("L").convert("RGB")
        elif filter_type == "sepia":
            img_array = np.array(img)
            sepia_filter = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
            sepia_img = np.dot(img_array, sepia_filter.T)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            img = Image.fromarray(sepia_img)
        elif filter_type == "invert":
            img_array = np.array(img)
            img_array = 255 - img_array
            img = Image.fromarray(img_array)
        elif filter_type == "blur":
            img = img.filter(ImageFilter.BLUR)
        self.current_image = img
        self.display_image_on_canvas()
        if filter_type != "original":
            self.add_to_history()

    def zoom_in(self):
        """Acercar"""
        if self.current_image:
            self.zoom_factor *= 1.2
            self.display_image_on_canvas()

    def zoom_out(self):
        """Alejar"""
        if self.current_image:
            self.zoom_factor /= 1.2
            self.display_image_on_canvas()

    def zoom_fit(self):
        """Ajustar zoom para que la imagen se vea completa en el lienzo."""
        if not self.current_image:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        canvas_width -= 10
        canvas_height -= 10

        if canvas_width <= 0 or canvas_height <= 0:
            self.zoom_factor = 1.0
            self.display_image_on_canvas()
            return

        img_width, img_height = self.current_image.size
        if img_width <= 0 or img_height <= 0:
            return

        ratio_w = canvas_width / img_width
        ratio_h = canvas_height / img_height

        self.zoom_factor = min(ratio_w, ratio_h)

        if self.zoom_factor <= 0:
            self.zoom_factor = 0.1

        self.display_image_on_canvas()
        self.status_bar.config(text=f"PhotoEscom - Zoom ajustado al lienzo ({self.zoom_factor:.2f})")

    def on_container_configure(self, event):
        """Evento de configuración de contenedor"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Evento de configuración de canvas"""
        self.canvas.itemconfig(self.canvas_window, width=event.width, height=event.height)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_resize(self, event):
        """Evento de redimensionamiento"""
        if event.widget == self.root:
            self.display_image_on_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoEditor(root)
    root.bind("<Configure>", app.on_resize)
    root.mainloop()