"""
PhotoEscom - Editor de Imágenes con Visión por Computadora
Reconocimiento de billetes 100% local con SVM (sin APIs externas)
"""

from vision_methods import OtsuThreshold, HarrisCornerDetector
from skeleton_perimeter import SkeletonizationMethods, PerimeterAnalysis
from segmentation_template import ImageSegmentation, TemplateMatching
from bill_detector import BillDetector
from train_bill_model import extract_enhanced_descriptors

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import os
import pickle
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label, binary_dilation


class PhotoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("PhotoEscom - Editor Profesional + Reconocimiento de Billetes")
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

        # Variables para detección de bordes
        self.edge_operator_var = tk.StringVar(value="sobel")
        self.threshold_var = tk.DoubleVar(value=30.0)
        self.sigma_var = tk.DoubleVar(value=1.0)
        self.canny_low_var = tk.DoubleVar(value=50.0)
        self.canny_high_var = tk.DoubleVar(value=150.0)
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

        # Variables para reconocimiento de billetes (SVM)
        self.recognition_confidence_var = tk.DoubleVar(value=0.5)
        self.preprocess_bills_var = tk.BooleanVar(value=True)
        self.show_all_detections_var = tk.BooleanVar(value=False)

        # Modelo SVM
        self.svm_model = None
        self.svm_scaler = None
        self.svm_class_names = None

        # Detector de billetes
        self.bill_detector = BillDetector()

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

        # Cargar modelo SVM al iniciar
        self.load_svm_model()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.bg_color = '#2e2e2e'
        self.frame_bg = '#3c3c3c'
        self.button_bg = '#4a4a4a'
        self.accent_color = '#007acc'
        self.text_color = '#ffffff'
        self.highlight_color = '#4a76cf'

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

    def load_svm_model(self):
        """Carga el modelo SVM entrenado"""
        model_path = 'models/bill_recognizer.pkl'

        if not os.path.exists(model_path):
            print("⚠️  Modelo SVM no encontrado")
            print(f"   Ejecuta 'python train_bill_model.py' primero")
            return False

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.svm_model = model_data['classifier']
            self.svm_scaler = model_data['scaler']
            self.svm_class_names = model_data['class_names']

            test_acc = model_data.get('test_accuracy', 0)
            print(f"✓ Modelo SVM cargado exitosamente")
            print(f"  Test Accuracy: {test_acc:.2%}")
            print(f"  Clases: {len(self.svm_class_names)}")

            return True

        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            return False

    def create_widgets(self):
        # Barra superior
        self.create_top_toolbar()

        # Panel principal
        main_panel = ttk.Frame(self.root)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Panel de herramientas (notebook único)
        tools_notebook = ttk.Notebook(main_panel, width=300)
        tools_notebook.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tools_notebook.pack_propagate(False)

        # Pestañas
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
        tools_notebook.add(recognition_frame, text="💵 Billetes")

        # Panel de visualización
        self.image_frame = ttk.Frame(main_panel)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Crear lienzo
        self.create_image_canvas()

        # Llenar pestañas
        self.create_basic_tools_panel(basic_tools_frame)
        self.create_transform_panel(transform_frame)
        self.create_adjustments_panel(adjust_frame)
        self.create_filters_panel(filter_frame)
        self.create_edge_detection_panel(edge_frame)
        self.create_otsu_panel(otsu_frame)
        self.create_harris_panel(harris_frame)
        self.create_skeleton_panel(skeleton_frame)
        self.create_perimeter_panel(perimeter_frame)
        self.create_segmentation_panel(segmentation_frame)
        self.create_template_panel(template_frame)
        self.create_recognition_panel(recognition_frame)

        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="PhotoEscom - Listo")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ============= PANEL DE RECONOCIMIENTO DE BILLETES (100% SVM) =============

    def create_recognition_panel(self, parent):
        """Panel para reconocimiento de billetes usando SVM"""

        # Título
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(title_frame, text="💵 Reconocimiento",
                 font=("Arial", 12, "bold")).pack()

        # Información
        info_text = """Reconocimiento 100% Local:

✔ Detección con OpenCV
✔ Descriptores personalizados
✔ Clasificación con SVM

Sin APIs externas, todo local."""

        info_label = ttk.Label(parent, text=info_text, justify=tk.LEFT,
                              wraplength=250, font=("Arial", 9))
        info_label.pack(pady=10, padx=5)

        # Estado del modelo
        model_frame = ttk.LabelFrame(parent, text="Estado del Modelo", padding=10)
        model_frame.pack(fill=tk.X, pady=10, padx=5)

        if self.svm_model:
            status_text = f"✓ Modelo cargado\nClases: {len(self.svm_class_names)}"
            status_color = "green"
        else:
            status_text = "✗ Modelo no disponible\nEjecuta train_bill_model.py"
            status_color = "red"

        self.model_status_label = ttk.Label(model_frame, text=status_text,
                                           foreground=status_color)
        self.model_status_label.pack()

        ttk.Button(model_frame, text="Recargar Modelo",
                  command=self.reload_svm_model).pack(pady=5, fill=tk.X)

        # Parámetros
        params_frame = ttk.LabelFrame(parent, text="Parámetros", padding=10)
        params_frame.pack(fill=tk.X, pady=10, padx=5)

        # Confianza mínima
        ttk.Label(params_frame, text="Confianza Mínima:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Scale(params_frame, from_=0.1, to=0.95,
                 variable=self.recognition_confidence_var).grid(
            row=0, column=1, sticky=tk.EW, pady=5
        )

        self.conf_display = ttk.Label(params_frame,
                                     text=f"{self.recognition_confidence_var.get():.0%}")
        self.conf_display.grid(row=0, column=2, padx=5, pady=5)

        def update_conf_display(*args):
            self.conf_display.config(
                text=f"{self.recognition_confidence_var.get():.0%}"
            )

        self.recognition_confidence_var.trace('w', update_conf_display)

        # Opciones
        ttk.Checkbutton(
            params_frame,
            text="Preprocesar imagen",
            variable=self.preprocess_bills_var
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)

        ttk.Checkbutton(
            params_frame,
            text="Mostrar todas las detecciones",
            variable=self.show_all_detections_var
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

        params_frame.columnconfigure(1, weight=1)

        # Botón principal
        ttk.Button(parent, text="🔍 Reconocer Billetes",
                  command=self.recognize_bills_svm).pack(pady=15, fill=tk.X, padx=5)

        # Resultados
        results_frame = ttk.LabelFrame(parent, text="Resultados", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        # Scrollbar para resultados
        result_scroll = ttk.Scrollbar(results_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.recognition_result_text = tk.Text(
            results_frame,
            height=15,
            width=30,
            wrap=tk.WORD,
            yscrollcommand=result_scroll.set,
            font=("Courier", 9)
        )
        self.recognition_result_text.pack(fill=tk.BOTH, expand=True)
        result_scroll.config(command=self.recognition_result_text.yview)

        # Tags para colores
        self.recognition_result_text.tag_config("header", font=("Arial", 10, "bold"))
        self.recognition_result_text.tag_config("success", foreground="green")
        self.recognition_result_text.tag_config("warning", foreground="orange")
        self.recognition_result_text.tag_config("error", foreground="red")
        self.recognition_result_text.tag_config("info", foreground="cyan")

        # Texto inicial
        self.recognition_result_text.insert("1.0", "Esperando imagen...\n\n", "info")
        self.recognition_result_text.insert("end", "Carga una imagen con billetes\n")
        self.recognition_result_text.insert("end", "y presiona 'Reconocer Billetes'")
        self.recognition_result_text.config(state=tk.DISABLED)

    def reload_svm_model(self):
        """Recarga el modelo SVM"""
        if self.load_svm_model():
            messagebox.showinfo("Éxito", "Modelo recargado correctamente")

            # Actualizar estado
            status_text = f"✓ Modelo cargado\nClases: {len(self.svm_class_names)}"
            self.model_status_label.config(text=status_text, foreground="green")
        else:
            messagebox.showerror("Error", "No se pudo cargar el modelo")

    def update_recognition_results(self, text, tag=""):
        """Actualiza el área de resultados"""
        self.recognition_result_text.config(state=tk.NORMAL)
        self.recognition_result_text.insert("end", text, tag)
        self.recognition_result_text.config(state=tk.DISABLED)
        self.recognition_result_text.see("end")
        self.root.update()

    def recognize_bills_svm(self):
        """Reconocimiento de billetes usando SVM puro (sin APIs)"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return

        if not self.svm_model:
            messagebox.showerror("Error",
                "Modelo SVM no disponible.\n\n"
                "Por favor:\n"
                "1. Ejecuta 'python train_bill_model.py'\n"
                "2. Presiona 'Recargar Modelo'")
            return

        try:
            # Limpiar resultados
            self.recognition_result_text.config(state=tk.NORMAL)
            self.recognition_result_text.delete("1.0", tk.END)

            self.recognition_result_text.insert("1.0", "🔍 RECONOCIENDO BILLETES\n", "header")
            self.recognition_result_text.insert("end", "="*30 + "\n\n")
            self.recognition_result_text.config(state=tk.DISABLED)
            self.root.update()

            # Convertir imagen
            img_array = np.array(self.current_image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            print(f"\n{'='*60}")
            print("RECONOCIMIENTO DE BILLETES - 100% SVM")
            print(f"{'='*60}")
            print(f"Imagen: {img_cv.shape[1]}x{img_cv.shape[0]} px")

            # PASO 1: PREPROCESAMIENTO
            if self.preprocess_bills_var.get():
                print("\n[1/4] Preprocesando...")
                self.update_recognition_results("\n[1/4] Preprocesando...\n", "info")
                img_cv = self.bill_detector.preprocess_image(img_cv)
                print("   ✓ CLAHE + Bilateral filter aplicados")
            else:
                print("\n[1/4] Sin preprocesamiento")
                self.update_recognition_results("\n[1/4] Sin preprocesamiento\n", "info")

            # PASO 2: DETECCIÓN
            print("\n[2/4] Detectando billetes...")
            self.update_recognition_results("\n[2/4] Detectando billetes...\n", "info")
            self.root.update()

            detections = self.bill_detector.detect_bills(img_cv, debug=False)

            print(f"   ✓ {len(detections)} objetos detectados")
            self.update_recognition_results(f"   Detectados: {len(detections)}\n", "success")

            if len(detections) == 0:
                print("   ⚠️  No se detectaron billetes")
                self.update_recognition_results(
                    "\n⚠️  No se detectaron billetes\n\n"
                    "Sugerencias:\n"
                    "• Mejora la iluminación\n"
                    "• Acerca más los billetes\n"
                    "• Activa el preprocesamiento\n"
                    "• Verifica que sean billetes mexicanos\n",
                    "warning"
                )
                self.status_bar.config(text="Sin detecciones")
                return

            # PASO 3: EXTRACCIÓN Y CLASIFICACIÓN
            print(f"\n[3/4] Clasificando con SVM...")
            self.update_recognition_results("\n[3/4] Clasificando...\n", "info")
            self.root.update()

            labels = []
            confidences = []
            min_confidence = self.recognition_confidence_var.get()

            for i, detection in enumerate(detections, 1):
                print(f"\n   Billete #{i}:")

                # Extraer región
                x, y, w, h = detection['bbox']
                crop = img_cv[y:y+h, x:x+w]

                # Extraer descriptores
                features, _ = extract_enhanced_descriptors(
                    None, None, None, None, crop=crop
                )

                if features is None:
                    print(f"      ✗ Error al extraer características")
                    labels.append("Error")
                    confidences.append(0.0)
                    continue

                # Escalar y predecir
                features_scaled = self.svm_scaler.transform([features])
                pred_id = self.svm_model.predict(features_scaled)[0]
                pred_proba = self.svm_model.predict_proba(features_scaled)[0]
                confidence = pred_proba[pred_id]

                class_name = self.svm_class_names.get(pred_id, str(pred_id))

                print(f"      Predicción: ${class_name} pesos")
                print(f"      Confianza: {confidence:.2%}")

                # Verificar umbral
                if confidence >= min_confidence or self.show_all_detections_var.get():
                    labels.append(class_name)
                    confidences.append(confidence)
                    print(f"      ✓ Aceptado")
                else:
                    labels.append(None)
                    confidences.append(None)
                    print(f"      ✗ Rechazado (< {min_confidence:.0%})")

            # Filtrar detecciones válidas
            valid_detections = []
            valid_labels = []
            valid_confidences = []

            for det, lab, conf in zip(detections, labels, confidences):
                if lab and conf:
                    valid_detections.append(det)
                    valid_labels.append(lab)
                    valid_confidences.append(conf)

            print(f"\n   ✓ {len(valid_detections)} billetes clasificados")

            # PASO 4: VISUALIZACIÓN
            print(f"\n[4/4] Generando visualización...")
            self.update_recognition_results("\n[4/4] Visualizando...\n", "info")
            self.root.update()

            result_img = self.bill_detector.visualize_detections(
                img_cv, valid_detections, valid_labels, valid_confidences
            )

            # Convertir a PIL y mostrar
            result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            self.current_image = result_pil
            self.add_to_history()
            self.display_image_on_canvas()

            # Actualizar resultados finales
            self.recognition_result_text.config(state=tk.NORMAL)
            self.recognition_result_text.delete("1.0", tk.END)

            self.recognition_result_text.insert("1.0", "✓ RECONOCIMIENTO COMPLETO\n", "header")
            self.recognition_result_text.insert("end", "="*30 + "\n\n", "header")

            if valid_detections:
                self.recognition_result_text.insert("end",
                    f"Billetes detectados: {len(valid_detections)}\n\n",
                    "success")

                for i, (label, conf) in enumerate(zip(valid_labels, valid_confidences), 1):
                    tag = "success" if conf >= 0.8 else "warning" if conf >= 0.6 else "info"
                    self.recognition_result_text.insert("end",
                        f"#{i}: ${label} pesos\n", tag)
                    self.recognition_result_text.insert("end",
                        f"    Confianza: {conf:.1%}\n\n")

                # Estadísticas
                self.recognition_result_text.insert("end",
                    "\nEstadísticas:\n", "header")
                self.recognition_result_text.insert("end",
                    f"• Total detectados: {len(detections)}\n")
                self.recognition_result_text.insert("end",
                    f"• Clasificados: {len(valid_detections)}\n")
                avg_conf = np.mean(valid_confidences)
                self.recognition_result_text.insert("end",
                    f"• Confianza promedio: {avg_conf:.1%}\n")
            else:
                self.recognition_result_text.insert("end",
                    "⚠️  Ningún billete superó\n"
                    f"el umbral de {min_confidence:.0%}\n\n",
                    "warning")
                self.recognition_result_text.insert("end",
                    "Sugerencias:\n"
                    "• Baja el umbral de confianza\n"
                    "• Activa 'Mostrar todas'\n"
                    "• Mejora la calidad de imagen\n")

            self.recognition_result_text.config(state=tk.DISABLED)

            self.status_bar.config(text=f"✓ {len(valid_detections)} billetes reconocidos")

            print(f"\n{'='*60}")
            print(f"✓ PROCESO COMPLETADO")
            print(f"{'='*60}\n")

        except Exception as e:
            error_msg = f"Error en reconocimiento:\n{str(e)}"
            messagebox.showerror("Error", error_msg)
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ============= MÉTODOS DE OTSU =============

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

    # ============= MÉTODOS DE HARRIS =============

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

    # ============= MÉTODOS DE ESQUELETONIZACIÓN =============

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

    # ============= MÉTODOS DE PERÍMETRO =============

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

    # ============= MÉTODOS DE SEGMENTACIÓN =============

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

    # ============= MÉTODOS DE TEMPLATE MATCHING =============

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

    # ============= MÉTODOS DE DETECCIÓN DE BORDES =============

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

    def normalize_image(self, image):
        """Normalizar imagen a rango [0, 255]"""
        if image.dtype != np.float64:
            image = image.astype(np.float64)

        min_val = np.min(image)
        max_val = np.max(image)

        if max_val == min_val:
            return np.zeros_like(image, dtype=np.uint8)

        normalized = (image - min_val) / (max_val - min_val)
        return (normalized * 255).astype(np.uint8)

    def sobel_operator(self, image_array):
        """Operador de Sobel"""
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

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
            return image_array, None, None, None

    def apply_edge_detection(self):
        """Aplicar detección de bordes"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return

        try:
            gray_image = self.history[self.history_index].convert('L')
            image_array = np.array(gray_image, dtype=np.float32)
            operator = self.edge_operator_var.get()

            if operator == "sobel":
                magnitude, angle, grad_x, grad_y = self.sobel_operator(image_array)
            elif operator == "canny":
                result, angle, grad_x, grad_y = self.canny_operator(image_array)
                edge_image = Image.fromarray(result).convert('RGB')
                self.current_image = edge_image
                self.add_to_history()
                self.display_image_on_canvas()
                self.status_bar.config(text=f"PhotoEscom - Operador {operator} aplicado")
                return
            else:
                magnitude, angle, grad_x, grad_y = self.sobel_operator(image_array)

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
        self.apply_edge_detection()

    # ============= HERRAMIENTAS BÁSICAS =============

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
        elif filter_type == "detail":
            img = img.filter(ImageFilter.DETAIL)
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
        """Ajustar zoom"""
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

    def on_container_configure(self, event):
        """Evento de configuración de contenedor"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Evento de configuración de canvas"""
        self.canvas.itemconfig(self.canvas_window, width=event.width, height=event.height)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_resize(self, event):
        if event.widget == self.root:
            self.display_image_on_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoEditor(root)
    root.bind("<Configure>", app.on_resize)
    root.mainloop()