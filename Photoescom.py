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

        # Panel de herramientas izquierdo
        tools_notebook = ttk.Notebook(main_panel, width=350)
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

        # Panel de visualización
        self.image_frame = ttk.Frame(main_panel)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Crear lienzo
        self.create_image_canvas()

        # Llenar pestañas
        self.create_basic_tools_panel(basic_tools_frame)
        self.create_transform_panel(transform_frame)
        self.create_adjustments_panel(adjust_frame)
        self.create_filters_panel(filter_frame)
        self.create_edge_detection_panel(edge_frame)

        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="PhotoEscom - Listo. Cargue una imagen para comenzar")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ============= MÉTODOS DE DETECCIÓN DE BORDES SEGÚN PDF =============

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

    # ===== OPERADORES DE PRIMERA DERIVADA (Sección 6.3) =====

    def gradient_operator(self, image_array):
        """
        Gradiente básico - Diferencias finitas (PDF pág 2)
        Gx = [f(x+1,y) - f(x-1,y)] / 2
        Gy = [f(x,y+1) - f(x,y-1)] / 2
        """
        grad_x = np.zeros_like(image_array, dtype=np.float64)
        grad_y = np.zeros_like(image_array, dtype=np.float64)

        # Diferencias finitas centradas
        grad_x[:, 1:-1] = (image_array[:, 2:] - image_array[:, :-2]) / 2.0
        grad_y[1:-1, :] = (image_array[2:, :] - image_array[:-2, :]) / 2.0

        # Bordes
        grad_x[:, 0] = image_array[:, 1] - image_array[:, 0]
        grad_x[:, -1] = image_array[:, -1] - image_array[:, -2]
        grad_y[0, :] = image_array[1, :] - image_array[0, :]
        grad_y[-1, :] = image_array[-1, :] - image_array[-2, :]

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def sobel_operator(self, image_array):
        """
        Operador de Sobel (PDF pág 5-7)
        Ecuación (6.6): Gx = (z3 + 2z6 + z9) - (z1 + 2z4 + z7)
                        Gy = (z1 + 2z2 + z3) - (z7 + 2z8 + z9)
        """
        # Máscaras exactas del PDF (Figura 6.4)
        gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)

        gy = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def prewitt_operator(self, image_array):
        """
        Operador de Prewitt (PDF pág 7-8)
        Similar a Sobel pero con coeficientes uniformes
        """
        # Máscaras del PDF (Figura 6.7)
        gx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float64)

        gy = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]], dtype=np.float64)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    def roberts_operator(self, image_array):
        """
        Operador de Roberts - CORREGIDO
        Ecuación (6.7): D1 = f(x,y) - f(x-1,y-1)
                        D2 = f(x,y-1) - f(x-1,y)
        Máscaras 2x2 del PDF (Figura 6.9)
        """
        # Usar convolución con padding para máscaras 2x2
        h, w = image_array.shape
        grad_x = np.zeros_like(image_array, dtype=np.float64)
        grad_y = np.zeros_like(image_array, dtype=np.float64)

        # Aplicar máscaras de Roberts manualmente para evitar problemas de padding
        # D1 = f(x,y) - f(x-1,y-1)
        grad_x[1:, 1:] = image_array[1:, 1:] - image_array[:-1, :-1]

        # D2 = f(x,y-1) - f(x-1,y)
        grad_y[1:, :-1] = image_array[1:, :-1] - image_array[:-1, 1:]

        # Seleccionar forma según variable (Ecuación 6.8)
        if self.roberts_form_var.get() == "sqrt":
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        else:  # "abs"
            magnitude = np.abs(grad_x) + np.abs(grad_y)

        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    # ===== OPERADORES COMPASS (Sección 6.3.5 y 6.3.6) =====

    def kirsch_operator(self, image_array):
        """
        Máscaras de Kirsch - CORREGIDO según Figura 6.11
        La máscara base es k0 y se rota en incrementos de 45°
        """
        # Máscara base k0 según Figura 6.11 del libro
        # Las 8 máscaras se generan rotando los valores externos
        # manteniendo el centro en 0

        # Patrón: 5 en una dirección, -3 en el resto
        masks = []

        # k0 (Este): 5 5 5 en el lado derecho
        masks.append(np.array([[-3, -3, 5],
                               [-3, 0, 5],
                               [-3, -3, 5]], dtype=np.float64))

        # k1 (Noreste): 5 5 5 en diagonal superior derecha
        masks.append(np.array([[-3, 5, 5],
                               [-3, 0, 5],
                               [-3, -3, -3]], dtype=np.float64))

        # k2 (Norte): 5 5 5 arriba
        masks.append(np.array([[5, 5, 5],
                               [-3, 0, -3],
                               [-3, -3, -3]], dtype=np.float64))

        # k3 (Noroeste): 5 5 5 en diagonal superior izquierda
        masks.append(np.array([[5, 5, -3],
                               [5, 0, -3],
                               [-3, -3, -3]], dtype=np.float64))

        # k4 (Oeste): 5 5 5 en el lado izquierdo
        masks.append(np.array([[5, -3, -3],
                               [5, 0, -3],
                               [5, -3, -3]], dtype=np.float64))

        # k5 (Suroeste): 5 5 5 en diagonal inferior izquierda
        masks.append(np.array([[-3, -3, -3],
                               [5, 0, -3],
                               [5, 5, -3]], dtype=np.float64))

        # k6 (Sur): 5 5 5 abajo
        masks.append(np.array([[-3, -3, -3],
                               [-3, 0, -3],
                               [5, 5, 5]], dtype=np.float64))

        # k7 (Sureste): 5 5 5 en diagonal inferior derecha
        masks.append(np.array([[-3, -3, -3],
                               [-3, 0, 5],
                               [-3, 5, 5]], dtype=np.float64))

        responses = []
        for mask in masks:
            response = ndimage.convolve(image_array.astype(np.float64), mask)
            responses.append(response)

        # Máximo en cada posición
        magnitude = np.maximum.reduce(responses)

        # Ángulo correspondiente al máximo
        responses_stack = np.stack(responses, axis=-1)
        angle_indices = np.argmax(responses_stack, axis=-1)
        angle = angle_indices * 45.0

        return magnitude, np.deg2rad(angle), None, None

    def robinson_operator(self, image_array):
        """
        Máscaras de Robinson - CORREGIDO según Figura 6.13
        Máscara base r0 rotada sistemáticamente
        """
        # Máscara base r0 según Figura 6.13
        base = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)

        masks = []

        # Generar las 8 máscaras rotando la base
        # r0 (0°)
        masks.append(np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float64))

        # r1 (45°)
        masks.append(np.array([[0, 1, 2],
                               [-1, 0, 1],
                               [-2, -1, 0]], dtype=np.float64))

        # r2 (90°)
        masks.append(np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=np.float64))

        # r3 (135°)
        masks.append(np.array([[2, 1, 0],
                               [1, 0, -1],
                               [0, -1, -2]], dtype=np.float64))

        # r4 (180°)
        masks.append(np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=np.float64))

        # r5 (225°)
        masks.append(np.array([[0, -1, -2],
                               [1, 0, -1],
                               [2, 1, 0]], dtype=np.float64))

        # r6 (270°)
        masks.append(np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float64))

        # r7 (315°)
        masks.append(np.array([[-2, -1, 0],
                               [-1, 0, 1],
                               [0, 1, 2]], dtype=np.float64))

        responses = []
        for mask in masks:
            response = ndimage.convolve(image_array.astype(np.float64), mask)
            responses.append(response)

        magnitude = np.maximum.reduce(responses)
        responses_stack = np.stack(responses, axis=-1)
        angle_indices = np.argmax(responses_stack, axis=-1)
        angle = angle_indices * 45.0

        return magnitude, np.deg2rad(angle), None, None

    # ===== OPERADOR DE FREI-CHEN (Sección 6.3.7) =====

    def frei_chen_operator(self, image_array):
        """
        Máscaras de Frei-Chen - CORREGIDO según Figura 6.15
        9 máscaras formando base vectorial completa
        """
        sqrt2 = np.sqrt(2)

        # Las 9 máscaras exactas del PDF (Figura 6.15)
        masks = [
            # f1 - Borde promedio horizontal
            (1 / (2 * sqrt2)) * np.array([[1, sqrt2, 1],
                                          [0, 0, 0],
                                          [-1, -sqrt2, -1]], dtype=np.float64),

            # f2 - Borde promedio vertical
            (1 / (2 * sqrt2)) * np.array([[1, 0, -1],
                                          [sqrt2, 0, -sqrt2],
                                          [1, 0, -1]], dtype=np.float64),

            # f3 - Borde diagonal (/)
            (1 / (2 * sqrt2)) * np.array([[0, -1, sqrt2],
                                          [1, 0, -1],
                                          [-sqrt2, 1, 0]], dtype=np.float64),

            # f4 - Borde diagonal (\)
            (1 / (2 * sqrt2)) * np.array([[sqrt2, -1, 0],
                                          [-1, 0, 1],
                                          [0, 1, -sqrt2]], dtype=np.float64),

            # f5 - Línea horizontal
            0.5 * np.array([[0, 1, 0],
                            [-1, 0, -1],
                            [0, 1, 0]], dtype=np.float64),

            # f6 - Línea diagonal
            0.5 * np.array([[-1, 0, 1],
                            [0, 0, 0],
                            [1, 0, -1]], dtype=np.float64),

            # f7 - Gaussiano (pasa bajas)
            (1 / 6) * np.array([[1, -2, 1],
                                [-2, 4, -2],
                                [1, -2, 1]], dtype=np.float64),

            # f8 - Laplaciano
            (1 / 6) * np.array([[-2, 1, -2],
                                [1, 4, 1],
                                [-2, 1, -2]], dtype=np.float64),

            # f9 - Promedio (constante)
            (1 / 3) * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]], dtype=np.float64)
        ]

        # Proyecciones según ecuación (6.10)
        projections = []
        for mask in masks:
            projection = ndimage.convolve(image_array.astype(np.float64), mask)
            projections.append(projection)

        # Subespacio de bordes: f1-f4 (ecuación 6.13)
        edge_responses = [p ** 2 for p in projections[:4]]
        all_responses = [p ** 2 for p in projections]

        M = np.sum(edge_responses, axis=0)
        S = np.sum(all_responses, axis=0)

        # Evitar división por cero
        S_safe = np.where(S > 0, S, 1)

        # cos(θ) = sqrt(M/S)
        cos_theta = np.sqrt(np.clip(M / S_safe, 0, 1))

        # Magnitud de bordes
        magnitude = np.sqrt(M)

        return magnitude, cos_theta, None, None

    # ===== EXTENSIÓN DE OPERADORES (Sección 6.3.8) =====

    def extended_sobel_operator(self, image_array, size=7):
        """
        Operador de Sobel Extendido - CORREGIDO con 9x9 y 11x11
        Según Figura 6.18
        """
        if size == 3:
            return self.sobel_operator(image_array)

        # Máscaras extendidas según el patrón del libro
        if size == 5:
            # Máscara 5x5
            gx = np.array([[-1, -2, 0, 2, 1],
                           [-2, -3, 0, 3, 2],
                           [-3, -5, 0, 5, 3],
                           [-2, -3, 0, 3, 2],
                           [-1, -2, 0, 2, 1]], dtype=np.float64)
            gy = gx.T

        elif size == 7:
            # Máscara 7x7 según Figura 6.18
            gx = np.array([[-1, -1, -1, 0, 1, 1, 1],
                           [-1, -2, -2, 0, 2, 2, 1],
                           [-1, -2, -3, 0, 3, 2, 1],
                           [-1, -2, -3, 0, 3, 2, 1],
                           [-1, -2, -3, 0, 3, 2, 1],
                           [-1, -2, -2, 0, 2, 2, 1],
                           [-1, -1, -1, 0, 1, 1, 1]], dtype=np.float64)
            gy = gx.T

        elif size == 9:
            # Máscara 9x9 (extrapolada del patrón)
            gx = np.array([[-1, -1, -1, -1, 0, 1, 1, 1, 1],
                           [-1, -2, -2, -2, 0, 2, 2, 2, 1],
                           [-1, -2, -3, -3, 0, 3, 3, 2, 1],
                           [-1, -2, -3, -4, 0, 4, 3, 2, 1],
                           [-1, -2, -3, -4, 0, 4, 3, 2, 1],
                           [-1, -2, -3, -4, 0, 4, 3, 2, 1],
                           [-1, -2, -3, -3, 0, 3, 3, 2, 1],
                           [-1, -2, -2, -2, 0, 2, 2, 2, 1],
                           [-1, -1, -1, -1, 0, 1, 1, 1, 1]], dtype=np.float64)
            gy = gx.T

        elif size == 11:
            # Máscara 11x11 (extrapolada del patrón)
            gx = np.array([[-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1],
                           [-1, -2, -2, -2, -2, 0, 2, 2, 2, 2, 1],
                           [-1, -2, -3, -3, -3, 0, 3, 3, 3, 2, 1],
                           [-1, -2, -3, -4, -4, 0, 4, 4, 3, 2, 1],
                           [-1, -2, -3, -4, -5, 0, 5, 4, 3, 2, 1],
                           [-1, -2, -3, -4, -5, 0, 5, 4, 3, 2, 1],
                           [-1, -2, -3, -4, -5, 0, 5, 4, 3, 2, 1],
                           [-1, -2, -3, -4, -4, 0, 4, 4, 3, 2, 1],
                           [-1, -2, -3, -3, -3, 0, 3, 3, 3, 2, 1],
                           [-1, -2, -2, -2, -2, 0, 2, 2, 2, 2, 1],
                           [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]], dtype=np.float64)
            gy = gx.T
        else:
            return self.sobel_operator(image_array)

        grad_x = ndimage.convolve(image_array.astype(np.float64), gx)
        grad_y = ndimage.convolve(image_array.astype(np.float64), gy)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        return magnitude, angle, grad_x, grad_y

    # ===== ALGORITMO DE CANNY (Sección 6.3.9) =====

    def canny_operator(self, image_array):
        """Algoritmo de Canny completo"""
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
        """Adelgazamiento de bordes - Supresión no máxima"""
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
        """Histéresis de umbral con dos umbrales t1 y t2"""
        strong_edges = image > high_threshold
        weak_edges = (image >= low_threshold) & (image <= high_threshold)

        strong_dilated = binary_dilation(strong_edges, structure=np.ones((3, 3)))
        connected_weak = weak_edges & strong_dilated

        edges = strong_edges | connected_weak

        return edges.astype(np.float32)

    # ===== OPERADOR LAPLACIANO (Segunda Derivada) =====

    def laplacian_operator(self, image_array):
        """Laplaciano de la Gaussiana (LoG) - Segunda derivada"""
        sigma = self.sigma_var.get()

        smoothed = gaussian_filter(image_array.astype(np.float64), sigma=sigma)

        laplacian = ndimage.laplace(smoothed)

        zero_crossings = self.detect_zero_crossings(laplacian)

        return zero_crossings, None, None, None

    def detect_zero_crossings(self, laplacian):
        """Detectar cambios de signo (zero-crossings)"""
        zc = np.zeros_like(laplacian, dtype=np.uint8)

        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            rolled = np.roll(np.roll(laplacian, dx, axis=0), dy, axis=1)
            zc |= ((laplacian * rolled) < 0).astype(np.uint8)

        return zc * 255

    # ===== APLICAR DETECCIÓN CON UMBRALIZACIÓN =====

    def apply_edge_detection(self):
        """Aplicar detección de bordes con umbralización (Ecuación 6.5)"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return

        try:
            gray_image = self.history[self.history_index].convert('L')
            image_array = np.array(gray_image, dtype=np.float32)

            if image_array.size > 10000000:
                response = messagebox.askyesno(
                    "Imagen Grande",
                    "La imagen es muy grande. ¿Desea continuar? (Puede tardar varios segundos)"
                )
                if not response:
                    return

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
            messagebox.showerror("Error", f"Error al aplicar detección de bordes: {str(e)}")
            import traceback
            traceback.print_exc()

    def preview_edge_detection(self):
        """Vista previa rápida"""
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen cargada")
            return

        try:
            original_img = self.history[self.history_index]

            if max(original_img.size) > 800:
                ratio = 800 / max(original_img.size)
                new_size = (int(original_img.size[0] * ratio),
                            int(original_img.size[1] * ratio))
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
            messagebox.showerror("Error", f"Error en vista previa: {str(e)}")

    def create_edge_detection_panel(self, parent):
        canvas = tk.Canvas(parent, bg=self.frame_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Primera Derivada
        first_deriv_frame = ttk.LabelFrame(scrollable_frame, text="Primera Derivada", padding=10)
        first_deriv_frame.pack(fill=tk.X, pady=(0, 10))

        operators_first = [
            ("Gradiente Básico", "gradient"),
            ("Sobel", "sobel"),
            ("Prewitt", "prewitt"),
            ("Roberts", "roberts")
        ]

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

        ttk.Label(params_frame, text="Umbral T (Ec. 6.5):").grid(row=0, column=0, sticky=tk.W, pady=5)
        threshold_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.threshold_var)
        threshold_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        threshold_value = ttk.Label(params_frame, text="30", width=5)
        threshold_value.grid(row=0, column=2, padx=(5, 0), pady=5)

        ttk.Label(params_frame, text="Sigma (σ):").grid(row=1, column=0, sticky=tk.W, pady=5)
        sigma_scale = ttk.Scale(params_frame, from_=0.5, to=3.0, variable=self.sigma_var)
        sigma_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        sigma_value = ttk.Label(params_frame, text="1.0", width=5)
        sigma_value.grid(row=1, column=2, padx=(5, 0), pady=5)

        ttk.Label(params_frame, text="Canny t1:").grid(row=2, column=0, sticky=tk.W, pady=5)
        canny_low_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.canny_low_var)
        canny_low_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        canny_low_value = ttk.Label(params_frame, text="50", width=5)
        canny_low_value.grid(row=2, column=2, padx=(5, 0), pady=5)

        ttk.Label(params_frame, text="Canny t2:").grid(row=3, column=0, sticky=tk.W, pady=5)
        canny_high_scale = ttk.Scale(params_frame, from_=0, to=255, variable=self.canny_high_var)
        canny_high_scale.grid(row=3, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        canny_high_value = ttk.Label(params_frame, text="150", width=5)
        canny_high_value.grid(row=3, column=2, padx=(5, 0), pady=5)

        params_frame.columnconfigure(1, weight=1)

        # Visualización
        viz_frame = ttk.LabelFrame(scrollable_frame, text="Visualización", padding=10)
        viz_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(viz_frame, text="Mostrar |G| (magnitud)",
                        variable=self.show_magnitude_var).pack(anchor=tk.W)
        ttk.Checkbutton(viz_frame, text="Mostrar θ (ángulo)",
                        variable=self.show_angle_var).pack(anchor=tk.W)

        # Extensión
        ext_frame = ttk.LabelFrame(scrollable_frame, text="Extensión (Sobel)", padding=10)
        ext_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ext_frame, text="Tamaño máscara:").pack(anchor=tk.W)
        for size, text in [(3, "3x3"), (5, "5x5"), (7, "7x7"), (9, "9x9"), (11, "11x11")]:
            ttk.Radiobutton(ext_frame, text=text, value=size,
                            variable=self.extended_size_var).pack(anchor=tk.W)

        # Roberts
        roberts_frame = ttk.LabelFrame(scrollable_frame, text="Roberts", padding=10)
        roberts_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(roberts_frame, text="Forma 1: √(D1²+D2²)", value="sqrt",
                        variable=self.roberts_form_var).pack(anchor=tk.W)
        ttk.Radiobutton(roberts_frame, text="Forma 2: |D1|+|D2|", value="abs",
                        variable=self.roberts_form_var).pack(anchor=tk.W)

        # Información
        info_frame = ttk.LabelFrame(scrollable_frame, text="Información", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.operator_info = ttk.Label(info_frame, text="Seleccione un operador",
                                       wraplength=300, justify=tk.LEFT)
        self.operator_info.pack(anchor=tk.W)

        self.edge_operator_var.trace('w', self.update_operator_info)

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

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)

    def update_operator_info(self, *args):
        """Actualizar información del operador"""
        operator = self.edge_operator_var.get()

        info_texts = {
            "gradient": "Gradiente Básico: Diferencias finitas. Rápido pero sensible al ruido.",
            "sobel": "Sobel: Balance óptimo entre precisión y suavizado. Ec. (6.6)",
            "prewitt": "Prewitt: Similar a Sobel con coeficientes uniformes.",
            "roberts": "Roberts: Operador diagonal 2x2. Ec. (6.7) y (6.8)",
            "kirsch": "Kirsch: 8 máscaras direccionales. Toma máximo. Fig. 6.11",
            "robinson": "Robinson: Similar a Kirsch, máscara inicial diferente. Fig. 6.13",
            "frei_chen": "Frei-Chen: 9 máscaras - base vectorial completa. Ec. (6.10), (6.13)",
            "canny": "Canny: Algoritmo óptimo. 3 módulos: gradiente, supresión no máxima, histéresis.",
            "laplacian": "Laplaciano: Segunda derivada. Detecta zero-crossings."
        }

        self.operator_info.config(text=info_texts.get(operator, "Información no disponible"))

    # ============= MÉTODOS ORIGINALES DEL EDITOR =============

    def create_top_toolbar(self):
        toolbar = ttk.Frame(self.root, height=50)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 5))
        toolbar.pack_propagate(False)

        title_label = ttk.Label(toolbar, text="PhotoEscom", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=(10, 20))

        btn_load = ttk.Button(toolbar, text="📁 Cargar", command=self.load_image, width=12)
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_save = ttk.Button(toolbar, text="💾 Guardar", command=self.save_image, width=12)
        btn_save.pack(side=tk.LEFT, padx=5)

        separator = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        btn_undo = ttk.Button(toolbar, text="↶ Deshacer", command=self.undo, width=10)
        btn_undo.pack(side=tk.LEFT, padx=5)

        btn_redo = ttk.Button(toolbar, text="↷ Rehacer", command=self.redo, width=10)
        btn_redo.pack(side=tk.LEFT, padx=5)

        btn_reset = ttk.Button(toolbar, text="🔄 Restaurar", command=self.reset_image, width=12)
        btn_reset.pack(side=tk.LEFT, padx=5)

        separator2 = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator2.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        zoom_frame = ttk.Frame(toolbar)
        zoom_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔍 Ajustar", command=self.zoom_fit, width=8).pack(side=tk.LEFT, padx=2)

    def create_image_canvas(self):
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
        ttk.Label(parent, text="Rotación:").grid(row=0, column=0, sticky=tk.W, pady=5)
        rotate_scale = ttk.Scale(parent, from_=-180, to=180, variable=self.rotate_var,
                                 command=lambda e: self.preview_transform('rotate'))
        rotate_scale.grid(row=0, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        ttk.Entry(parent, textvariable=self.rotate_var, width=5).grid(row=0, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Escala X:").grid(row=1, column=0, sticky=tk.W, pady=5)
        scale_x_scale = ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_x_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_x_scale.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        ttk.Entry(parent, textvariable=self.scale_x_var, width=5).grid(row=1, column=2, padx=(5, 0), pady=5)

        ttk.Label(parent, text="Escala Y:").grid(row=2, column=0, sticky=tk.W, pady=5)
        scale_y_scale = ttk.Scale(parent, from_=0.1, to=5.0, variable=self.scale_y_var,
                                  command=lambda e: self.preview_transform('scale'))
        scale_y_scale.grid(row=2, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
        ttk.Entry(parent, textvariable=self.scale_y_var, width=5).grid(row=2, column=2, padx=(5, 0), pady=5)

        ttk.Button(parent, text="Aplicar", command=self.apply_transforms).grid(row=3, column=0, columnspan=3, pady=10)
        parent.columnconfigure(1, weight=1)

    def create_adjustments_panel(self, parent):
        adjustments = [
            ("Brillo:", self.brightness_var, 0.1, 3.0),
            ("Contraste:", self.contrast_var, 0.1, 3.0),
            ("Saturación:", self.saturation_var, 0.0, 3.0),
            ("Nitidez:", self.sharpen_var, 1.0, 5.0)
        ]

        value_labels = []
        for i, (label, var, min_val, max_val) in enumerate(adjustments):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var,
                              command=lambda e: self.apply_adjustments())
            scale.grid(row=i, column=1, sticky=tk.EW, pady=5, padx=(5, 0))
            value_label = ttk.Label(parent, text="1.0", width=5)
            value_label.grid(row=i, column=2, padx=(5, 0), pady=5)
            value_labels.append((var, value_label))

        ttk.Button(parent, text="Aplicar Ajustes", command=self.finalize_adjustments).grid(
            row=len(adjustments), column=0, columnspan=3, pady=10)

        parent.columnconfigure(1, weight=1)

        def update_values(*args):
            for var, label in value_labels:
                label.config(text=f"{var.get():.1f}")

        for var, _ in value_labels:
            var.trace('w', update_values)

    def create_filters_panel(self, parent):
        filters = [
            ("Original", "original"),
            ("Escala de Grises", "grayscale"),
            ("Sepia", "sepia"),
            ("Invertir", "invert"),
            ("Desenfoque", "blur"),
            ("Detalle", "detail"),
            ("Bordes", "edges"),
            ("Realce", "enhance")
        ]

        for i, (text, mode) in enumerate(filters):
            ttk.Radiobutton(parent, text=text, value=mode,
                            variable=self.filter_var, command=self.apply_filter).grid(
                row=i, column=0, sticky=tk.W, pady=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )

        if file_path:
            try:
                self.original_image = Image.open(file_path).convert('RGB')
                self.current_image = self.original_image.copy()
                self.filename = os.path.basename(file_path)

                self.history = [self.current_image.copy()]
                self.history_index = 0
                self.zoom_factor = 1.0

                self.reset_controls()
                self.display_image_on_canvas()
                self.update_info()

                self.status_bar.config(text=f"PhotoEscom - Imagen cargada: {self.filename}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")

    def reset_controls(self):
        self.rotate_var.set(0)
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.sharpen_var.set(1.0)
        self.filter_var.set("original")
        self.edge_operator_var.set("sobel")
        self.threshold_var.set(30.0)
        self.sigma_var.set(1.0)
        self.canny_low_var.set(50.0)
        self.canny_high_var.set(150.0)
        self.roberts_form_var.set("sqrt")
        self.show_magnitude_var.set(False)
        self.show_angle_var.set(False)
        self.extended_size_var.set(3)

    def update_info(self):
        if self.current_image:
            width, height = self.current_image.size
            info_text = f"""Archivo: {self.filename or 'N/A'}
Dimensiones: {width} x {height}
Historial: {self.history_index + 1}/{len(self.history)}"""
            self.info_label.config(text=info_text)
        else:
            self.info_label.config(text="Sin imagen cargada")

    def display_image_on_canvas(self):
        if self.current_image:
            width, height = self.current_image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)

            img = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(img)

            if hasattr(self, 'image_label'):
                self.image_label.config(image=self.display_image)
            else:
                self.image_label = ttk.Label(self.image_container, image=self.display_image)
                self.image_label.pack()

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def save_image(self):
        if not self.current_image:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")]
        )

        if file_path:
            try:
                self.current_image.save(file_path)
                messagebox.showinfo("Éxito", "Imagen guardada correctamente")
                self.status_bar.config(text=f"PhotoEscom - Guardado: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar: {str(e)}")

    def add_to_history(self):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(self.current_image.copy())
        self.history_index = len(self.history) - 1
        if len(self.history) > 20:
            self.history.pop(0)
            self.history_index -= 1
        self.update_info()

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.update_info()

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image = self.history[self.history_index].copy()
            self.display_image_on_canvas()
            self.update_info()

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.add_to_history()
            self.reset_controls()
            self.display_image_on_canvas()
            self.zoom_factor = 1.0

    def quick_rotate(self, angle):
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        img = img.rotate(angle, expand=True)
        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()

    def preview_transform(self, transform_type):
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        if transform_type == 'rotate':
            angle = self.rotate_var.get()
            img = img.rotate(angle, expand=True)
        elif transform_type == 'scale':
            scale_x = self.scale_x_var.get()
            scale_y = self.scale_y_var.get()
            width, height = img.size
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.current_image = img
        self.display_image_on_canvas()

    def apply_transforms(self):
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        angle = self.rotate_var.get()
        if angle != 0:
            img = img.rotate(angle, expand=True)
        scale_x = self.scale_x_var.get()
        scale_y = self.scale_y_var.get()
        if scale_x != 1.0 or scale_y != 1.0:
            width, height = img.size
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.current_image = img
        self.add_to_history()
        self.display_image_on_canvas()
        self.rotate_var.set(0)
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)

    def apply_flip(self, direction):
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

    def apply_adjustments(self):
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
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        self.current_image = img
        self.display_image_on_canvas()

    def finalize_adjustments(self):
        if not self.current_image:
            return
        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        saturation = self.saturation_var.get()
        sharpen = self.sharpen_var.get()
        if brightness != 1.0 or contrast != 1.0 or saturation != 1.0 or sharpen != 1.0:
            self.add_to_history()
            self.brightness_var.set(1.0)
            self.contrast_var.set(1.0)
            self.saturation_var.set(1.0)
            self.sharpen_var.set(1.0)

    def apply_filter(self):
        if not self.current_image:
            return
        img = self.history[self.history_index].copy()
        filter_type = self.filter_var.get()
        if filter_type == "grayscale":
            img = img.convert("L").convert("RGB")
        elif filter_type == "sepia":
            img_array = np.array(img)
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
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
        elif filter_type == "edges":
            img = img.filter(ImageFilter.FIND_EDGES)
        elif filter_type == "enhance":
            img = img.filter(ImageFilter.EDGE_ENHANCE)
        self.current_image = img
        self.display_image_on_canvas()
        if filter_type != "original":
            self.add_to_history()

    def zoom_in(self):
        if self.current_image:
            self.zoom_factor *= 1.2
            self.display_image_on_canvas()

    def zoom_out(self):
        if self.current_image:
            self.zoom_factor /= 1.2
            self.display_image_on_canvas()

    def zoom_fit(self):
        if self.current_image:
            self.zoom_factor = 1.0
            self.display_image_on_canvas()

    def on_container_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
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