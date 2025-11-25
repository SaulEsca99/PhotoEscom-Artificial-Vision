"""
DETECCIÓN DE BILLETES SIN APIS EXTERNAS
Usa técnicas de visión por computadora puras
"""

import cv2
import numpy as np


class BillDetector:
    """
    Detecta billetes en imágenes usando:
    - Segmentación por color
    - Detección de bordes
    - Análisis de contornos
    - Filtrado por características geométricas
    """

    def __init__(self):
        # Parámetros de detección
        self.min_area = 5000  # Área mínima del billete en píxeles
        self.max_area = 500000  # Área máxima
        self.min_aspect_ratio = 1.5  # Los billetes son rectangulares
        self.max_aspect_ratio = 3.5

    def preprocess_image(self, image):
        """Preprocesamiento para mejorar detección"""
        # Mejorar contraste con CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Reducir ruido
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised

    def detect_by_edges(self, image):
        """Detección basada en bordes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detección de bordes con Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Dilatar para conectar bordes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Cerrar gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

        return closed

    def detect_by_color(self, image):
        """Detección basada en color (billetes tienen colores distintivos)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Máscara para excluir fondo blanco/negro
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])

        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # Invertir: queremos objetos de color (no blancos ni negros)
        mask_background = cv2.bitwise_or(mask_white, mask_black)
        mask_objects = cv2.bitwise_not(mask_background)

        # Limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)

        return mask_cleaned

    def combine_detection_methods(self, image):
        """Combina detección por bordes y color"""
        edge_mask = self.detect_by_edges(image)
        color_mask = self.detect_by_color(image)

        # Combinar máscaras (AND para ser más estrictos)
        combined = cv2.bitwise_and(edge_mask, color_mask)

        # Limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

        return combined

    def filter_contours(self, contours, image_shape):
        """Filtra contornos para quedarse solo con billetes probables"""
        h_img, w_img = image_shape[:2]
        image_area = h_img * w_img

        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filtro 1: Área razonable
            if area < self.min_area or area > self.max_area:
                continue

            # Filtro 2: No muy grande respecto a la imagen
            if area > 0.8 * image_area:
                continue

            # Filtro 3: Forma rectangular
            rect = cv2.minAreaRect(contour)
            box_width, box_height = rect[1]

            if box_width == 0 or box_height == 0:
                continue

            aspect_ratio = max(box_width, box_height) / min(box_width, box_height)

            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Filtro 4: Extent (qué tan lleno está el rectángulo delimitador)
            rect_area = box_width * box_height
            extent = area / rect_area if rect_area > 0 else 0

            if extent < 0.6:  # Los billetes llenan bien su rectángulo
                continue

            # Filtro 5: Solidez (qué tan convexo es)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            if solidity < 0.8:  # Los billetes son bastante sólidos
                continue

            valid_contours.append(contour)

        return valid_contours

    def get_bounding_boxes(self, contours):
        """Obtiene bounding boxes de los contornos"""
        boxes = []

        for contour in contours:
            # Rectángulo rotado (mejor para billetes inclinados)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # También guardar bounding box normal
            x, y, w, h = cv2.boundingRect(contour)

            boxes.append({
                'contour': contour,
                'rotated_box': box,
                'bbox': (x, y, w, h),
                'center': (int(x + w / 2), int(y + h / 2)),
                'area': cv2.contourArea(contour)
            })

        return boxes

    def detect_bills(self, image, debug=False):
        """
        Detecta billetes en la imagen

        Returns:
            list: Lista de diccionarios con información de cada billete detectado
        """
        # Preprocesar
        preprocessed = self.preprocess_image(image)

        # Detectar
        mask = self.combine_detection_methods(preprocessed)

        # Si debug, guardar máscara
        if debug:
            cv2.imwrite('debug_mask.png', mask)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"   Contornos encontrados: {len(contours)}")

        # Filtrar
        valid_contours = self.filter_contours(contours, image.shape)

        print(f"   Contornos válidos: {len(valid_contours)}")

        # Obtener bounding boxes
        detections = self.get_bounding_boxes(valid_contours)

        # Ordenar por área (de mayor a menor)
        detections.sort(key=lambda x: x['area'], reverse=True)

        return detections

    def extract_bill_crops(self, image, detections, margin=0.05):
        """
        Extrae recortes de los billetes detectados

        Args:
            image: Imagen original
            detections: Lista de detecciones
            margin: Margen adicional (porcentaje)

        Returns:
            list: Lista de imágenes recortadas
        """
        crops = []

        for detection in detections:
            x, y, w, h = detection['bbox']

            # Agregar margen
            margin_x = int(w * margin)
            margin_y = int(h * margin)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)

            crop = image[y1:y2, x1:x2]
            crops.append(crop)

        return crops

    def visualize_detections(self, image, detections, labels=None, confidences=None):
        """
        Dibuja las detecciones sobre la imagen

        Args:
            image: Imagen original
            detections: Lista de detecciones
            labels: Lista de etiquetas (opcional)
            confidences: Lista de confianzas (opcional)

        Returns:
            Imagen con detecciones dibujadas
        """
        result = image.copy()

        for i, detection in enumerate(detections):
            # Dibujar rectángulo rotado
            cv2.drawContours(result, [detection['rotated_box']], 0, (0, 255, 0), 3)

            # Dibujar bounding box normal
            x, y, w, h = detection['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Dibujar centro
            center = detection['center']
            cv2.circle(result, center, 5, (255, 0, 0), -1)

            # Agregar etiqueta si está disponible
            if labels and i < len(labels):
                label = labels[i]
                conf = confidences[i] if confidences and i < len(confidences) else None

                text = f"${label}"
                if conf:
                    text += f" ({conf:.0%})"

                # Fondo del texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Color según confianza
                if conf and conf >= 0.8:
                    color = (0, 255, 0)
                    bg_color = (0, 180, 0)
                elif conf and conf >= 0.6:
                    color = (0, 255, 255)
                    bg_color = (0, 180, 180)
                else:
                    color = (0, 165, 255)
                    bg_color = (0, 120, 200)

                # Dibujar
                padding = 10
                cv2.rectangle(result,
                              (x, y - th - baseline - padding * 2),
                              (x + tw + padding * 2, y),
                              bg_color, -1)

                cv2.rectangle(result,
                              (x, y - th - baseline - padding * 2),
                              (x + tw + padding * 2, y),
                              color, 3)

                cv2.putText(result, text,
                            (x + padding, y - baseline - padding),
                            font, font_scale, (255, 255, 255), thickness)

        return result


if __name__ == "__main__":
    # Prueba simple
    print("Módulo BillDetector cargado correctamente")
    print("\nUso:")
    print("  detector = BillDetector()")
    print("  detections = detector.detect_bills(image)")