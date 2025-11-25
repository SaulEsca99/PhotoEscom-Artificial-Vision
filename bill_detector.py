"""
DETECCIÓN DE BILLETES - VERSIÓN ANTI-RUIDO
Incluye filtros para ignorar objetos diminutos (basura/interfaz).
"""

import cv2
import numpy as np

class BillDetector:
    def __init__(self):
        self.min_aspect_ratio = 1.0
        self.max_aspect_ratio = 6.0

    def preprocess_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return denoised

    def detect_simple_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Umbralización para atrapar el billete
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Separación de objetos
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel_small, iterations=2)
        mask = cv2.dilate(mask, kernel_small, iterations=1)

        kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

        return mask

    def combine_detection_methods(self, image):
        # Para fondos blancos limpios, el threshold simple es el mejor
        mask = self.detect_simple_threshold(image)

        # Limpieza final de ruido (puntos aislados)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def filter_contours(self, contours, image_shape):
        """Filtra lo que no parece un billete por tamaño o forma"""
        h_img, w_img = image_shape[:2]
        image_area = h_img * w_img
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # --- FILTRO 1: TAMAÑO RELATIVO (NUEVO) ---
            # Si el objeto ocupa menos del 1% de la imagen, es basura/ruido
            if area < (image_area * 0.01):
                continue

            # Si ocupa más del 95%, es el fondo
            if area > (image_area * 0.95):
                continue

            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), angle = rect

            box_width = min(width, height)
            box_height = max(width, height)

            if box_width == 0: continue

            # --- FILTRO 2: DIMENSIONES MÍNIMAS ---
            # Un billete no puede medir menos de 20px de ancho
            if box_width < 20 or box_height < 20:
                continue

            aspect_ratio = box_height / box_width

            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue

            solidity = area / hull_area
            if solidity < 0.55:
                continue

            valid_contours.append(contour)

        return valid_contours

    def get_bounding_boxes(self, contours):
        boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)

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
        preprocessed = self.preprocess_image(image)
        mask = self.combine_detection_methods(preprocessed)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = self.filter_contours(contours, image.shape)
        detections = self.get_bounding_boxes(valid_contours)
        detections.sort(key=lambda x: x['area'], reverse=True)

        return detections

    def visualize_detections(self, image, detections, labels=None, confidences=None):
        result = image.copy()

        for i, detection in enumerate(detections):
            cv2.drawContours(result, [detection['rotated_box']], 0, (0, 255, 0), 2)

            x, y, w, h = detection['bbox']

            if labels and i < len(labels):
                label = labels[i]
                conf = confidences[i] if confidences and i < len(confidences) else 0.0

                if conf >= 0.8: color = (0, 255, 0)
                elif conf >= 0.6: color = (0, 255, 255)
                else: color = (0, 0, 255)

                text = f"${label} ({conf:.0%})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 2

                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

                # Asegurar que el texto no se salga de la imagen por arriba
                text_y = max(y - 5, th + 5)

                cv2.rectangle(result, (x, text_y - th - 5), (x + tw + 5, text_y + 5), color, -1)
                cv2.putText(result, text, (x + 2, text_y), font, scale, (0, 0, 0), thickness)

        return result