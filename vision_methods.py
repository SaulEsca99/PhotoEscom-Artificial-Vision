"""
Implementaciones de métodos de Visión por Computadora
- Método de Otsu
- Detección de esquinas de Harris (manual y OpenCV)
"""

import numpy as np
import cv2
from scipy import ndimage


class OtsuThreshold:
    """
    Implementación del método de Otsu para umbralización automática.
    
    El método de Otsu encuentra el umbral óptimo que maximiza la varianza
    entre clases (foreground y background).
    """
    
    @staticmethod
    def calculate_threshold(image_array):
        """
        Calcula el umbral óptimo usando el método de Otsu.
        
        Args:
            image_array: Imagen en escala de grises (numpy array)
            
        Returns:
            threshold: Valor de umbral óptimo
            result_image: Imagen binarizada
            metrics: Diccionario con métricas del proceso
        """
        # Asegurar que es escala de grises
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.astype(np.uint8)
        
        # Calcular histograma
        hist, bins = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        
        # Normalizar histograma (probabilidades)
        total_pixels = gray.size
        prob_hist = hist / total_pixels
        
        # Inicializar variables
        max_variance = 0
        optimal_threshold = 0
        
        # Probar todos los posibles umbrales (0-255)
        for t in range(256):
            # Clase 0: píxeles con intensidad [0, t]
            w0 = np.sum(prob_hist[:t+1])
            
            # Clase 1: píxeles con intensidad [t+1, 255]
            w1 = np.sum(prob_hist[t+1:])
            
            # Evitar división por cero
            if w0 == 0 or w1 == 0:
                continue
            
            # Media de clase 0
            mu0 = np.sum(np.arange(t+1) * prob_hist[:t+1]) / w0
            
            # Media de clase 1
            mu1 = np.sum(np.arange(t+1, 256) * prob_hist[t+1:]) / w1
            
            # Varianza entre clases
            variance_between = w0 * w1 * (mu0 - mu1) ** 2
            
            # Actualizar si encontramos mayor varianza
            if variance_between > max_variance:
                max_variance = variance_between
                optimal_threshold = t
        
        # Aplicar umbralización
        result_image = np.where(gray > optimal_threshold, 255, 0).astype(np.uint8)
        
        # Métricas
        metrics = {
            'threshold': optimal_threshold,
            'max_variance': max_variance,
            'foreground_pixels': np.sum(result_image == 255),
            'background_pixels': np.sum(result_image == 0)
        }
        
        return optimal_threshold, result_image, metrics
    
    @staticmethod
    def apply_otsu_opencv(image_array):
        """
        Aplica el método de Otsu usando OpenCV.
        
        Args:
            image_array: Imagen en escala de grises
            
        Returns:
            threshold: Umbral calculado
            result_image: Imagen binarizada
        """
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.astype(np.uint8)
        
        # Aplicar Otsu con OpenCV
        threshold, result_image = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return threshold, result_image


class HarrisCornerDetector:
    """
    Implementación de detección de esquinas de Harris.
    Detecta puntos de interés (esquinas) en imágenes.
    """
    
    @staticmethod
    def detect_corners_manual(image_array, k=0.04, threshold=0.01, window_size=3):
        """
        Implementación manual del detector de Harris.
        
        El algoritmo de Harris detecta esquinas basándose en la matriz de
        estructura local y la función de respuesta R = det(M) - k*trace(M)^2
        
        Args:
            image_array: Imagen en escala de grises
            k: Parámetro de sensibilidad de Harris (típicamente 0.04-0.06)
            threshold: Umbral para considerar una esquina
            window_size: Tamaño de ventana para calcular derivadas
            
        Returns:
            corners: Imagen con esquinas marcadas
            response: Mapa de respuesta de Harris
            corner_coords: Coordenadas de las esquinas detectadas
        """
        # Convertir a escala de grises si es necesario
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
        
        gray = gray.astype(np.float32)
        
        # Paso 1: Calcular derivadas usando Sobel
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Paso 2: Calcular productos de derivadas
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Paso 3: Aplicar ventana Gaussiana (suavizado)
        sigma = window_size / 6.0  # Regla empírica
        kernel_size = window_size * 2 + 1
        
        Sxx = cv2.GaussianBlur(Ixx, (kernel_size, kernel_size), sigma)
        Syy = cv2.GaussianBlur(Iyy, (kernel_size, kernel_size), sigma)
        Sxy = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), sigma)
        
        # Paso 4: Calcular respuesta de Harris para cada píxel
        # R = det(M) - k * trace(M)^2
        # donde M = [[Sxx, Sxy], [Sxy, Syy]]
        
        det_M = Sxx * Syy - Sxy * Sxy
        trace_M = Sxx + Syy
        
        response = det_M - k * (trace_M ** 2)
        
        # Paso 5: Normalizar respuesta
        response_normalized = response.copy()
        if response.max() > 0:
            response_normalized = response / response.max()
        
        # Paso 6: Aplicar umbralización
        threshold_value = threshold * response.max()
        corner_mask = response > threshold_value
        
        # Paso 7: Supresión no máxima (mantener solo máximos locales)
        corner_mask = HarrisCornerDetector._non_maximum_suppression(
            response, corner_mask, window_size
        )
        
        # Obtener coordenadas de esquinas
        corner_coords = np.argwhere(corner_mask)
        
        # Crear imagen de visualización
        corners_image = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Marcar esquinas en rojo
        for coord in corner_coords:
            y, x = coord
            cv2.circle(corners_image, (x, y), 3, (255, 0, 0), -1)
        
        return corners_image, response_normalized, corner_coords
    
    @staticmethod
    def _non_maximum_suppression(response, corner_mask, window_size):
        """
        Aplica supresión no máxima para mantener solo máximos locales.
        
        Args:
            response: Mapa de respuesta de Harris
            corner_mask: Máscara binaria de esquinas candidatas
            window_size: Tamaño de ventana para buscar máximos
            
        Returns:
            suppressed_mask: Máscara con solo máximos locales
        """
        from scipy.ndimage import maximum_filter
        
        # Encontrar máximos locales
        local_max = maximum_filter(response, size=window_size)
        
        # Mantener solo píxeles que son máximos locales
        suppressed_mask = (response == local_max) & corner_mask
        
        return suppressed_mask
    
    @staticmethod
    def detect_corners_opencv(image_array, block_size=2, ksize=3, k=0.04, 
                             threshold=0.01):
        """
        Detección de esquinas usando la implementación de OpenCV.
        
        Args:
            image_array: Imagen en escala de grises
            block_size: Tamaño del vecindario
            ksize: Tamaño de kernel para Sobel
            k: Parámetro de Harris
            threshold: Umbral relativo (0-1)
            
        Returns:
            corners_image: Imagen con esquinas marcadas
            response: Mapa de respuesta de Harris
            corner_coords: Coordenadas de las esquinas
        """
        # Convertir a escala de grises
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
        
        gray = np.float32(gray)
        
        # Detectar esquinas con Harris
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        
        # Dilatar para marcar las esquinas
        dst_dilated = cv2.dilate(dst, None)
        
        # Umbralización
        threshold_value = threshold * dst.max()
        corner_mask = dst > threshold_value
        
        # Obtener coordenadas
        corner_coords = np.argwhere(corner_mask)
        
        # Crear imagen de visualización
        corners_image = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        corners_image[dst > threshold_value] = [0, 0, 255]  # Rojo
        
        # Normalizar respuesta para visualización
        response_normalized = dst / dst.max() if dst.max() > 0 else dst
        
        return corners_image, response_normalized, corner_coords
    
    @staticmethod
    def compare_implementations(image_array, k=0.04, threshold=0.01):
        """
        Compara implementación manual vs OpenCV.
        
        Returns:
            dict con resultados de ambas implementaciones
        """
        # Manual
        corners_manual, response_manual, coords_manual = \
            HarrisCornerDetector.detect_corners_manual(
                image_array, k=k, threshold=threshold
            )
        
        # OpenCV
        corners_opencv, response_opencv, coords_opencv = \
            HarrisCornerDetector.detect_corners_opencv(
                image_array, k=k, threshold=threshold
            )
        
        return {
            'manual': {
                'image': corners_manual,
                'response': response_manual,
                'corners': coords_manual,
                'count': len(coords_manual)
            },
            'opencv': {
                'image': corners_opencv,
                'response': response_opencv,
                'corners': coords_opencv,
                'count': len(coords_opencv)
            }
        }


# Funciones auxiliares para integración con la GUI
def apply_otsu_to_image(pil_image):
    """
    Aplica Otsu a una imagen PIL.
    
    Returns:
        result_pil: Imagen PIL con resultado
        metrics: Diccionario con información
    """
    from PIL import Image
    
    # Convertir PIL a numpy
    img_array = np.array(pil_image)
    
    # Aplicar Otsu
    threshold, result, metrics = OtsuThreshold.calculate_threshold(img_array)
    
    # Convertir resultado a PIL
    result_pil = Image.fromarray(result)
    
    # Añadir información adicional
    metrics['method'] = 'Otsu Manual'
    
    return result_pil, metrics


def apply_harris_to_image(pil_image, method='manual', k=0.04, threshold=0.01):
    """
    Aplica detección de Harris a una imagen PIL.
    
    Args:
        pil_image: Imagen PIL
        method: 'manual' o 'opencv'
        k: Parámetro de sensibilidad
        threshold: Umbral para detección
        
    Returns:
        result_pil: Imagen PIL con esquinas marcadas
        info: Diccionario con información
    """
    from PIL import Image
    
    # Convertir PIL a numpy
    img_array = np.array(pil_image)
    
    if method == 'manual':
        corners_img, response, coords = \
            HarrisCornerDetector.detect_corners_manual(
                img_array, k=k, threshold=threshold
            )
    else:  # opencv
        corners_img, response, coords = \
            HarrisCornerDetector.detect_corners_opencv(
                img_array, k=k, threshold=threshold
            )
    
    # Convertir a PIL
    result_pil = Image.fromarray(corners_img)
    
    info = {
        'method': f'Harris Corner Detection ({method})',
        'corners_detected': len(coords),
        'k_parameter': k,
        'threshold': threshold
    }
    
    return result_pil, info


if __name__ == "__main__":
    # Pruebas básicas
    print("Módulo de métodos de visión por computadora cargado correctamente.")
    print("\nMétodos disponibles:")
    print("1. OtsuThreshold - Umbralización automática")
    print("2. HarrisCornerDetector - Detección de esquinas")
