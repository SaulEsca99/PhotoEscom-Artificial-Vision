"""
Implementaciones de Segmentación y Template Matching
- Segmentación por umbralización, K-means, Watershed
- Template Matching manual y con OpenCV
"""

import numpy as np
import cv2
from scipy import ndimage
from sklearn.cluster import KMeans


class ImageSegmentation:
    """
    Diferentes métodos de segmentación de imágenes.
    """
    
    @staticmethod
    def threshold_segmentation(image_array, threshold=127, method='binary'):
        """
        Segmentación por umbralización simple.
        
        Args:
            image_array: Imagen de entrada
            threshold: Valor de umbral
            method: 'binary', 'binary_inv', 'truncate', 'tozero', 'tozero_inv'
            
        Returns:
            segmented: Imagen segmentada
            info: Información del proceso
        """
        # Convertir a escala de grises
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
        
        # Mapeo de métodos
        methods_map = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'truncate': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV
        }
        
        thresh_type = methods_map.get(method, cv2.THRESH_BINARY)
        
        # Aplicar umbralización
        _, segmented = cv2.threshold(gray, threshold, 255, thresh_type)
        
        # Convertir a RGB para visualización
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
        
        info = {
            'method': f'Threshold ({method})',
            'threshold_value': threshold,
            'unique_values': len(np.unique(segmented))
        }
        
        return segmented_rgb, info
    
    @staticmethod
    def kmeans_segmentation(image_array, n_clusters=3, max_iter=100):
        """
        Segmentación usando K-means clustering.
        
        Agrupa píxeles con colores similares en K clusters.
        
        Args:
            image_array: Imagen RGB
            n_clusters: Número de clusters (regiones)
            max_iter: Máximo de iteraciones
            
        Returns:
            segmented: Imagen segmentada
            info: Información de clusters
        """
        # Asegurar RGB
        if len(image_array.shape) == 2:
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image_array.copy()
        
        # Reshape: (height*width, 3)
        h, w = img_rgb.shape[:2]
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, 
                       random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Obtener centros de clusters
        centers = kmeans.cluster_centers_.astype(np.uint8)
        
        # Crear imagen segmentada con colores de centros
        segmented = centers[labels]
        segmented = segmented.reshape(h, w, 3)
        
        # Crear versión con etiquetas
        labels_image = labels.reshape(h, w)
        
        # Colorear cada cluster de forma distintiva
        colors = [
            [255, 0, 0],    # Rojo
            [0, 255, 0],    # Verde
            [0, 0, 255],    # Azul
            [255, 255, 0],  # Amarillo
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cian
            [128, 128, 128],# Gris
            [255, 128, 0],  # Naranja
        ]
        
        colored_labels = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(n_clusters):
            mask = labels_image == i
            colored_labels[mask] = colors[i % len(colors)]
        
        # Calcular estadísticas
        cluster_info = []
        for i in range(n_clusters):
            mask = labels_image == i
            num_pixels = np.sum(mask)
            percentage = (num_pixels / (h * w)) * 100
            
            cluster_info.append({
                'cluster_id': i,
                'center_color': centers[i].tolist(),
                'num_pixels': int(num_pixels),
                'percentage': round(percentage, 2)
            })
        
        info = {
            'method': 'K-means Clustering',
            'n_clusters': n_clusters,
            'iterations': kmeans.n_iter_,
            'inertia': kmeans.inertia_,
            'clusters': cluster_info,
            'colored_labels': colored_labels
        }
        
        return segmented, info
    
    @staticmethod
    def watershed_segmentation(image_array, marker_method='auto'):
        """
        Segmentación usando el algoritmo Watershed.
        
        Trata la imagen como una superficie topográfica donde los bordes
        son crestas y las regiones homogéneas son valles.
        
        Args:
            image_array: Imagen de entrada
            marker_method: 'auto' o 'manual'
            
        Returns:
            segmented: Imagen segmentada
            info: Información del proceso
        """
        # Convertir a escala de grises
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            img_rgb = image_array.copy()
        else:
            gray = image_array.copy()
            img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Aplicar umbral Otsu
        _, binary = cv2.threshold(gray, 0, 255, 
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening para eliminar ruido
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Dilatación para encontrar fondo seguro
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Distance transform para encontrar primer plano seguro
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 
                                   255, 0)
        
        # Región desconocida
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Etiquetar marcadores
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Agregar 1 a todas las etiquetas para que el fondo sea 1, no 0
        markers = markers + 1
        
        # Marcar región desconocida como 0
        markers[unknown == 255] = 0
        
        # Aplicar watershed
        markers = cv2.watershed(img_rgb, markers)
        
        # Crear imagen segmentada
        segmented = img_rgb.copy()
        
        # Marcar bordes en rojo
        segmented[markers == -1] = [255, 0, 0]
        
        # Colorear regiones
        num_regions = len(np.unique(markers)) - 2  # -1 para bordes, -1 para fondo
        
        # Crear mapa de colores aleatorio para cada región
        colors = np.random.randint(0, 255, size=(num_regions + 2, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Fondo negro
        
        colored_regions = colors[markers]
        
        # Mezclar con imagen original
        alpha = 0.5
        segmented_blend = cv2.addWeighted(img_rgb, alpha, colored_regions, 1-alpha, 0)
        
        info = {
            'method': 'Watershed',
            'num_regions': int(num_regions),
            'markers_range': (int(markers.min()), int(markers.max())),
            'pure_colored': colored_regions,
            'blended': segmented_blend
        }
        
        return segmented, info
    
    @staticmethod
    def region_growing(image_array, seed_point=None, threshold=10):
        """
        Segmentación por crecimiento de regiones.
        
        Comienza desde un punto semilla y agrega píxeles vecinos
        con intensidades similares.
        
        Args:
            image_array: Imagen en escala de grises
            seed_point: Punto inicial (x, y) o None para centro
            threshold: Diferencia máxima de intensidad
            
        Returns:
            segmented: Imagen segmentada
            info: Información del proceso
        """
        # Convertir a escala de grises
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
        
        h, w = gray.shape
        
        # Punto semilla: centro si no se especifica
        if seed_point is None:
            seed_point = (w // 2, h // 2)
        
        x_seed, y_seed = seed_point
        
        # Inicializar
        segmented = np.zeros_like(gray)
        visited = np.zeros_like(gray, dtype=bool)
        
        # Valor del punto semilla
        seed_value = gray[y_seed, x_seed]
        
        # Cola de píxeles a procesar
        queue = [(x_seed, y_seed)]
        visited[y_seed, x_seed] = True
        
        # 8-conectividad
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
        
        pixels_added = 0
        
        while queue:
            x, y = queue.pop(0)
            
            # Si está dentro del umbral, agregar a región
            if abs(int(gray[y, x]) - int(seed_value)) <= threshold:
                segmented[y, x] = 255
                pixels_added += 1
                
                # Agregar vecinos no visitados
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
        
        # Visualización
        result_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result_rgb[segmented > 0] = [0, 255, 0]  # Verde para región
        
        # Marcar punto semilla
        cv2.circle(result_rgb, seed_point, 5, (255, 0, 0), -1)
        
        info = {
            'method': 'Region Growing',
            'seed_point': seed_point,
            'seed_value': int(seed_value),
            'threshold': threshold,
            'pixels_in_region': int(pixels_added),
            'percentage': round((pixels_added / (h * w)) * 100, 2)
        }
        
        return result_rgb, info


class TemplateMatching:
    """
    Búsqueda de un template (plantilla) dentro de una imagen.
    """
    
    @staticmethod
    def match_template_manual(image_array, template_array, method='ssd'):
        """
        Template matching implementado manualmente.
        
        Args:
            image_array: Imagen donde buscar
            template_array: Template a buscar
            method: 'ssd' (Sum of Squared Differences) o 'ncc' (Normalized Cross-Correlation)
            
        Returns:
            result_image: Imagen con ubicación del template marcada
            match_info: Información del mejor match
        """
        # Convertir a escala de grises
        if len(image_array.shape) > 2:
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
            image_rgb = image_array.copy()
        else:
            image = image_array.astype(np.float32)
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        if len(template_array.shape) > 2:
            template = cv2.cvtColor(template_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            template = template_array.astype(np.float32)
        
        h_img, w_img = image.shape
        h_tmp, w_tmp = template.shape
        
        # Verificar que template cabe en imagen
        if h_tmp > h_img or w_tmp > w_img:
            raise ValueError("Template es más grande que la imagen")
        
        # Mapa de resultados
        result_h = h_img - h_tmp + 1
        result_w = w_img - w_tmp + 1
        result_map = np.zeros((result_h, result_w), dtype=np.float32)
        
        if method == 'ssd':
            # Sum of Squared Differences
            best_score = float('inf')
            best_location = (0, 0)
            
            for y in range(result_h):
                for x in range(result_w):
                    # Extraer región de la imagen
                    region = image[y:y+h_tmp, x:x+w_tmp]
                    
                    # Calcular SSD
                    diff = region - template
                    ssd = np.sum(diff ** 2)
                    
                    result_map[y, x] = ssd
                    
                    # Actualizar mejor match (menor SSD)
                    if ssd < best_score:
                        best_score = ssd
                        best_location = (x, y)
            
            # Normalizar para visualización
            result_map_normalized = 1 - (result_map / result_map.max())
            
        else:  # ncc - Normalized Cross-Correlation
            # Normalizar template
            template_mean = np.mean(template)
            template_std = np.std(template)
            template_norm = (template - template_mean) / (template_std + 1e-10)
            
            best_score = -1
            best_location = (0, 0)
            
            for y in range(result_h):
                for x in range(result_w):
                    # Extraer región
                    region = image[y:y+h_tmp, x:x+w_tmp]
                    
                    # Normalizar región
                    region_mean = np.mean(region)
                    region_std = np.std(region)
                    region_norm = (region - region_mean) / (region_std + 1e-10)
                    
                    # Calcular correlación
                    ncc = np.sum(region_norm * template_norm) / template.size
                    
                    result_map[y, x] = ncc
                    
                    # Actualizar mejor match (mayor NCC)
                    if ncc > best_score:
                        best_score = ncc
                        best_location = (x, y)
            
            result_map_normalized = result_map
        
        # Dibujar rectángulo en la ubicación encontrada
        x_best, y_best = best_location
        result_image = image_rgb.copy()
        
        cv2.rectangle(result_image, 
                     (x_best, y_best), 
                     (x_best + w_tmp, y_best + h_tmp),
                     (0, 255, 0), 2)
        
        # Marcar centro
        center_x = x_best + w_tmp // 2
        center_y = y_best + h_tmp // 2
        cv2.circle(result_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # Añadir texto
        cv2.putText(result_image, f"Match: {best_score:.2f}", 
                   (x_best, y_best - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        match_info = {
            'method': f'Template Matching Manual ({method.upper()})',
            'best_location': best_location,
            'best_score': float(best_score),
            'template_size': (w_tmp, h_tmp),
            'result_map': (result_map_normalized * 255).astype(np.uint8)
        }
        
        return result_image, match_info
    
    @staticmethod
    def match_template_opencv(image_array, template_array, method='ccoeff_normed'):
        """
        Template matching usando OpenCV.
        
        Args:
            image_array: Imagen donde buscar
            template_array: Template a buscar
            method: Método de OpenCV
                - 'sqdiff': cv2.TM_SQDIFF
                - 'sqdiff_normed': cv2.TM_SQDIFF_NORMED
                - 'ccorr': cv2.TM_CCORR
                - 'ccorr_normed': cv2.TM_CCORR_NORMED
                - 'ccoeff': cv2.TM_CCOEFF
                - 'ccoeff_normed': cv2.TM_CCOEFF_NORMED (recomendado)
                
        Returns:
            result_image: Imagen con template marcado
            match_info: Información del match
        """
        # Preparar imágenes
        if len(image_array.shape) > 2:
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            image_rgb = image_array.copy()
        else:
            image = image_array.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if len(template_array.shape) > 2:
            template = cv2.cvtColor(template_array, cv2.COLOR_RGB2GRAY)
        else:
            template = template_array.copy()
        
        # Mapeo de métodos
        methods_map = {
            'sqdiff': cv2.TM_SQDIFF,
            'sqdiff_normed': cv2.TM_SQDIFF_NORMED,
            'ccorr': cv2.TM_CCORR,
            'ccorr_normed': cv2.TM_CCORR_NORMED,
            'ccoeff': cv2.TM_CCOEFF,
            'ccoeff_normed': cv2.TM_CCOEFF_NORMED
        }
        
        cv_method = methods_map.get(method, cv2.TM_CCOEFF_NORMED)
        
        # Aplicar template matching
        result = cv2.matchTemplate(image, template, cv_method)
        
        # Encontrar mejor ubicación
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Para métodos SQDIFF, el mejor match es el mínimo
        if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_loc = min_loc
            best_val = min_val
        else:
            best_loc = max_loc
            best_val = max_val
        
        h_tmp, w_tmp = template.shape
        
        # Dibujar rectángulo
        result_image = image_rgb.copy()
        top_left = best_loc
        bottom_right = (top_left[0] + w_tmp, top_left[1] + h_tmp)
        
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        
        # Marcar centro
        center_x = top_left[0] + w_tmp // 2
        center_y = top_left[1] + h_tmp // 2
        cv2.circle(result_image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # Texto
        cv2.putText(result_image, f"Score: {best_val:.3f}", 
                   (top_left[0], top_left[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Normalizar result map para visualización
        result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        match_info = {
            'method': f'Template Matching OpenCV ({method})',
            'best_location': best_loc,
            'best_score': float(best_val),
            'template_size': (w_tmp, h_tmp),
            'min_val': float(min_val),
            'max_val': float(max_val),
            'result_map': result_normalized
        }
        
        return result_image, match_info
    
    @staticmethod
    def multi_scale_matching(image_array, template_array, scales=None):
        """
        Template matching a múltiples escalas.
        
        Busca el template a diferentes tamaños.
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        best_match = None
        best_score = -1
        best_scale = 1.0
        
        h_tmp, w_tmp = template_array.shape[:2]
        
        for scale in scales:
            # Redimensionar template
            new_w = int(w_tmp * scale)
            new_h = int(h_tmp * scale)
            
            if new_w < 10 or new_h < 10:
                continue
            
            scaled_template = cv2.resize(template_array, (new_w, new_h))
            
            try:
                # Buscar con esta escala
                result_img, info = TemplateMatching.match_template_opencv(
                    image_array, scaled_template, method='ccoeff_normed'
                )
                
                score = info['best_score']
                
                if score > best_score:
                    best_score = score
                    best_match = result_img
                    best_scale = scale
            except:
                continue
        
        info = {
            'method': 'Multi-Scale Template Matching',
            'best_scale': best_scale,
            'best_score': best_score,
            'scales_tested': scales
        }
        
        return best_match, info


# Funciones auxiliares para GUI
def apply_segmentation_to_image(pil_image, method='kmeans', **kwargs):
    """Aplica segmentación a imagen PIL."""
    from PIL import Image
    
    img_array = np.array(pil_image)
    
    if method == 'threshold':
        result, info = ImageSegmentation.threshold_segmentation(img_array, **kwargs)
    elif method == 'kmeans':
        result, info = ImageSegmentation.kmeans_segmentation(img_array, **kwargs)
    elif method == 'watershed':
        result, info = ImageSegmentation.watershed_segmentation(img_array, **kwargs)
    elif method == 'region_growing':
        result, info = ImageSegmentation.region_growing(img_array, **kwargs)
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    result_pil = Image.fromarray(result)
    return result_pil, info


def apply_template_matching_to_image(pil_image, pil_template, method='opencv', **kwargs):
    """Aplica template matching."""
    from PIL import Image
    
    img_array = np.array(pil_image)
    template_array = np.array(pil_template)
    
    if method == 'manual':
        result, info = TemplateMatching.match_template_manual(
            img_array, template_array, **kwargs
        )
    elif method == 'opencv':
        result, info = TemplateMatching.match_template_opencv(
            img_array, template_array, **kwargs
        )
    elif method == 'multiscale':
        result, info = TemplateMatching.multi_scale_matching(
            img_array, template_array, **kwargs
        )
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    result_pil = Image.fromarray(result)
    return result_pil, info


if __name__ == "__main__":
    print("Módulo de segmentación y template matching cargado.")
    print("\nMétodos disponibles:")
    print("1. ImageSegmentation - Threshold, K-means, Watershed, Region Growing")
    print("2. TemplateMatching - Manual (SSD/NCC), OpenCV, Multi-Scale")
