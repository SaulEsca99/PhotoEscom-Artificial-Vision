"""
EJEMPLOS DE USO - Demostración de todas las funcionalidades

Este archivo contiene ejemplos de cómo usar cada una de las nuevas
funcionalidades implementadas.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Importar módulos
from vision_methods import OtsuThreshold, HarrisCornerDetector
from skeleton_perimeter import SkeletonizationMethods, PerimeterAnalysis
from segmentation_template import ImageSegmentation, TemplateMatching


def demo_otsu():
    """Demostración del método de Otsu"""
    print("=" * 60)
    print("DEMOSTRACIÓN: MÉTODO DE OTSU")
    print("=" * 60)
    
    # Crear imagen de prueba con dos regiones
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 200  # Región clara
    
    # Agregar ruido
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    print("\n1. Método Manual:")
    threshold, result, metrics = OtsuThreshold.calculate_threshold(img)
    print(f"   Umbral óptimo encontrado: {threshold}")
    print(f"   Varianza máxima: {metrics['max_variance']:.2f}")
    print(f"   Píxeles foreground: {metrics['foreground_pixels']}")
    print(f"   Píxeles background: {metrics['background_pixels']}")
    
    print("\n2. Método OpenCV:")
    threshold_cv, result_cv = OtsuThreshold.apply_otsu_opencv(img)
    print(f"   Umbral OpenCV: {threshold_cv:.2f}")
    
    print("\n✓ Ambos métodos deberían dar umbrales similares")
    print(f"  Diferencia: {abs(threshold - threshold_cv):.2f}")


def demo_harris():
    """Demostración de detección de Harris"""
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN: DETECCIÓN DE ESQUINAS DE HARRIS")
    print("=" * 60)
    
    # Crear imagen con esquinas
    img = np.zeros((300, 300), dtype=np.uint8)
    
    # Dibujar rectángulo (4 esquinas)
    cv2.rectangle(img, (50, 50), (250, 250), 255, 2)
    
    # Dibujar cruces (esquinas en forma de +)
    cv2.line(img, (150, 50), (150, 100), 255, 2)
    cv2.line(img, (125, 75), (175, 75), 255, 2)
    
    print("\n1. Detección Manual:")
    corners_manual, response_manual, coords_manual = \
        HarrisCornerDetector.detect_corners_manual(img, k=0.04, threshold=0.01)
    print(f"   Esquinas detectadas: {len(coords_manual)}")
    print(f"   Primeras 5 coordenadas:")
    for i, coord in enumerate(coords_manual[:5]):
        print(f"      {i+1}. (x={coord[1]}, y={coord[0]})")
    
    print("\n2. Detección OpenCV:")
    corners_opencv, response_opencv, coords_opencv = \
        HarrisCornerDetector.detect_corners_opencv(img, k=0.04, threshold=0.01)
    print(f"   Esquinas detectadas: {len(coords_opencv)}")
    
    print("\n3. Comparación:")
    results = HarrisCornerDetector.compare_implementations(img, k=0.04, threshold=0.01)
    print(f"   Manual: {results['manual']['count']} esquinas")
    print(f"   OpenCV: {results['opencv']['count']} esquinas")


def demo_skeleton():
    """Demostración de esqueletonización"""
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN: ESQUELETONIZACIÓN")
    print("=" * 60)
    
    # Crear forma simple para esqueletonizar
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 60, 255, -1)  # Círculo relleno
    
    print("\n1. Método Morfológico OpenCV:")
    skeleton1, iter1 = SkeletonizationMethods.morphological_skeleton(img, method='opencv')
    print(f"   Iteraciones: {iter1}")
    print(f"   Píxeles de esqueleto: {np.sum(skeleton1 > 0)}")
    
    print("\n2. Método Morfológico Manual:")
    skeleton2, iter2 = SkeletonizationMethods.morphological_skeleton(img, method='manual')
    print(f"   Iteraciones: {iter2}")
    print(f"   Píxeles de esqueleto: {np.sum(skeleton2 > 0)}")
    
    print("\n3. Zhang-Suen:")
    skeleton3, iter3 = SkeletonizationMethods.zhang_suen_skeleton(img)
    print(f"   Píxeles de esqueleto: {np.sum(skeleton3 > 0)}")
    
    print("\n4. Comparación de métodos:")
    results = SkeletonizationMethods.compare_methods(img)
    for key, data in results.items():
        iter_text = data['iterations'] if data['iterations'] >= 0 else "N/A"
        pixels = np.sum(data['skeleton'] > 0)
        print(f"   {data['name']}: {iter_text} iteraciones, {pixels} píxeles")


def demo_perimeter():
    """Demostración de análisis de perímetro"""
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN: ANÁLISIS DE PERÍMETRO")
    print("=" * 60)
    
    # Crear imagen con múltiples objetos
    img = np.zeros((300, 300), dtype=np.uint8)
    
    # Círculo
    cv2.circle(img, (75, 75), 40, 255, -1)
    
    # Rectángulo
    cv2.rectangle(img, (150, 50), (250, 150), 255, -1)
    
    # Triángulo
    pts = np.array([[75, 200], [150, 250], [0, 250]], np.int32)
    cv2.fillPoly(img, [pts], 255)
    
    print("\n1. Método OpenCV (Contours):")
    perimeter_img1, measurements1 = PerimeterAnalysis.calculate_perimeter(
        img, method='opencv'
    )
    print(f"   Objetos detectados: {measurements1['num_objects']}")
    print(f"   Perímetro total: {measurements1['total_perimeter']:.2f}")
    
    for i, obj in enumerate(measurements1['objects']):
        print(f"\n   Objeto {i}:")
        print(f"      Perímetro: {obj['perimeter']:.2f}")
        print(f"      Área: {obj['area']:.2f}")
        print(f"      Circularidad: {obj['circularity']:.3f}")
        print(f"      Centroide: {obj['centroid']}")
    
    print("\n2. Método Chain Code:")
    perimeter_img2, measurements2 = PerimeterAnalysis.calculate_perimeter(
        img, method='chain_code'
    )
    print(f"   Objetos detectados: {measurements2['num_objects']}")
    print(f"   Perímetro total: {measurements2['total_perimeter']:.2f}")
    
    print("\n3. Método Morfológico:")
    perimeter_img3, measurements3 = PerimeterAnalysis.calculate_perimeter(
        img, method='morphological'
    )
    print(f"   Objetos detectados: {measurements3['num_objects']}")
    print(f"   Píxeles de perímetro: {measurements3['total_perimeter_pixels']}")


def demo_segmentation():
    """Demostración de segmentación"""
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN: SEGMENTACIÓN")
    print("=" * 60)
    
    # Crear imagen con regiones de diferentes colores
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[0:100, 0:100] = [255, 0, 0]      # Rojo
    img[0:100, 100:200] = [0, 255, 0]    # Verde
    img[0:100, 200:300] = [0, 0, 255]    # Azul
    img[100:200, 0:150] = [255, 255, 0]  # Amarillo
    img[100:200, 150:300] = [255, 0, 255] # Magenta
    
    # Agregar ruido
    noise = np.random.normal(0, 15, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    print("\n1. Segmentación por Threshold:")
    result1, info1 = ImageSegmentation.threshold_segmentation(img, threshold=127)
    print(f"   Método: {info1['method']}")
    print(f"   Valores únicos: {info1['unique_values']}")
    
    print("\n2. Segmentación K-means:")
    result2, info2 = ImageSegmentation.kmeans_segmentation(img, n_clusters=5)
    print(f"   Método: {info2['method']}")
    print(f"   Clusters: {info2['n_clusters']}")
    print(f"   Iteraciones: {info2['iterations']}")
    print(f"   Inercia: {info2['inertia']:.2f}")
    
    print("\n   Distribución de clusters:")
    for cluster in info2['clusters']:
        print(f"      Cluster {cluster['cluster_id']}: " 
              f"{cluster['percentage']:.1f}% ({cluster['num_pixels']} píxeles)")
    
    print("\n3. Segmentación Watershed:")
    result3, info3 = ImageSegmentation.watershed_segmentation(img)
    print(f"   Método: {info3['method']}")
    print(f"   Regiones detectadas: {info3['num_regions']}")
    
    print("\n4. Region Growing:")
    result4, info4 = ImageSegmentation.region_growing(img, threshold=20)
    print(f"   Método: {info4['method']}")
    print(f"   Punto semilla: {info4['seed_point']}")
    print(f"   Píxeles en región: {info4['pixels_in_region']}")
    print(f"   Porcentaje de imagen: {info4['percentage']:.2f}%")


def demo_template_matching():
    """Demostración de template matching"""
    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN: TEMPLATE MATCHING")
    print("=" * 60)
    
    # Crear imagen grande
    img = np.random.randint(50, 100, (400, 400), dtype=np.uint8)
    
    # Crear y colocar template
    template = np.ones((50, 50), dtype=np.uint8) * 200
    
    # Insertar template en posición conocida
    x_pos, y_pos = 150, 100
    img[y_pos:y_pos+50, x_pos:x_pos+50] = template
    
    print(f"\nTemplate insertado en posición: ({x_pos}, {y_pos})")
    
    print("\n1. Template Matching Manual (SSD):")
    result1, info1 = TemplateMatching.match_template_manual(
        img, template, method='ssd'
    )
    print(f"   Método: {info1['method']}")
    print(f"   Ubicación encontrada: {info1['best_location']}")
    print(f"   Score (SSD): {info1['best_score']:.2f}")
    print(f"   Error de posición: "
          f"({abs(info1['best_location'][0] - x_pos)}, "
          f"{abs(info1['best_location'][1] - y_pos)})")
    
    print("\n2. Template Matching Manual (NCC):")
    result2, info2 = TemplateMatching.match_template_manual(
        img, template, method='ncc'
    )
    print(f"   Método: {info2['method']}")
    print(f"   Ubicación encontrada: {info2['best_location']}")
    print(f"   Score (NCC): {info2['best_score']:.3f}")
    print(f"   Error de posición: "
          f"({abs(info2['best_location'][0] - x_pos)}, "
          f"{abs(info2['best_location'][1] - y_pos)})")
    
    print("\n3. Template Matching OpenCV:")
    result3, info3 = TemplateMatching.match_template_opencv(
        img, template, method='ccoeff_normed'
    )
    print(f"   Método: {info3['method']}")
    print(f"   Ubicación encontrada: {info3['best_location']}")
    print(f"   Score: {info3['best_score']:.3f}")
    print(f"   Error de posición: "
          f"({abs(info3['best_location'][0] - x_pos)}, "
          f"{abs(info3['best_location'][1] - y_pos)})")
    
    print("\n✓ Los tres métodos deberían encontrar el template en la misma ubicación")


def demo_all():
    """Ejecutar todas las demostraciones"""
    print("\n" + "=" * 60)
    print("SUITE COMPLETA DE DEMOSTRACIONES")
    print("=" * 60)
    print("\nEstas demostraciones muestran el funcionamiento de todas")
    print("las funcionalidades implementadas sin necesidad de GUI.\n")
    
    try:
        demo_otsu()
        demo_harris()
        demo_skeleton()
        demo_perimeter()
        demo_segmentation()
        demo_template_matching()
        
        print("\n" + "=" * 60)
        print("✓ TODAS LAS DEMOSTRACIONES COMPLETADAS EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error durante la demostración: {str(e)}")
        import traceback
        traceback.print_exc()


def create_test_images():
    """Crear imágenes de prueba para testing"""
    print("\n" + "=" * 60)
    print("CREANDO IMÁGENES DE PRUEBA")
    print("=" * 60)
    
    # 1. Imagen para Otsu
    img_otsu = np.zeros((200, 200), dtype=np.uint8)
    img_otsu[50:150, 50:150] = 200
    noise = np.random.normal(0, 20, img_otsu.shape)
    img_otsu = np.clip(img_otsu + noise, 0, 255).astype(np.uint8)
    cv2.imwrite('/home/claude/test_otsu.png', img_otsu)
    print("✓ test_otsu.png creada")
    
    # 2. Imagen para Harris
    img_harris = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img_harris, (50, 50), (250, 250), 255, 2)
    cv2.rectangle(img_harris, (100, 100), (200, 200), 255, 2)
    cv2.imwrite('/home/claude/test_harris.png', img_harris)
    print("✓ test_harris.png creada")
    
    # 3. Imagen para Esqueletonización
    img_skeleton = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img_skeleton, (100, 100), 60, 255, -1)
    cv2.imwrite('/home/claude/test_skeleton.png', img_skeleton)
    print("✓ test_skeleton.png creada")
    
    # 4. Imagen para Perímetro
    img_perimeter = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img_perimeter, (75, 75), 40, 255, -1)
    cv2.rectangle(img_perimeter, (150, 50), (250, 150), 255, -1)
    cv2.imwrite('/home/claude/test_perimeter.png', img_perimeter)
    print("✓ test_perimeter.png creada")
    
    # 5. Imagen para Segmentación
    img_seg = np.zeros((200, 300, 3), dtype=np.uint8)
    img_seg[0:100, 0:100] = [255, 0, 0]
    img_seg[0:100, 100:200] = [0, 255, 0]
    img_seg[0:100, 200:300] = [0, 0, 255]
    img_seg[100:200, 0:150] = [255, 255, 0]
    img_seg[100:200, 150:300] = [255, 0, 255]
    cv2.imwrite('/home/claude/test_segmentation.png', img_seg)
    print("✓ test_segmentation.png creada")
    
    # 6. Imágenes para Template Matching
    img_template_main = np.random.randint(50, 100, (400, 400), dtype=np.uint8)
    template = np.ones((50, 50), dtype=np.uint8) * 200
    img_template_main[100:150, 150:200] = template
    cv2.imwrite('/home/claude/test_template_main.png', img_template_main)
    cv2.imwrite('/home/claude/test_template_small.png', template)
    print("✓ test_template_main.png y test_template_small.png creadas")
    
    print("\n✓ Todas las imágenes de prueba creadas en /home/claude/")
    print("  Puedes usarlas para probar las funcionalidades en la GUI")


if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA DE DEMOSTRACIÓN")
    print("Nuevas funcionalidades de Visión por Computadora")
    print("=" * 60)
    
    print("\nOpciones:")
    print("1. Ejecutar demostraciones (muestra funcionamiento)")
    print("2. Crear imágenes de prueba (para usar en GUI)")
    print("3. Ambas")
    
    choice = input("\nSelecciona una opción (1/2/3): ").strip()
    
    if choice == '1':
        demo_all()
    elif choice == '2':
        create_test_images()
    elif choice == '3':
        create_test_images()
        print("\n")
        demo_all()
    else:
        print("Opción no válida. Ejecutando demostraciones...")
        demo_all()
    
    print("\n" + "=" * 60)
    print("FIN DE LA DEMOSTRACIÓN")
    print("=" * 60)
