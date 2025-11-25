"""
ENTRENAMIENTO DEL MODELO SVM PARA RECONOCIMIENTO DE BILLETES
Sin Roboflow - 100% Local
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml


def extract_enhanced_descriptors(image_path, label_path=None, img_width=None, img_height=None, crop=None):
    """
    Extrae descriptores mejorados de región y perímetro

    Descriptores extraídos:
    - Geométricos: área, perímetro, compacidad, aspect ratio, etc.
    - Momentos: momentos de Hu (7)
    - Color: histogramas HSV + estadísticas
    - Textura: gradientes Sobel
    """
    class_id = None

    if crop is not None:
        img = crop
    else:
        img = cv2.imread(image_path)
        if img is None:
            return None, None

        if label_path:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                return None, None

            label_parts = lines[0].strip().split()
            if len(label_parts) < 5:
                return None, None

            class_id = int(label_parts[0])
            x_center = float(label_parts[1])
            y_center = float(label_parts[2])
            w = float(label_parts[3])
            h = float(label_parts[4])

            x1 = int((x_center - w / 2) * img_width)
            y1 = int((y_center - h / 2) * img_height)
            x2 = int((x_center + w / 2) * img_width)
            y2 = int((y_center + h / 2) * img_height)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            margin = int(min(w, h) * img_width * 0.05)
            crop_img = img[max(0, y1 - margin):min(img_height, y2 + margin),
            max(0, x1 - margin):min(img_width, x2 + margin)]

            if crop_img.size == 0:
                return None, None
            img = crop_img

    # Preprocesamiento mejorado
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    # === DESCRIPTORES GEOMÉTRICOS ===
    area = cv2.contourArea(contour)
    if area == 0:
        return None, None

    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / area if area > 0 else 0

    rect = cv2.minAreaRect(contour)
    box_width, box_height = rect[1]
    aspect_ratio = max(box_width, box_height) / min(box_width, box_height) if min(box_width, box_height) > 0 else 0
    eccentricity = min(box_width, box_height) / max(box_width, box_height) if max(box_width, box_height) > 0 else 0
    rect_area = box_width * box_height
    extent = area / rect_area if rect_area > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = hull_perimeter / perimeter if perimeter > 0 else 0

    # === MOMENTOS ===
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    mu20 = moments['mu20'] / (moments['m00'] ** 2) if moments['m00'] > 0 else 0
    mu02 = moments['mu02'] / (moments['m00'] ** 2) if moments['m00'] > 0 else 0

    # === COLOR EN HSV ===
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    hist_h = cv2.calcHist([img_hsv], [0], mask, [8], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], mask, [8], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], mask, [8], [0, 256])

    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-10)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-10)
    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-10)

    mean_h, std_h = cv2.meanStdDev(img_hsv[:, :, 0], mask=mask)
    mean_s, std_s = cv2.meanStdDev(img_hsv[:, :, 1], mask=mask)
    mean_v, std_v = cv2.meanStdDev(img_hsv[:, :, 2], mask=mask)

    # === TEXTURA ===
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    mean_texture = np.mean(gradient_magnitude[mask > 0])
    std_texture = np.std(gradient_magnitude[mask > 0])

    # === VECTOR DE CARACTERÍSTICAS ===
    features = np.array([
        area, perimeter, compactness, aspect_ratio, eccentricity,
        extent, solidity, convexity, mu20, mu02,
        *hu_moments,  # 7 momentos de Hu
        mean_h[0][0], std_h[0][0], mean_s[0][0], std_s[0][0], mean_v[0][0], std_v[0][0],
        *hist_h, *hist_s, *hist_v,  # 24 bins de histograma
        mean_texture, std_texture
    ])

    return features, class_id


def load_class_names():
    """Cargar nombres de clases desde data.yaml"""
    data_yaml_path = "dataset/Billetes Mexicanos.v5i.yolov8/data.yaml"

    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            names = data.get('names', [])
            return {i: name for i, name in enumerate(names)}

    # Mapeo por defecto
    return {
        0: '1',
        1: '100',
        2: '1000',
        3: '20',
        4: '200',
        5: '5',
        6: '50',
        7: '500',
        8: '500 nuevo'
    }


def train_and_save_model():
    """Entrenar y guardar el modelo SVM"""
    base_dataset_path = "dataset/Billetes Mexicanos.v5i.yolov8"

    if not os.path.exists(base_dataset_path):
        print("❌ Error: No se encontró el dataset")
        print(f"   Buscando en: {os.path.abspath(base_dataset_path)}")
        return

    print(f"🗂️  Dataset: {base_dataset_path}\n")

    class_names = load_class_names()
    print(f"📋 Clases detectadas: {len(class_names)}")
    for idx, name in class_names.items():
        print(f"   {idx}: ${name} pesos")
    print()

    splits = {
        'train': os.path.join(base_dataset_path, 'train'),
        'valid': os.path.join(base_dataset_path, 'valid'),
        'test': os.path.join(base_dataset_path, 'test')
    }

    data = {'train': ([], []), 'valid': ([], []), 'test': ([], [])}

    for split_name, split_path in splits.items():
        img_dir = os.path.join(split_path, 'images')
        label_dir = os.path.join(split_path, 'labels')

        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"⚠️  {split_name}: No encontrado, omitiendo...")
            continue

        print(f"📁 Procesando {split_name}...")
        count = 0
        errors = 0

        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(img_dir, img_file)
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                errors += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                errors += 1
                continue

            h, w = img.shape[:2]
            features, label = extract_enhanced_descriptors(img_path, label_path, w, h)

            if features is None:
                errors += 1
                continue

            data[split_name][0].append(features)
            data[split_name][1].append(label)
            count += 1

        print(f"   ✓ {count} imágenes procesadas")
        if errors > 0:
            print(f"   ⚠️  {errors} errores")

    X_train, y_train = data['train']
    X_valid, y_valid = data['valid']
    X_test, y_test = data['test']

    if not X_train:
        print("\n❌ Error: No hay datos de entrenamiento")
        return

    print(f"\n{'=' * 60}")
    print(f"📊 RESUMEN:")
    print(f"{'=' * 60}")
    print(f"Train: {len(X_train)} imágenes")
    print(f"Valid: {len(X_valid)} imágenes")
    print(f"Test: {len(X_test)} imágenes")
    print(f"Features por imagen: {len(X_train[0])}")
    print(f"Clases en train: {sorted(set(y_train))}")
    print(f"{'=' * 60}\n")

    # Mostrar distribución
    from collections import Counter
    dist = Counter(y_train)
    print(f"💵 Distribución de clases:")
    for class_id in sorted(dist.keys()):
        name = class_names.get(class_id, str(class_id))
        print(f"   ${name:15s}: {dist[class_id]:3d} imágenes")
    print()

    # Escalar
    print("⚙️  Escalando características...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if len(X_valid) > 0:
        X_valid = scaler.transform(X_valid)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test)

    # Entrenar SVM
    print("🤖 Entrenando SVM con parámetros optimizados...")
    clf = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("✓ Modelo entrenado!\n")

    # Evaluar
    train_acc = accuracy_score(y_train, clf.predict(X_train))

    valid_acc = None
    if len(X_valid) > 0:
        y_pred_valid = clf.predict(X_valid)
        valid_acc = accuracy_score(y_valid, y_pred_valid)

        print(f"{'=' * 60}")
        print("📈 VALIDATION:")
        print(f"{'=' * 60}")
        print(f"Accuracy: {valid_acc:.4f}\n")

        unique_classes = sorted(set(y_valid))
        target_names = [class_names.get(i, str(i)) for i in unique_classes]

        print(classification_report(
            y_valid,
            y_pred_valid,
            labels=unique_classes,
            target_names=target_names,
            zero_division=0
        ))

    test_acc = None
    if len(X_test) > 0:
        y_pred_test = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"\n{'=' * 60}")
        print("📈 TEST:")
        print(f"{'=' * 60}")
        print(f"Accuracy: {test_acc:.4f}\n")

        unique_classes = sorted(set(y_test))
        target_names = [class_names.get(i, str(i)) for i in unique_classes]

        print(classification_report(
            y_test,
            y_pred_test,
            labels=unique_classes,
            target_names=target_names,
            zero_division=0
        ))

    # Guardar modelo
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'class_names': class_names,
        'feature_count': len(X_train[0]),
        'train_accuracy': train_acc,
        'valid_accuracy': valid_acc,
        'test_accuracy': test_acc
    }

    os.makedirs('models', exist_ok=True)
    model_path = 'models/bill_recognizer.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n{'=' * 60}")
    print("💾 MODELO GUARDADO")
    print(f"{'=' * 60}")
    print(f"Ruta: {model_path}")
    print(f"Tamaño: {os.path.getsize(model_path) / 1024:.2f} KB")
    print(f"Train Acc: {train_acc:.4f}")
    if valid_acc:
        print(f"Valid Acc: {valid_acc:.4f}")
    if test_acc:
        print(f"Test Acc: {test_acc:.4f}")
    print(f"{'=' * 60}\n")
    print("✅ ¡Listo! Puedes usar el modelo en PhotoEscom")


if __name__ == "__main__":
    train_and_save_model()