import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from inference_sdk import InferenceHTTPClient


def extract_descriptors(image_path, label_path=None, img_width=None, img_height=None, crop=None):
    """Extrae descriptores de región y perímetro"""
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    if area == 0:
        return None, None

    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / area if area > 0 else 0

    rect = cv2.minAreaRect(contour)
    box_width, box_height = rect[1]
    eccentricity = min(box_width, box_height) / max(box_width, box_height) if max(box_width, box_height) > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    features = np.array([area, perimeter, compactness, eccentricity, solidity] + hu_moments.tolist())

    return features, class_id


def load_class_names():
    """Cargar nombres de clases desde data.yaml"""
    import yaml
    data_yaml_path = "dataset/Billetes Mexicanos.v5i.yolov8/data.yaml"

    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            names = data.get('names', [])
            # Convertir lista a diccionario {índice: nombre}
            return {i: name for i, name in enumerate(names)}

    # Mapeo por defecto si no existe el archivo
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


def infer_with_roboflow(image_path, model_version=5, use_descriptors=False, clf=None, scaler=None, class_names=None):
    """Inferencia con Roboflow SDK"""
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="tkhi9LqnfsZxo8AK0ULH"
    )

    # Probar con la versión especificada
    model_id = f"billetes-mexicanos-9s5an/{model_version}"
    print(f"Usando modelo: {model_id}")

    try:
        result = CLIENT.infer(image_path, model_id=model_id)
    except Exception as e:
        print(f"❌ Error con versión {model_version}: {e}")
        print("Intentando con versión 1...")
        try:
            result = CLIENT.infer(image_path, model_id="billetes-mexicanos-9s5an/1")
        except Exception as e2:
            print(f"❌ Error con versión 1: {e2}")
            return None, None

    print(f"\n✓ Resultado de inferencia Roboflow:")
    print(f"  Tiempo: {result.get('time', 0):.3f}s")
    print(f"  Imagen: {result['image']['width']}x{result['image']['height']} px")

    if 'predictions' in result and result['predictions']:
        print(f"\n💵 {len(result['predictions'])} billete(s) detectado(s):\n")

        for i, pred in enumerate(result['predictions'], 1):
            class_name = pred['class']
            confidence = pred['confidence']
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']

            print(f"Billete #{i}:")
            print(f"  💰 Denominación: ${class_name} pesos")
            print(f"  📊 Confianza: {confidence:.4f} ({confidence * 100:.1f}%)")
            print(f"  📍 Posición: ({x:.0f}, {y:.0f})")
            print(f"  📏 Tamaño: {w:.0f}x{h:.0f} px")

            if use_descriptors and clf and scaler and class_names and i == 1:
                try:
                    img = cv2.imread(image_path)
                    h_img, w_img = img.shape[:2]
                    x1 = max(0, int(x - w / 2))
                    y1 = max(0, int(y - h / 2))
                    x2 = min(w_img, int(x + w / 2))
                    y2 = min(h_img, int(y + h / 2))

                    crop = img[y1:y2, x1:x2]

                    if crop.size > 0:
                        features, _ = extract_descriptors(None, None, None, None, crop=crop)
                        if features is not None:
                            features_scaled = scaler.transform([features])
                            svm_pred = clf.predict(features_scaled)[0]
                            svm_class_name = class_names.get(svm_pred, f"Clase {svm_pred}")
                            print(f"  🤖 Clasificación SVM: ${svm_class_name} pesos")
                except Exception as e:
                    print(f"  ⚠️  Error en clasificación SVM: {e}")
            print()

        return result['predictions'][0]['class'], result['predictions'][0]['confidence']
    else:
        print("❌ No se detectaron billetes.")
        return None, None


def main():
    base_dataset_path = "dataset/Billetes Mexicanos.v5i.yolov8"

    if not os.path.exists(base_dataset_path):
        print("ERROR: No se encontró el dataset")
        return

    print(f"🗂️  Dataset encontrado en: {base_dataset_path}\n")

    # Cargar nombres de clases
    class_names = load_class_names()
    print(f"📋 Mapeo de clases:")
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
            features, label = extract_descriptors(img_path, label_path, w, h)

            if features is None:
                errors += 1
                continue

            data[split_name][0].append(features)
            data[split_name][1].append(label)
            count += 1

        print(f"   ✓ {count} imágenes procesadas")
        if errors > 0:
            print(f"   ⚠️  {errors} imágenes con errores")

    X_train, y_train = data['train']
    X_valid, y_valid = data['valid']
    X_test, y_test = data['test']

    if not X_train:
        print("\nERROR: No se extrajeron características del training set.")
        return

    print(f"\n{'=' * 60}")
    print(f"📊 RESUMEN DEL DATASET:")
    print(f"{'=' * 60}")
    print(f"Train: {len(X_train)} imágenes")
    print(f"Valid: {len(X_valid)} imágenes")
    print(f"Test: {len(X_test)} imágenes")
    print(f"Clases encontradas: {sorted(set(y_train))}")

    # Mostrar distribución de clases
    from collections import Counter
    class_dist = Counter(y_train)
    print(f"\n💵 Distribución de clases en train:")
    for class_id in sorted(class_dist.keys()):
        class_name = class_names.get(class_id, f"Clase {class_id}")
        print(f"   ${class_name:15s}: {class_dist[class_id]:3d} imágenes")

    print(f"{'=' * 60}\n")

    print("⚙️  Escalando características...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if len(X_valid) > 0:
        X_valid = scaler.transform(X_valid)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test)

    print("🤖 Entrenando modelo SVM...")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)
    print("✓ Modelo SVM entrenado!\n")

    if len(X_valid) > 0:
        y_pred_valid = clf.predict(X_valid)
        print(f"{'=' * 60}")
        print("📈 RESULTADOS EN VALIDATION SET:")
        print(f"{'=' * 60}")
        print(f"Accuracy: {accuracy_score(y_valid, y_pred_valid):.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(y_valid, y_pred_valid, zero_division=0))

    if len(X_test) > 0:
        y_pred_test = clf.predict(X_test)
        print(f"\n{'=' * 60}")
        print("📈 RESULTADOS EN TEST SET:")
        print(f"{'=' * 60}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred_test, zero_division=0))

    print(f"\n{'=' * 60}")
    print("🔍 EJEMPLOS DE INFERENCIA CON ROBOFLOW:")
    print(f"{'=' * 60}\n")

    test_img_dir = os.path.join(base_dataset_path, 'test', 'images')
    if os.path.exists(test_img_dir):
        images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Probar con 3 imágenes diferentes
        num_tests = min(3, len(images))

        for test_num in range(num_tests):
            example_image = os.path.join(test_img_dir, images[test_num])
            print(f"{'─' * 60}")
            print(f"Ejemplo {test_num + 1}: {os.path.basename(example_image)}")
            print(f"{'─' * 60}")

            infer_with_roboflow(example_image, model_version=5, use_descriptors=True,
                                clf=clf, scaler=scaler, class_names=class_names)

            if test_num < num_tests - 1:
                print()


if __name__ == "__main__":
    main()