"""
ENTRENAMIENTO DEL MODELO SVM - CORRECCIÓN DE ERROR DE TIPOS
Soluciona el error 'float object is not subscriptable' eliminando índices innecesarios.
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import yaml

def resize_keep_aspect(image, target_width=320):
    if image is None: return None
    h, w = image.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h))

def get_texture_score(img_gray):
    """Calcula qué tan 'rugoso' es el billete (Papel vs Polímero)"""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def extract_enhanced_descriptors(image_path, label_path=None, img_width=None, img_height=None, crop=None):
    """
    Función compatible con Photoescom.py con corrección de tipos numéricos.
    """

    # --- 1. Obtener la imagen ---
    if crop is not None:
        img = crop
    else:
        img = cv2.imread(image_path)
        if img is None: return None, None

        # Recorte por etiqueta (solo para entrenamiento)
        if label_path:
            try:
                with open(label_path, 'r') as f:
                    parts = f.readline().strip().split()
                c_id, cx, cy, cw, ch = map(float, parts)

                h_current, w_current = img.shape[:2]
                x = int((cx - cw/2) * w_current)
                y = int((cy - ch/2) * h_current)
                w = int(cw * w_current)
                h = int(ch * h_current)

                # Evitar salirse de la imagen
                x, y = max(0, x), max(0, y)
                w, h = min(w, w_current-x), min(h, h_current-y)

                img = img[y:y+h, x:x+w]
                if img.size == 0: return None, None

                return _compute_robust_features(img), int(c_id)
            except:
                return None, None

    return _compute_robust_features(img), None

def _compute_robust_features(img):
    """Lógica interna corregida: cv2.mean devuelve escalares, no arrays"""

    # 1. Normalizar tamaño
    img = resize_keep_aspect(img, target_width=320)

    # Separar canales
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # --- A. MÁSCARA INTELIGENTE ---
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) < (img.size * 0.2):
        mask = np.ones_like(gray) * 255

    # --- B. COLOR (HSV + LAB) ---
    # meanStdDev devuelve arrays numpy -> Usamos [0][0]
    mean_h, std_h = cv2.meanStdDev(hsv[:,:,0], mask=mask)
    mean_s, std_s = cv2.meanStdDev(hsv[:,:,1], mask=mask)

    # cv2.mean devuelve TUPLA DE FLOATS (escalares) -> NO USAR [0]
    mean_l, mean_a, mean_b_lab = cv2.mean(lab, mask=mask)[:3]

    # --- C. TEXTURA (Material) ---
    texture = get_texture_score(gray)

    # --- D. GEOMETRÍA (Forma) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        dim1, dim2 = min(w, h), max(w, h)
        aspect_ratio = dim2 / dim1 if dim1 > 0 else 0
    else:
        solidity, aspect_ratio = 0, 0

    # === VECTOR DE CARACTERÍSTICAS CORREGIDO ===
    features = np.array([
        mean_h[0][0],   # Array -> OK
        std_h[0][0],    # Array -> OK
        mean_s[0][0],   # Array -> OK
        mean_a,         # Float -> OK (CORREGIDO: Quitamos [0])
        mean_b_lab,     # Float -> OK (CORREGIDO: Quitamos [0])
        texture,        # Float -> OK
        solidity,       # Float -> OK
        aspect_ratio,   # Float -> OK
        mean_l,         # Float -> OK (CORREGIDO: Quitamos [0])
        std_s[0][0]     # Array -> OK
    ])

    return np.nan_to_num(features)

def load_class_names():
    path = "dataset/Billetes Mexicanos.v5i.yolov8/data.yaml"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return {i: n for i, n in enumerate(yaml.safe_load(f)['names'])}
    return {0:'20', 1:'50', 2:'100', 3:'200', 4:'500', 5:'1000'}

def train_and_save_model():
    base_path = "dataset/Billetes Mexicanos.v5i.yolov8"
    print("🚀 Entrenando modelo corregido...")

    if not os.path.exists(base_path):
        print("❌ Dataset no encontrado.")
        return

    X, y = [], []

    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base_path, split, 'images')
        lbl_dir = os.path.join(base_path, split, 'labels')
        if not os.path.exists(img_dir): continue

        files = [f for f in os.listdir(img_dir) if f.endswith(('jpg','png','jpeg'))]
        print(f"   Leyendo {split}: {len(files)} imágenes...")

        for f in files:
            img_p = os.path.join(img_dir, f)
            lbl_p = os.path.join(lbl_dir, f.rsplit('.',1)[0]+'.txt')
            if os.path.exists(lbl_p):
                feat, label = extract_enhanced_descriptors(img_p, label_path=lbl_p)
                if feat is not None:
                    X.append(feat)
                    y.append(label)

    print(f"✅ Total: {len(X)} muestras.")
    if len(X) == 0: return

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GridSearch
    print("🤖 Optimizando SVM...")
    param_grid = {
        'C': [1, 10, 50, 100],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=3)
    grid.fit(X_scaled, y)

    print(f"🏆 Precisión: {grid.best_score_:.2%}")

    # Guardar
    os.makedirs('models', exist_ok=True)
    with open('models/bill_recognizer.pkl', 'wb') as f:
        pickle.dump({
            'classifier': grid.best_estimator_,
            'scaler': scaler,
            'class_names': load_class_names()
        }, f)

    print("💾 Modelo guardado. ¡Ahora sí funcionará!")

if __name__ == "__main__":
    train_and_save_model()