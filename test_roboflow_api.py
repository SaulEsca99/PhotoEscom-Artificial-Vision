from inference_sdk import InferenceHTTPClient
import cv2

# Tu imagen
image_path = "bile.jpg"  # O la ruta completa a tu imagen

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="tkhi9LqnfsZxo8AK0ULH"
)

print("=" * 60)
print("PROBANDO DIFERENTES VERSIONES DEL MODELO")
print("=" * 60)

# Probar versión 5
print("\n📡 Probando versión 5...")
try:
    result = CLIENT.infer(image_path, model_id="billetes-mexicanos-9s5an/5")
    print(f"✓ Respuesta recibida")
    print(f"  Tiempo: {result.get('time', 'N/A')}")
    print(f"  Imagen: {result.get('image', {})}")
    print(f"  Predicciones: {len(result.get('predictions', []))}")

    if result.get('predictions'):
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n  Predicción {i}:")
            print(f"    Clase: {pred.get('class', 'N/A')}")
            print(f"    Confianza: {pred.get('confidence', 0):.2%}")
            print(f"    Posición: ({pred.get('x', 0):.0f}, {pred.get('y', 0):.0f})")
            print(f"    Tamaño: {pred.get('width', 0):.0f}x{pred.get('height', 0):.0f}")
    else:
        print("  ❌ No hay predicciones")
        print(f"  Respuesta completa: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# Probar versión 1
print("\n" + "=" * 60)
print("📡 Probando versión 1...")
try:
    result = CLIENT.infer(image_path, model_id="billetes-mexicanos-9s5an/1")
    print(f"✓ Respuesta recibida")
    print(f"  Tiempo: {result.get('time', 'N/A')}")
    print(f"  Imagen: {result.get('image', {})}")
    print(f"  Predicciones: {len(result.get('predictions', []))}")

    if result.get('predictions'):
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n  Predicción {i}:")
            print(f"    Clase: {pred.get('class', 'N/A')}")
            print(f"    Confianza: {pred.get('confidence', 0):.2%}")
            print(f"    Posición: ({pred.get('x', 0):.0f}, {pred.get('y', 0):.0f})")
            print(f"    Tamaño: {pred.get('width', 0):.0f}x{pred.get('height', 0):.0f}")
    else:
        print("  ❌ No hay predicciones")
        print(f"  Respuesta completa: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)

# Verificar la imagen
img = cv2.imread(image_path)
if img is not None:
    h, w = img.shape[:2]
    print(f"\n📸 Imagen cargada correctamente:")
    print(f"  Dimensiones: {w}x{h} px")
    print(f"  Tamaño archivo: {len(open(image_path, 'rb').read()) / 1024:.2f} KB")
else:
    print(f"\n❌ No se pudo cargar la imagen: {image_path}")