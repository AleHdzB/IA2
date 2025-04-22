from bing_image_downloader import downloader
import os
from PIL import Image

def validate_images(directory):
    """Elimina imágenes no válidas o de formatos incompatibles."""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verifica si es una imagen válida
                # Opcional: Asegurarse de que el formato sea compatible
                if img.format not in ["JPEG", "PNG", "GIF", "BMP"]:
                    print(f"Formato incompatible eliminado: {file_path}")
                    os.remove(file_path)
            except (IOError, SyntaxError):
                print(f"Archivo no válido eliminado: {file_path}")
                os.remove(file_path)

# Descargar 100 imágenes de peces
downloader.download("fish", limit=100, output_dir='train', adult_filter_off=True, force_replace=False, timeout=60)
validate_images('train/fish')

# Descargar 100 imágenes de tigres
downloader.download("tiger", limit=100, output_dir='train', adult_filter_off=True, force_replace=False, timeout=60)
validate_images('train/tiger')

# Lista de clases
classes = ["fish", "tiger", "dog", "cat", "horse"]

# Descargar 20 imágenes para cada clase
for cls in classes:
    downloader.download(cls, limit=20, output_dir='test', adult_filter_off=True, force_replace=False, timeout=60)
    validate_images(f'test/{cls}')