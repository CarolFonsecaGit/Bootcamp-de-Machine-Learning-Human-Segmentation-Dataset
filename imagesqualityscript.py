import os
import pandas as pd
from PIL import Image

# --- QUALIDADE DAS IMAGENS ---

# Configurações de caminho
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")

# Carregar o DataFrame
df = pd.read_csv(os.path.join(BASE_PATH, "df.csv"))

corrupted_images = []

print("Verificando integridade dos arquivos de imagem...")

for col in ["images", "masks", "collages"]:
    for path in df[col]:
        full_path = os.path.join(DATA_PATH, path)
        
        if os.path.isfile(full_path):
            try:
                with Image.open(full_path) as img:
                    img.verify() # Verifica se o arquivo está corrompido
            except Exception as e:
                corrupted_images.append(f"{path}: {e}")

print(f"\nTotal de imagens corrompidas identificadas: {len(corrupted_images)}")
if corrupted_images:
    print("Arquivos corrompidos:")
    for img in corrupted_images[:10]:
        print(f"- {img}")
else:
    print("Todas as imagens foram abertas e verificadas com sucesso.")
