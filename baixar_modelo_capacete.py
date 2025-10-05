"""
Script para baixar modelo YOLOv8 treinado para detecção de capacetes
do Roboflow Universe
"""

from roboflow import Roboflow
import os

# Inicializa Roboflow (não precisa de API key para modelos públicos)
rf = Roboflow(api_key="")

print("Baixando modelo de detecção de capacetes...")
print("="*60)

# Usa projeto público de detecção de capacetes de segurança
# Este modelo foi treinado com milhares de imagens de pessoas com/sem capacete
project = rf.workspace("roboflow-universe").project("hard-hat-detection")
dataset = project.version(1).download("yolov8")

print(f"\n✓ Modelo baixado com sucesso!")
print(f"📂 Localização: {dataset.location}")
print(f"\nArquivos:")
print(f"  - data.yaml: {os.path.join(dataset.location, 'data.yaml')}")
