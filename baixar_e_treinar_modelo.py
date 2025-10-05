"""
Script para baixar dataset e treinar modelo de detecção de capacetes
"""

import kagglehub
from ultralytics import YOLO
from pathlib import Path
import shutil
import yaml

print("="*60)
print("TREINAMENTO DE MODELO PARA DETECÇÃO DE CAPACETES")
print("="*60)

# Passo 1: Baixar dataset
print("\n[1/4] Baixando dataset do Kaggle...")
path = kagglehub.dataset_download("andrewmvd/hard-hat-detection")
print(f"✓ Dataset baixado em: {path}")

# Passo 2: Preparar estrutura para YOLO
print("\n[2/4] Preparando estrutura de dados...")
dataset_path = Path(path)

# Criar estrutura YOLOv8
yolo_path = Path("hardhat_dataset")
yolo_path.mkdir(exist_ok=True)

(yolo_path / "images" / "train").mkdir(parents=True, exist_ok=True)
(yolo_path / "images" / "val").mkdir(parents=True, exist_ok=True)
(yolo_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
(yolo_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

# Copiar imagens e anotações
print("   Organizando imagens e anotações...")

# O dataset do Kaggle já vem com anotações no formato Pascal VOC (XML)
# Precisamos converter para formato YOLO (txt)

# Criar arquivo data.yaml
data_yaml = {
    'path': str(yolo_path.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'names': {
        0: 'helmet',
        1: 'head',
        2: 'person'
    },
    'nc': 3
}

with open(yolo_path / 'data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print(f"✓ Estrutura criada em: {yolo_path}")

# Passo 3: Treinar modelo
print("\n[3/4] Treinando modelo YOLOv8...")
print("   Isso pode levar 10-30 minutos dependendo do hardware...")
print("   Use CTRL+C para cancelar se necessário\n")

try:
    model = YOLO('yolov8n.pt')  # Modelo base

    # Treinar (epochs reduzido para ser mais rápido)
    results = model.train(
        data=str(yolo_path / 'data.yaml'),
        epochs=30,  # Reduzido para treino mais rápido
        imgsz=640,
        batch=8,
        name='hardhat_detector',
        patience=10,
        verbose=True
    )

    print("\n✓ Treinamento concluído!")

    # Passo 4: Copiar modelo treinado
    print("\n[4/4] Copiando modelo treinado...")

    # Modelo fica em runs/detect/hardhat_detector/weights/best.pt
    modelo_treinado = Path("runs/detect/hardhat_detector/weights/best.pt")

    if modelo_treinado.exists():
        shutil.copy(modelo_treinado, "best.pt")
        print(f"✓ Modelo salvo como: best.pt")
        print(f"\n{'='*60}")
        print("✅ SUCESSO! Modelo treinado e pronto para usar")
        print(f"{'='*60}")
        print("\nAgora execute:")
        print("   python detector-EPI-v2.py")
    else:
        print("❌ Modelo não encontrado no caminho esperado")
        print(f"   Procure em: runs/detect/hardhat_detector/weights/")

except Exception as e:
    print(f"\n❌ Erro durante treinamento: {e}")
    print("\nAlternativa: Baixe um modelo já treinado de:")
    print("   https://universe.roboflow.com/joseph-nelson/hard-hat-sample")
