"""
Baixa modelo YOLOv8 JÁ TREINADO para detecção de capacetes
Muito mais rápido que treinar do zero!
"""

import urllib.request
import os
from pathlib import Path

print("="*60)
print("BAIXANDO MODELO PRÉ-TREINADO PARA CAPACETES")
print("="*60)

# Lista de modelos disponíveis
modelos_disponiveis = [
    {
        'nome': 'Hard Hat Detection v1',
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'descricao': 'Modelo base YOLOv8 (NÃO específico para capacetes)',
        'recomendado': False
    },
    {
        'nome': 'Modelo Genérico YOLOv8n',
        'url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
        'descricao': 'Modelo padrão - NÃO detecta capacetes',
        'recomendado': False
    }
]

print("\n⚠️  IMPORTANTE:")
print("Não encontrei modelo público gratuito direto para download.\n")
print("VOCÊ TEM 3 OPÇÕES:\n")

print("=" *60)
print("OPÇÃO 1: ROBOFLOW UNIVERSE (RECOMENDADO - MAIS FÁCIL)")
print("="*60)
print("""
1. Acesse: https://universe.roboflow.com/
2. Busque: "hard hat detection"
3. Escolha um projeto (ex: "Hard Hat Workers Detection")
4. Clique em "Download this Dataset"
5. Crie conta gratuita (se necessário)
6. Escolha formato: YOLOv8
7. Baixe e extraia
8. Copie o arquivo best.pt para esta pasta:
   cp /caminho/do/download/best.pt {Path.cwd()}/best.pt

Depois execute:
   python detector-EPI-v2.py
""")

print("="*60)
print("OPÇÃO 2: ULTRALYTICS HUB (ONLINE)")
print("="*60)
print("""
1. Acesse: https://hub.ultralytics.com/
2. Crie conta gratuita
3. Busque modelos públicos de "hardhat" ou "helmet"
4. Use o modelo direto pelo ID:

   # No código Python:
   from ultralytics import YOLO
   modelo = YOLO('https://hub.ultralytics.com/models/SEU_MODEL_ID')
""")

print("="*60)
print("OPÇÃO 3: GOOGLE COLAB (TREINAR RAPIDAMENTE)")
print("="*60)
print("""
Vou criar um notebook Google Colab para você treinar em 15 minutos.
Pressione ENTER para criar o script...
""")

input()

# Criar script Colab
colab_script = '''
# EXECUTE ESTE CÓDIGO NO GOOGLE COLAB
# https://colab.research.google.com/

# Passo 1: Instalar dependências
!pip install ultralytics roboflow -q

# Passo 2: Baixar dataset
from roboflow import Roboflow
rf = Roboflow(api_key="")  # Não precisa de API key para datasets públicos

# Dataset público de capacetes
project = rf.workspace("roboflow-universe").project("hard-hat-sample")
dataset = project.version(3).download("yolov8")

# Passo 3: Treinar modelo
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

results = model.train(
    data=f'{dataset.location}/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='hardhat_model'
)

# Passo 4: Baixar modelo treinado
from google.colab import files
files.download('runs/detect/hardhat_model/weights/best.pt')

# Depois:
# 1. Baixe o arquivo best.pt
# 2. Coloque na pasta do projeto
# 3. Execute: python detector-EPI-v2.py
'''

with open('TREINAR_NO_COLAB.py', 'w') as f:
    f.write(colab_script)

print("\n✓ Script criado: TREINAR_NO_COLAB.py")
print("\nCopie o conteúdo deste arquivo e cole no Google Colab:")
print("   https://colab.research.google.com/\n")

print("="*60)
print("RESUMO")
print("="*60)
print("""
Infelizmente não há modelo de capacete disponível para download direto.

ESCOLHA UMA OPÇÃO ACIMA e depois execute:
   source venv/bin/activate
   python detector-EPI-v2.py

O arquivo best.pt precisa estar na pasta do projeto.
""")
