"""
Script para testar qual modelo funciona melhor para detecção de capacetes
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

# Lista de modelos para testar
modelos = [
    {
        'nome': 'PPE (Personal Protective Equipment)',
        'url': 'https://github.com/RizwanMunawar/yolov8-object-tracking/releases/download/v1.0/ppe.pt'
    },
    {
        'nome': 'YOLOv8 Nano padrão',
        'url': 'yolov8n.pt'
    },
]

pasta_imagens = Path("imagens")
imagens = list(pasta_imagens.glob("*.jpg")) + list(pasta_imagens.glob("*.png"))

if not imagens:
    print("Nenhuma imagem encontrada na pasta 'imagens/'")
    exit()

print(f"Encontradas {len(imagens)} imagens\n")
print("="*60)

for modelo_info in modelos:
    print(f"\n🔍 Testando: {modelo_info['nome']}")
    print("-"*60)

    try:
        modelo = YOLO(modelo_info['url'])
        print(f"✓ Modelo carregado")
        print(f"📋 Classes detectadas pelo modelo:")
        print(f"   {modelo.names}\n")

        # Testa primeira imagem
        img_teste = str(imagens[0])
        resultado = modelo(img_teste, conf=0.3)

        print(f"🖼️  Testando em: {imagens[0].name}")
        for r in resultado:
            if r.boxes:
                for box in r.boxes:
                    classe = modelo.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    print(f"   → Detectado: {classe} (confiança: {conf:.2%})")
            else:
                print("   → Nenhum objeto detectado")

    except Exception as e:
        print(f"❌ Erro: {e}")

print("\n" + "="*60)
print("\n💡 Use o modelo que detectar classes como:")
print("   - 'hardhat' / 'helmet' (com capacete)")
print("   - 'no-hardhat' / 'no-helmet' (sem capacete)")
