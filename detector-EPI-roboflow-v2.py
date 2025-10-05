"""
Detector de Capacete MELHORADO - Detecta pessoas E verifica capacete
Combina: Roboflow para capacetes + YOLOv8 para pessoas
"""

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
from pathlib import Path

# CONFIGURAÇÕES
PASTA_IMAGENS = "imagens"
PASTA_SAIDA = "resultados"

# MODELO ROBOFLOW (para detectar capacetes)
MODELO_ID = "yolov8-hat-detection/3"  # Seu modelo
API_KEY = "UCavLvCNLtLkvAL92w9H"

# Confiança mínima
CONFIANCA_CAPACETE = 0.4
CONFIANCA_PESSOA = 0.5


def verificar_sobreposicao(box1, box2):
    """Verifica se duas caixas se sobrepõem (IoU)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calcula interseção
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return False, 0

    area_inter = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Se a interseção cobre mais de 30% da cabeça, considera que o capacete está nela
    overlap = area_inter / area_box2 if area_box2 > 0 else 0
    return overlap > 0.3, overlap


def processar_imagem(imagem_path, client_roboflow, modelo_yolo):
    """Detecta pessoas e verifica se têm capacete"""

    # Carrega imagem
    imagem = cv2.imread(str(imagem_path))
    if imagem is None:
        raise ValueError(f"Erro ao carregar: {imagem_path}")

    altura_img = imagem.shape[0]

    # 1. Detecta CAPACETES com Roboflow
    resultado_roboflow = client_roboflow.infer(str(imagem_path), model_id=MODELO_ID)
    capacetes_detectados = []

    for pred in resultado_roboflow.get('predictions', []):
        classe = pred.get('class', '').lower()
        confianca = pred.get('confidence', 0)

        # Apenas hardhat/helmet (não "hat" genérico)
        if ('hardhat' in classe or 'helmet' in classe) and confianca >= CONFIANCA_CAPACETE:
            x = pred['x']
            y = pred['y']
            w = pred['width']
            h = pred['height']

            capacetes_detectados.append({
                'bbox': (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)),
                'confianca': confianca,
                'classe': pred.get('class', 'capacete')
            })

    # 2. Detecta PESSOAS com YOLOv8
    resultados_yolo = modelo_yolo(imagem, conf=CONFIANCA_PESSOA, verbose=False)

    pessoas_detectadas = []
    for resultado in resultados_yolo:
        if resultado.boxes is None:
            continue

        for box in resultado.boxes:
            classe_id = int(box.cls[0])
            classe_nome = modelo_yolo.names[classe_id]

            # Apenas pessoas
            if classe_nome.lower() == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confianca = float(box.conf[0])

                pessoas_detectadas.append({
                    'bbox': (x1, y1, x2, y2),
                    'confianca': confianca
                })

    # 3. Verifica quais PESSOAS têm CAPACETE
    deteccoes_finais = []

    for pessoa in pessoas_detectadas:
        x1, y1, x2, y2 = pessoa['bbox']

        # Região da cabeça (20% superior da pessoa)
        altura_pessoa = y2 - y1
        cabeca_bbox = (x1, y1, x2, int(y1 + altura_pessoa * 0.25))

        # Verifica se algum capacete está na região da cabeça
        tem_capacete = False
        capacete_info = None

        for capacete in capacetes_detectados:
            sobrepoe, overlap = verificar_sobreposicao(capacete['bbox'], cabeca_bbox)
            if sobrepoe:
                tem_capacete = True
                capacete_info = capacete
                break

        deteccoes_finais.append({
            'bbox': (x1, y1, x2, y2),
            'tem_capacete': tem_capacete,
            'confianca_pessoa': pessoa['confianca'],
            'confianca_capacete': capacete_info['confianca'] if capacete_info else 0,
            'tipo': 'COM CAPACETE' if tem_capacete else 'SEM CAPACETE'
        })

    return imagem, deteccoes_finais


def desenhar_deteccoes(imagem, deteccoes):
    """Desenha as detecções na imagem"""

    resultado = imagem.copy()

    for det in deteccoes:
        x1, y1, x2, y2 = det['bbox']
        tem_capacete = det['tem_capacete']
        conf_pessoa = det['confianca_pessoa']
        conf_capacete = det['confianca_capacete']

        # Define cor
        if tem_capacete:
            cor = (0, 255, 0)  # Verde
            texto = f"COM CAPACETE: {conf_capacete:.2f}"
        else:
            cor = (0, 0, 255)  # Vermelho
            texto = f"SEM CAPACETE: {conf_pessoa:.2f}"

        # Desenha retângulo
        cv2.rectangle(resultado, (x1, y1), (x2, y2), cor, 3)

        # Texto
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(resultado, (x1, y1 - th - 10), (x1 + tw, y1), cor, -1)
        cv2.putText(resultado, texto, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return resultado


def main():
    """Função principal"""
    print("="*60)
    print("DETECTOR DE CAPACETE - VERSÃO MELHORADA")
    print("="*60)

    # Cria pasta de saída
    Path(PASTA_SAIDA).mkdir(exist_ok=True)

    # Carrega modelos
    print("\n[1/2] Carregando modelo Roboflow (capacetes)...")
    client_roboflow = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY
    )
    print(f"      ✓ Roboflow conectado")

    print("\n[2/2] Carregando YOLOv8 (pessoas)...")
    modelo_yolo = YOLO('yolov8n.pt')
    print(f"      ✓ YOLOv8 carregado\n")

    # Processa imagens
    pasta = Path(PASTA_IMAGENS)

    if not pasta.exists():
        print(f"❌ Pasta '{PASTA_IMAGENS}' não existe!")
        return

    extensoes = ['.jpg', '.jpeg', '.png', '.bmp']
    imagens = [f for f in pasta.iterdir() if f.suffix.lower() in extensoes]

    if not imagens:
        print(f"❌ Nenhuma imagem em '{PASTA_IMAGENS}'")
        return

    print(f"📁 Encontradas {len(imagens)} imagens")
    print("="*60)

    for i, img_path in enumerate(imagens, 1):
        print(f"\n[{i}/{len(imagens)}] {img_path.name}")
        print("-"*60)

        try:
            # Processa
            imagem, deteccoes = processar_imagem(img_path, client_roboflow, modelo_yolo)

            # Estatísticas
            com_capacete = sum(1 for d in deteccoes if d['tem_capacete'])
            sem_capacete = len(deteccoes) - com_capacete

            print(f"  Pessoas detectadas: {len(deteccoes)}")
            print(f"  ✓ COM capacete: {com_capacete}")
            print(f"  ✗ SEM capacete: {sem_capacete}")

            # Desenha e salva
            resultado_img = desenhar_deteccoes(imagem, deteccoes)
            saida = Path(PASTA_SAIDA) / f"{img_path.stem}_detectado{img_path.suffix}"
            cv2.imwrite(str(saida), resultado_img)

            print(f"  💾 Salvo: {saida}")

        except Exception as e:
            print(f"  ❌ Erro: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ CONCLUÍDO!")
    print(f"📂 Resultados em: {PASTA_SAIDA}/")


if __name__ == "__main__":
    main()
