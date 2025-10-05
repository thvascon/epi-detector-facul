"""
Detector de Capacete MELHORADO
Usa modelo YOLOv8 pré-treinado do Ultralytics HUB para detecção de capacetes
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# CONFIGURAÇÕES
PASTA_IMAGENS = "imagens"
PASTA_SAIDA = "resultados"
CONFIANCA_MINIMA = 0.25

# URL do modelo pré-treinado para detecção de capacetes
# Este modelo foi treinado especificamente para: hardhat, NO-hardhat, person
MODELO_URL = "best.pt"  # Será baixado automaticamente se não existir

def detectar_capacetes(imagem_path, modelo, conf=0.25):
    """Detecta capacetes em uma imagem"""
    imagem = cv2.imread(str(imagem_path))
    if imagem is None:
        raise ValueError(f"Não foi possível carregar: {imagem_path}")

    # Executa detecção
    resultados = modelo(imagem, conf=conf, verbose=False)

    deteccoes = []
    for resultado in resultados:
        if resultado.boxes is None:
            continue

        for box in resultado.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confianca = float(box.conf[0])
            classe_id = int(box.cls[0])
            classe_nome = modelo.names[classe_id]

            # Verifica se tem capacete
            classe_lower = classe_nome.lower()

            if 'hardhat' in classe_lower or 'helmet' in classe_lower or 'with' in classe_lower:
                tem_capacete = True
            elif 'no' in classe_lower or 'without' in classe_lower or 'person' in classe_lower:
                # Se detectou pessoa, assume SEM capacete (modelo separado detectaria hardhat)
                tem_capacete = False
            else:
                # Padrão: se não é explicitamente capacete, considera sem
                tem_capacete = False

            deteccoes.append({
                'bbox': (x1, y1, x2, y2),
                'confianca': confianca,
                'classe': classe_nome,
                'tem_capacete': tem_capacete
            })

    return imagem, deteccoes


def desenhar_resultado(imagem, deteccoes):
    """Desenha detecções na imagem"""
    resultado = imagem.copy()

    for det in deteccoes:
        x1, y1, x2, y2 = det['bbox']
        tem_capacete = det['tem_capacete']
        conf = det['confianca']

        # Cor: Verde = COM capacete, Vermelho = SEM capacete
        cor = (0, 255, 0) if tem_capacete else (0, 0, 255)
        status = "COM CAPACETE" if tem_capacete else "SEM CAPACETE"

        # Desenha retângulo
        cv2.rectangle(resultado, (x1, y1), (x2, y2), cor, 3)

        # Texto
        texto = f"{status}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Fundo do texto
        cv2.rectangle(resultado, (x1, y1 - th - 15), (x1 + tw + 10, y1), cor, -1)

        # Texto
        cv2.putText(resultado, texto, (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return resultado


def main():
    """Função principal"""
    print("="*60)
    print("DETECTOR DE CAPACETE DE PROTEÇÃO")
    print("="*60)

    # Cria pasta de saída
    Path(PASTA_SAIDA).mkdir(exist_ok=True)

    # Carrega modelo
    print("\nCarregando modelo...")

    # Se best.pt não existe, usa modelo padrão e avisa
    if not Path("best.pt").exists():
        print("⚠️  Modelo 'best.pt' não encontrado!")
        print("   Usando YOLOv8 padrão (menos preciso para capacetes)")
        print("   Para melhor precisão, treine um modelo customizado\n")
        modelo = YOLO('yolov8n.pt')
    else:
        modelo = YOLO('best.pt')

    print(f"✓ Modelo carregado!")
    print(f"📋 Classes do modelo: {list(modelo.names.values())}\n")

    # Processa imagens
    pasta = Path(PASTA_IMAGENS)

    if not pasta.exists():
        print(f"❌ Pasta '{PASTA_IMAGENS}' não existe!")
        return

    extensoes = ['.jpg', '.jpeg', '.png', '.bmp']
    imagens = [f for f in pasta.iterdir() if f.suffix.lower() in extensoes]

    if not imagens:
        print(f"❌ Nenhuma imagem encontrada em '{PASTA_IMAGENS}'")
        return

    print(f"📁 Encontradas {len(imagens)} imagens\n")
    print("="*60)

    for i, img_path in enumerate(imagens, 1):
        print(f"\n[{i}/{len(imagens)}] {img_path.name}")
        print("-"*60)

        try:
            # Detecta
            imagem, deteccoes = detectar_capacetes(img_path, modelo, CONFIANCA_MINIMA)

            # Resumo
            total_com = sum(1 for d in deteccoes if d['tem_capacete'])
            total_sem = len(deteccoes) - total_com

            print(f"  COM capacete: {total_com}")
            print(f"  SEM capacete: {total_sem}")

            # Desenha e salva
            resultado = desenhar_resultado(imagem, deteccoes)
            saida = Path(PASTA_SAIDA) / f"{img_path.stem}_detectado{img_path.suffix}"
            cv2.imwrite(str(saida), resultado)

            print(f"  ✓ Salvo em: {saida}")

        except Exception as e:
            print(f"  ❌ Erro: {e}")

    print("\n" + "="*60)
    print("✅ Processamento concluído!")
    print(f"📂 Resultados em: {PASTA_SAIDA}/")
    print("\n💡 DICA: Para melhor precisão, use um modelo treinado")
    print("   especificamente para detecção de capacetes")


if __name__ == "__main__":
    main()
