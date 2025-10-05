"""
Detector de Capacete usando Roboflow API
Usa modelo já treinado hospedado no Roboflow
"""

import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient
from pathlib import Path

# CONFIGURAÇÕES
PASTA_IMAGENS = "imagens"
PASTA_SAIDA = "resultados"

# MODELO ROBOFLOW
# Substitua pelo ID do modelo que você encontrou no Roboflow
# Formato: "workspace/projeto/versao"
# Exemplo: "hard-hat-workers-detection/1"
MODELO_ID = "yolov8-hat-detection/3"  # ⚠️ VOCÊ PRECISA SUBSTITUIR ISSO

# API KEY do Roboflow
API_KEY = "UCavLvCNLtLkvAL92w9H"


def detectar_capacetes_roboflow(imagem_path, client):
    """Detecta capacetes usando modelo Roboflow"""

    # Carrega imagem
    imagem = cv2.imread(str(imagem_path))
    if imagem is None:
        raise ValueError(f"Não foi possível carregar: {imagem_path}")

    # Faz inferência no modelo
    resultado = client.infer(str(imagem_path), model_id=MODELO_ID)

    # Converte para formato Supervision
    deteccoes = sv.Detections.from_inference(resultado)

    return imagem, deteccoes, resultado


def analisar_deteccoes(resultado):
    """Analisa resultados e conta capacetes"""

    predictions = resultado.get('predictions', [])

    com_capacete = 0
    sem_capacete = 0

    for pred in predictions:
        classe = pred.get('class', '').lower()

        # Identifica se tem CAPACETE DE SEGURANÇA (hardhat/helmet)
        if 'hardhat' in classe or 'helmet' in classe:
            com_capacete += 1
        # Se detectou explicitamente SEM capacete ou "hat" (cabeça sem proteção)
        elif 'no-hardhat' in classe or 'no-helmet' in classe or 'without' in classe or 'head' in classe or 'person' in classe or classe == 'hat':
            sem_capacete += 1

    return com_capacete, sem_capacete


def desenhar_resultado(imagem, deteccoes):
    """Desenha detecções na imagem com cores personalizadas"""

    resultado_img = imagem.copy()

    # Mapeia labels para exibição correta
    labels = []
    for i in range(len(deteccoes)):
        # Obtém a classe original
        if deteccoes.data and 'class_name' in deteccoes.data:
            classe = deteccoes.data['class_name'][i].lower()
        else:
            classe = deteccoes['class_name'][i].lower() if 'class_name' in deteccoes else ''

        # Se for "hat", exibe "no helmet"
        if classe == 'hat':
            labels.append("no helmet")
        else:
            labels.append(classe)

    # Cria anotadores do Supervision
    bounding_box_annotator = sv.BoxAnnotator(
        thickness=3,
        color=sv.Color.from_rgb_tuple((0, 255, 0))  # Verde padrão
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=0.7
    )

    # Anota a imagem
    resultado_img = bounding_box_annotator.annotate(
        scene=resultado_img,
        detections=deteccoes
    )
    resultado_img = label_annotator.annotate(
        scene=resultado_img,
        detections=deteccoes,
        labels=labels
    )

    return resultado_img


def main():
    """Função principal"""
    print("="*60)
    print("DETECTOR DE CAPACETE - ROBOFLOW API")
    print("="*60)

    # Verificações
    if MODELO_ID == "hard-hat-workers-detection/1":
        print("\n⚠️  ATENÇÃO: Configure o MODELO_ID correto!")
        print("\n   Exemplos de Model IDs públicos:")
        print("   - joseph-nelson/hard-hat-sample/3")
        print("   - roboflow-universe/hard-hat-workers/2")
        print("\n   Para encontrar:")
        print("   1. Vá em: https://universe.roboflow.com/")
        print("   2. Busque: 'hard hat detection'")
        print("   3. Escolha um modelo")
        print("   4. Na URL, o Model ID está no formato:")
        print("      universe.roboflow.com/workspace/projeto/model/versao")
        print("      Use: workspace/projeto/versao")
        print()

    # Cria pasta de saída
    Path(PASTA_SAIDA).mkdir(exist_ok=True)

    # Inicializa cliente Roboflow
    print("\nConectando ao Roboflow...")
    try:
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )
        print(f"✓ Conectado!")
        print(f"📋 Usando modelo: {MODELO_ID}\n")
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        print("\nVerifique:")
        print("  1. API_KEY está correta")
        print("  2. Conexão com internet")
        return

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
            imagem, deteccoes, resultado = detectar_capacetes_roboflow(img_path, client)

            # Analisa
            com_capacete, sem_capacete = analisar_deteccoes(resultado)

            print(f"  COM capacete: {com_capacete}")
            print(f"  SEM capacete: {sem_capacete}")
            print(f"  Total detectado: {len(deteccoes)}")

            # Mostra classes detectadas
            if resultado.get('predictions'):
                print(f"  Classes encontradas:")
                for pred in resultado['predictions'][:3]:  # Mostra até 3
                    classe = pred.get('class', 'desconhecido')
                    conf = pred.get('confidence', 0)
                    print(f"    - {classe} ({conf:.2%})")

            # Desenha e salva
            resultado_img = desenhar_resultado(imagem, deteccoes)
            saida = Path(PASTA_SAIDA) / f"{img_path.stem}_detectado{img_path.suffix}"
            cv2.imwrite(str(saida), resultado_img)

            print(f"  ✓ Salvo em: {saida}")

        except Exception as e:
            print(f"  ❌ Erro: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ Processamento concluído!")
    print(f"📂 Resultados em: {PASTA_SAIDA}/")


if __name__ == "__main__":
    main()
