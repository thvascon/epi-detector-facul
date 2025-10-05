"""
Detector de Capacete em Tempo Real usando Webcam
Usa modelo já treinado hospedado no Roboflow
"""

import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# MODELO ROBOFLOW
MODELO_ID = "yolov8-hat-detection/3"

# API KEY do Roboflow
API_KEY = "UCavLvCNLtLkvAL92w9H"


def main():
    """Função principal - Detecção em tempo real"""
    print("="*60)
    print("DETECTOR DE CAPACETE - WEBCAM")
    print("="*60)
    print("\nPressione 'q' para sair")
    print("="*60)

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
        return

    # Inicializa webcam - tenta diferentes índices
    print("Procurando webcam...")
    cap = None

    for i in range(5):  # Tenta índices 0 a 4
        print(f"  Tentando índice {i}...")
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            cap = test_cap
            print(f"✓ Webcam encontrada no índice {i}!\n")
            break
        test_cap.release()

    if cap is None or not cap.isOpened():
        print("\n❌ Nenhuma webcam encontrada!")
        print("\nVerifique:")
        print("  1. A webcam está conectada")
        print("  2. Permissões de acesso à webcam")
        print("  3. Use: ls /dev/video* para ver dispositivos disponíveis")
        return

    # Cria anotadores do Supervision
    bounding_box_annotator = sv.BoxAnnotator(
        thickness=3,
        color=sv.Color.from_rgb_tuple((0, 255, 0))
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=0.7
    )

    # Contador de frames
    frame_count = 0

    while True:
        # Captura frame
        ret, frame = cap.read()

        if not ret:
            print("❌ Erro ao capturar frame")
            break

        # Processa a cada 3 frames para melhor performance
        if frame_count % 3 == 0:
            # Salva frame temporário
            cv2.imwrite('temp_frame.jpg', frame)

            try:
                # Faz inferência
                resultado = client.infer('temp_frame.jpg', model_id=MODELO_ID)

                # Converte para formato Supervision
                deteccoes = sv.Detections.from_inference(resultado)

                # Mapeia labels (hat -> no helmet)
                labels = []
                for i in range(len(deteccoes)):
                    if deteccoes.data and 'class_name' in deteccoes.data:
                        classe = deteccoes.data['class_name'][i].lower()
                    else:
                        classe = deteccoes['class_name'][i].lower() if 'class_name' in deteccoes else ''

                    if classe == 'hat':
                        labels.append("no helmet")
                    else:
                        labels.append(classe)

                # Desenha anotações
                frame = bounding_box_annotator.annotate(
                    scene=frame,
                    detections=deteccoes
                )
                frame = label_annotator.annotate(
                    scene=frame,
                    detections=deteccoes,
                    labels=labels
                )

                # Conta detecções
                com_capacete = sum(1 for label in labels if 'helmet' in label and 'no' not in label)
                sem_capacete = sum(1 for label in labels if 'no helmet' in label or label == 'hat')

                # Adiciona texto com contagem
                cv2.putText(frame, f"Com capacete: {com_capacete}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Sem capacete: {sem_capacete}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                # Em caso de erro, apenas mostra o frame
                cv2.putText(frame, f"Erro: {str(e)[:50]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostra frame
        cv2.imshow('Detector de Capacete - Pressione Q para sair', frame)

        # Verifica tecla pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()

    # Remove arquivo temporário
    import os
    if os.path.exists('temp_frame.jpg'):
        os.remove('temp_frame.jpg')

    print("\n✅ Programa encerrado!")


if __name__ == "__main__":
    main()
