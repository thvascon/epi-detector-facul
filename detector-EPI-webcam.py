
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

MODELO_ID = "yolov8-hat-detection/3"

API_KEY = "UCavLvCNLtLkvAL92w9H"


def main():
    print("="*60)
    print("DETECTOR DE CAPACETE - WEBCAM")
    print("="*60)
    print("\nPressione 'q' para sair")
    print("="*60)

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

    print("Procurando webcam...")
    cap = None

    for i in range(5):
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    bounding_box_annotator = sv.BoxAnnotator(
        thickness=3,
        color=sv.Color.from_rgb_tuple((0, 255, 0))
    )
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=0.7
    )

    frame_count = 0
    ultima_deteccao = None
    labels_cache = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Erro ao capturar frame")
            break

        # Processa a cada 5 frames para melhor performance
        if frame_count %  10 == 0:
            # Redimensiona frame antes de enviar (mais rápido)
            frame_pequeno = cv2.resize(frame, (320, 320))

            # Salva frame temporário com qualidade reduzida
            cv2.imwrite('temp_frame.jpg', frame_pequeno, [cv2.IMWRITE_JPEG_QUALITY, 70])

            try:
                # Faz inferência
                resultado = client.infer('temp_frame.jpg', model_id=MODELO_ID)

                # Converte para formato Supervision
                deteccoes_temp = sv.Detections.from_inference(resultado)

                # Escala as detecções de volta para o tamanho original
                if len(deteccoes_temp) > 0:
                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 320
                    deteccoes_temp.xyxy[:, [0, 2]] *= scale_x
                    deteccoes_temp.xyxy[:, [1, 3]] *= scale_y

                ultima_deteccao = deteccoes_temp

                # Mapeia labels
                labels_cache = []
                for i in range(len(ultima_deteccao)):
                    if ultima_deteccao.data and 'class_name' in ultima_deteccao.data:
                        classe = ultima_deteccao.data['class_name'][i].lower()
                    else:
                        classe = ultima_deteccao['class_name'][i].lower() if 'class_name' in ultima_deteccao else ''

                    # Mapeia labels (modelo detecta "hat" quando SEM capacete)
                    if 'hat' in classe and 'no' not in classe:
                        labels_cache.append("NO HELMET")
                    elif 'no' in classe or 'helmet' in classe:
                        labels_cache.append("Capacete")
                    else:
                        labels_cache.append(classe)

            except Exception as e:
                # Em caso de erro, mantém detecção anterior
                cv2.putText(frame, f"Erro: {str(e)[:50]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Desenha detecções em cache (mesmo quando não processa)
        if ultima_deteccao is not None and len(ultima_deteccao) > 0:
            frame = bounding_box_annotator.annotate(
                scene=frame,
                detections=ultima_deteccao
            )
            frame = label_annotator.annotate(
                scene=frame,
                detections=ultima_deteccao,
                labels=labels_cache
            )

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
