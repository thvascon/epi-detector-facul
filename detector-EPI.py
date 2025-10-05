"""
Detector de Capacete de Proteção usando YOLOv8
Detecta se pessoas em fotos estão usando capacete de proteção (amarelo de obra)
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse


class DetectorCapacete:
    def __init__(self, modelo_path=None, confianca_min=0.4):
        """
        Inicializa o detector de capacete

        Args:
            modelo_path: Caminho para o modelo YOLO. Se None, usa modelo padrão YOLOv8
            confianca_min: Confiança mínima para detecção (0-1)
        """
        # Se não especificou modelo, tenta usar modelo treinado para PPE (Personal Protective Equipment)
        if modelo_path is None:
            # Tenta usar modelo pré-treinado para detecção de EPIs/capacetes
            # Este modelo foi treinado especificamente para: hardhat, mask, no-hardhat, no-mask, etc
            try:
                print("Tentando carregar modelo pré-treinado para detecção de capacetes...")
                # Modelo do Roboflow treinado para detecção de capacetes
                modelo_path = 'https://github.com/RizwanMunawar/yolov8-object-tracking/releases/download/v1.0/ppe.pt'
                self.modelo = YOLO(modelo_path)
                print("✓ Modelo PPE carregado com sucesso!")
            except:
                print("⚠ Modelo PPE não disponível. Usando YOLOv8 padrão com detecção por pose...")
                modelo_path = 'yolov8n-pose.pt'  # Modelo com detecção de pose
                self.modelo = YOLO(modelo_path)
        else:
            self.modelo = YOLO(modelo_path)

        self.confianca_min = confianca_min

        # Verifica se o modelo tem classes relacionadas a capacete
        self.tem_classe_capacete = any(
            classe in str(self.modelo.names).lower()
            for classe in ['hardhat', 'helmet', 'hat', 'ppe']
        )

    def detectar_em_imagem(self, imagem_path):
        """
        Detecta capacetes em uma imagem

        Args:
            imagem_path: Caminho para a imagem

        Returns:
            dict com resultados da detecção
        """
        # Carrega imagem
        imagem = cv2.imread(str(imagem_path))
        if imagem is None:
            raise ValueError(f"Não foi possível carregar a imagem: {imagem_path}")

        # Executa detecção
        resultados = self.modelo(imagem, conf=self.confianca_min)

        # Processa resultados
        deteccoes = []
        for resultado in resultados:
            boxes = resultado.boxes
            for box in boxes:
                # Extrai informações
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confianca = float(box.conf[0])
                classe_id = int(box.cls[0])
                classe_nome = self.modelo.names[classe_id]

                # Determina se tem capacete baseado na classe detectada
                tem_capacete = False
                classe_lower = classe_nome.lower()

                # Se o modelo detectou diretamente hardhat/helmet
                if any(palavra in classe_lower for palavra in ['hardhat', 'helmet', 'hat']):
                    tem_capacete = True

                # Se detectou "no-hardhat" ou "no-helmet" explicitamente
                elif any(palavra in classe_lower for palavra in ['no-hardhat', 'no-helmet', 'without']):
                    tem_capacete = False

                # Se é pessoa e o modelo não tem classe específica de capacete, tenta detectar
                elif 'person' in classe_lower and not self.tem_classe_capacete:
                    tem_capacete = self._detecta_capacete_na_cabeca(imagem, int(x1), int(y1), int(x2), int(y2))

                deteccoes.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confianca': confianca,
                    'classe': classe_nome,
                    'tem_capacete': tem_capacete
                })

        return {
            'imagem': imagem,
            'deteccoes': deteccoes,
            'total_pessoas': sum(1 for d in deteccoes if 'person' in d['classe'].lower()),
            'total_capacetes': sum(1 for d in deteccoes if d['tem_capacete'])
        }

    def _verifica_capacete(self, classe_nome):
        """Verifica se a classe detectada é um capacete"""
        palavras_capacete = ['helmet', 'hardhat', 'safety', 'hat']
        return any(palavra in classe_nome.lower() for palavra in palavras_capacete)

    def _detecta_capacete_na_cabeca(self, imagem, x1, y1, x2, y2):
        """
        Detecta se há capacete na região da cabeça da pessoa
        Usa análise de bordas e forma arredondada para detectar qualquer capacete

        Args:
            imagem: Imagem BGR
            x1, y1, x2, y2: Coordenadas do bounding box da pessoa

        Returns:
            True se detectou capacete, False caso contrário
        """
        altura = y2 - y1

        # Define região da cabeça (25% superior do corpo)
        cabeca_y1 = max(0, y1)
        cabeca_y2 = min(imagem.shape[0], int(y1 + altura * 0.28))
        cabeca_x1 = max(0, x1)
        cabeca_x2 = min(imagem.shape[1], x2)

        # Extrai região da cabeça
        regiao_cabeca = imagem[cabeca_y1:cabeca_y2, cabeca_x1:cabeca_x2]

        if regiao_cabeca.size == 0 or regiao_cabeca.shape[0] < 20 or regiao_cabeca.shape[1] < 20:
            return False

        # Converte para escala de cinza
        gray = cv2.cvtColor(regiao_cabeca, cv2.COLOR_BGR2GRAY)

        # Detecta bordas
        bordas = cv2.Canny(gray, 50, 150)

        # Conta pixels de borda na metade superior (onde fica o capacete)
        altura_regiao = bordas.shape[0]
        metade_superior = bordas[0:int(altura_regiao * 0.6), :]

        total_pixels_superior = metade_superior.size
        pixels_borda = cv2.countNonZero(metade_superior)
        densidade_bordas = (pixels_borda / total_pixels_superior) * 100

        # Detecta círculos (formato arredondado do capacete)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        # Se detectou círculo NA PARTE SUPERIOR da região OU densidade de bordas alta, tem capacete
        tem_circulo_superior = False
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cy = circle[1]
                # Verifica se o círculo está na metade superior
                if cy < altura_regiao * 0.6:
                    tem_circulo_superior = True
                    break

        # Critérios para detectar capacete:
        # 1. Círculo detectado na parte superior OU
        # 2. Alta densidade de bordas na parte superior (objeto sólido como capacete)
        return tem_circulo_superior or densidade_bordas > 8

    def desenhar_deteccoes(self, imagem, deteccoes):
        """
        Desenha bounding boxes e labels na imagem

        Args:
            imagem: Imagem numpy array
            deteccoes: Lista de detecções

        Returns:
            Imagem com detecções desenhadas
        """
        imagem_resultado = imagem.copy()

        for det in deteccoes:
            x1, y1, x2, y2 = det['bbox']
            tem_capacete = det['tem_capacete']
            confianca = det['confianca']
            classe = det['classe']

            # Define cor: Verde se tem capacete, Vermelho se não tem
            if tem_capacete:
                cor = (0, 255, 0)  # Verde
                status = "COM CAPACETE"
            else:
                cor = (0, 0, 255)  # Vermelho
                status = "SEM CAPACETE" if 'person' in classe.lower() else classe.upper()

            # Desenha retângulo
            cv2.rectangle(imagem_resultado, (x1, y1), (x2, y2), cor, 2)

            # Prepara texto
            texto = f"{status}: {confianca:.2f}"

            # Calcula tamanho do texto para fundo
            (texto_w, texto_h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Desenha fundo do texto
            cv2.rectangle(imagem_resultado, (x1, y1 - texto_h - 10),
                         (x1 + texto_w, y1), cor, -1)

            # Desenha texto
            cv2.putText(imagem_resultado, texto, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return imagem_resultado

    def processar_e_salvar(self, imagem_path, saida_path=None):
        """
        Processa imagem e salva resultado

        Args:
            imagem_path: Caminho da imagem de entrada
            saida_path: Caminho para salvar resultado (opcional)
        """
        print(f"Processando: {imagem_path}")

        # Detecta
        resultado = self.detectar_em_imagem(imagem_path)

        # Desenha detecções
        imagem_resultado = self.desenhar_deteccoes(resultado['imagem'], resultado['deteccoes'])

        # Exibe resumo
        print(f"  Pessoas detectadas: {resultado['total_pessoas']}")
        print(f"  Capacetes detectados: {resultado['total_capacetes']}")

        for i, det in enumerate(resultado['deteccoes'], 1):
            status = "COM capacete" if det['tem_capacete'] else "SEM capacete"
            print(f"  Detecção {i}: {det['classe']} ({status}) - Confiança: {det['confianca']:.2%}")

        # Salva resultado
        if saida_path is None:
            path = Path(imagem_path)
            saida_path = path.parent / f"{path.stem}_detectado{path.suffix}"

        cv2.imwrite(str(saida_path), imagem_resultado)
        print(f"  Resultado salvo em: {saida_path}\n")

        return resultado


def main():
    """Função principal com exemplo de uso"""
    parser = argparse.ArgumentParser(description='Detector de Capacete de Proteção')
    parser.add_argument('imagem', type=str, help='Caminho para a imagem ou pasta com imagens')
    parser.add_argument('--modelo', type=str, default='yolov8n.pt',
                       help='Caminho do modelo YOLO (padrão: yolov8n.pt)')
    parser.add_argument('--confianca', type=float, default=0.5,
                       help='Confiança mínima (0-1, padrão: 0.5)')
    parser.add_argument('--output', type=str, help='Pasta de saída para resultados')

    args = parser.parse_args()

    # Inicializa detector
    print("Carregando modelo...")
    detector = DetectorCapacete(modelo_path=args.modelo, confianca_min=args.confianca)
    print("Modelo carregado!\n")

    # Processa imagem ou pasta
    caminho = Path(args.imagem)

    if caminho.is_file():
        # Processa uma imagem
        saida = Path(args.output) if args.output else None
        detector.processar_e_salvar(caminho, saida)

    elif caminho.is_dir():
        # Processa todas as imagens da pasta
        extensoes = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        imagens = [f for f in caminho.iterdir() if f.suffix.lower() in extensoes]

        print(f"Encontradas {len(imagens)} imagens\n")

        for imagem in imagens:
            if args.output:
                saida = Path(args.output) / f"{imagem.stem}_detectado{imagem.suffix}"
            else:
                saida = None

            try:
                detector.processar_e_salvar(imagem, saida)
            except Exception as e:
                print(f"Erro ao processar {imagem}: {e}\n")

    else:
        print(f"Erro: {caminho} não é um arquivo ou pasta válida")


if __name__ == "__main__":
    # CONFIGURAÇÃO: Defina a pasta com as imagens aqui
    PASTA_IMAGENS = "imagens"  # Altere para o caminho da sua pasta
    PASTA_SAIDA = "resultados"  # Pasta onde os resultados serão salvos
    CONFIANCA_MINIMA = 0.5  # Confiança mínima (0.0 a 1.0)

    # Cria pasta de saída se não existir
    Path(PASTA_SAIDA).mkdir(exist_ok=True)

    # Inicializa detector
    print("Carregando modelo...")
    detector = DetectorCapacete(confianca_min=CONFIANCA_MINIMA)
    print("Modelo carregado!\n")

    # Processa todas as imagens da pasta
    caminho_pasta = Path(PASTA_IMAGENS)

    if not caminho_pasta.exists():
        print(f"⚠️  ERRO: A pasta '{PASTA_IMAGENS}' não existe!")
        print(f"   Crie a pasta e coloque suas imagens nela, depois rode o código novamente.")
        exit(1)

    # Busca todas as imagens
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    imagens = [f for f in caminho_pasta.iterdir() if f.suffix.lower() in extensoes]

    if not imagens:
        print(f"⚠️  Nenhuma imagem encontrada na pasta '{PASTA_IMAGENS}'")
        print(f"   Formatos aceitos: {', '.join(extensoes)}")
        exit(1)

    print(f"📁 Encontradas {len(imagens)} imagens na pasta '{PASTA_IMAGENS}'\n")
    print("=" * 60)

    # Processa cada imagem
    for i, imagem in enumerate(imagens, 1):
        print(f"\n[{i}/{len(imagens)}] {imagem.name}")
        print("-" * 60)

        saida = Path(PASTA_SAIDA) / f"{imagem.stem}_detectado{imagem.suffix}"

        try:
            detector.processar_e_salvar(imagem, saida)
        except Exception as e:
            print(f"❌ Erro ao processar {imagem.name}: {e}\n")

    print("\n" + "=" * 60)
    print(f"✅ Processamento concluído!")
    print(f"📂 Resultados salvos em: {PASTA_SAIDA}/")

    # OU use via linha de comando (descomente a linha abaixo):
    # main()
