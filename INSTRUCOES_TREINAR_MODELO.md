# 🎯 Como Treinar Modelo Próprio para Detecção de Capacetes

O YOLOv8 padrão **NÃO** detecta capacetes especificamente. Para ter boa precisão, você precisa de um modelo treinado.

## Opção 1: Usar Modelo Pré-Treinado Público (RECOMENDADO)

### Passo 1: Vá ao Roboflow Universe
https://universe.roboflow.com/

### Passo 2: Busque por "hard hat detection" ou "helmet detection"

### Passo 3: Escolha um dataset público e baixe o modelo
Exemplo de bons datasets:
- "Hard Hat Detection" (>5000 imagens)
- "PPE Detection" (Personal Protective Equipment)
- "Safety Helmet Detection"

### Passo 4: Baixe no formato YOLOv8
- Clique em "Download"
- Selecione formato "YOLOv8"
- Baixe o arquivo `best.pt` ou `weights/best.pt`

### Passo 5: Coloque o arquivo best.pt na pasta do projeto
```bash
cp caminho/do/best.pt /home/Estudo/Trabalho_Faculdade/best.pt
```

### Passo 6: Execute o detector
```bash
source venv/bin/activate
python detector-EPI-v2.py
```

---

## Opção 2: Treinar Seu Próprio Modelo

### Requisitos:
- Pelo menos 100 imagens anotadas (50 com capacete, 50 sem)
- Uso do Roboflow ou Label Studio para anotar

### Passo 1: Anotar Imagens no Roboflow

1. Crie conta grátis: https://roboflow.com
2. Crie novo projeto: "Detecção de Capacete"
3. Faça upload das suas imagens
4. Anote cada imagem:
   - Desenhe caixa em volta de pessoas COM capacete → classe "hardhat"
   - Desenhe caixa em volta de pessoas SEM capacete → classe "no-hardhat"

### Passo 2: Gerar Dataset

1. Clique em "Generate" → "Train/Test Split"
2. Use: 70% treino, 20% validação, 10% teste
3. Aplique augmentações (rotação, flip, brilho)

### Passo 3: Treinar Modelo

**Método A: Treinar no Roboflow (fácil)**
```python
# No Roboflow, clique em "Train" > YOLOv8
# Aguarde treinamento (demora ~30 min)
# Baixe o modelo treinado (best.pt)
```

**Método B: Treinar Localmente (avançado)**
```python
from ultralytics import YOLO

# Carrega modelo base
modelo = YOLO('yolov8n.pt')

# Treina (precisa do data.yaml do Roboflow)
modelo.train(
    data='path/to/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)

# Modelo treinado fica em: runs/detect/train/weights/best.pt
```

---

## Opção 3: Usar Modelo do Ultralytics HUB (Mais Fácil)

### Passo 1: Acesse Ultralytics HUB
https://hub.ultralytics.com/

### Passo 2: Busque modelo público
- Busque "hardhat" ou "helmet"
- Escolha modelo com boa performance (mAP > 0.8)

### Passo 3: Use o modelo direto pelo ID
```python
modelo = YOLO('hub://usuario/modelo_id')
```

---

## ⚠️ IMPORTANTE

**Sem modelo treinado**, o detector vai usar YOLOv8 padrão que:
- ❌ Não detecta capacetes diretamente
- ❌ Só detecta "person"
- ❌ Baixa precisão

**Com modelo treinado**:
- ✅ Detecta "hardhat" e "no-hardhat" diretamente
- ✅ Alta precisão (>90%)
- ✅ Funciona com qualquer cor de capacete

---

## 📝 Datasets Públicos Prontos

1. **Roboflow Universe - Hard Hat Detection**
   - URL: https://universe.roboflow.com/roboflow-universe/hard-hat-detection
   - 5000+ imagens
   - Já treinado, só baixar

2. **Kaggle - Hard Hat Detection**
   - URL: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
   - Dataset anotado pronto

3. **GitHub - PPE Detection**
   - Busque: "ppe detection yolov8 github"
   - Vários repositórios com modelos prontos
