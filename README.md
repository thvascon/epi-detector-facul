# 🦺 Detector de Capacete de Proteção

Detecta se pessoas em fotos estão usando capacete de proteção (amarelo de obra) usando YOLOv8.

## 📦 Instalação

```bash
# 1. Criar ambiente virtual
python3 -m venv venv

# 2. Ativar ambiente virtual
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt
```

## 🚀 Como Usar

### Modo 1: Uma imagem

```bash
source venv/bin/activate
python detector-EPI.py foto_obra.jpg
```

### Modo 2: Pasta com várias imagens

```bash
source venv/bin/activate
python detector-EPI.py pasta_fotos/
```

### Modo 3: Com opções personalizadas

```bash
# Ajustar confiança mínima (0.0 a 1.0)
python detector-EPI.py foto.jpg --confianca 0.4

# Especificar pasta de saída
python detector-EPI.py foto.jpg --output resultados/

# Usar modelo customizado
python detector-EPI.py foto.jpg --modelo meu_modelo.pt
```

## 🎯 Resultado

O script cria uma imagem com:
- ✅ **Caixa VERDE** = pessoa COM capacete
- ❌ **Caixa VERMELHA** = pessoa SEM capacete
- Confiança da detecção em percentual

A imagem processada é salva automaticamente com sufixo `_detectado`.

## 📝 Exemplo no Código

```python
from detector-EPI import DetectorCapacete

# Criar detector
detector = DetectorCapacete(confianca_min=0.5)

# Processar imagem
resultado = detector.processar_e_salvar('foto.jpg')

# Ver resultados
print(f"Pessoas: {resultado['total_pessoas']}")
print(f"Capacetes: {resultado['total_capacetes']}")
```

## ⚠️ Importante

O modelo padrão YOLOv8 (yolov8n.pt) detecta pessoas e alguns objetos gerais.

**Para melhor precisão com capacetes amarelos de obra:**
1. Use um modelo treinado especificamente para EPIs
2. Você pode encontrar modelos pré-treinados em:
   - [Roboflow Universe](https://universe.roboflow.com/)
   - [Ultralytics HUB](https://hub.ultralytics.com/)
3. Ou treinar seu próprio modelo com dataset customizado

## 🔧 Estrutura do Projeto

```
.
├── detector-EPI.py      # Código principal
├── requirements.txt     # Dependências
├── venv/               # Ambiente virtual (criado após instalação)
└── README.md           # Este arquivo
```

## 📚 Dependências

- **ultralytics** - Framework YOLOv8
- **opencv-python** - Processamento de imagem
- **numpy** - Operações numéricas
- **torch/torchvision** - Deep learning (instalado automaticamente)
# epi-detector-facul
