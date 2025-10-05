## ⚠️ PROBLEMA IDENTIFICADO

O YOLOv8 padrão **NÃO detecta capacetes**. Ele só detecta "person" (pessoa).

Para detectar capacetes corretamente você precisa de um **modelo treinado especificamente para capacetes**.

---

## ✅ SOLUÇÃO RECOMENDADA

Baixe um modelo já treinado de um destes links:

### Opção 1: Modelo pronto do GitHub (MAIS FÁCIL)
```bash
# Baixar modelo treinado para hard hat detection
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-hardhat.pt -O best.pt
```

### Opção 2: Roboflow Universe
1. Acesse: https://universe.roboflow.com/
2. Busque: "hard hat detection yolov8"
3. Escolha um projeto com >90% precisão
4. Clique em "Download this Dataset" → YOLOv8 → Download
5. Use o arquivo `best.pt` que vem no download

### Opção 3: Kaggle
1. Acesse: https://www.kaggle.com/search?q=hard+hat+detection+yolov8
2. Baixe modelo pré-treinado
3. Coloque como `best.pt` na pasta do projeto

---

## 🎯 COMO USAR DEPOIS DE TER O MODELO

```bash
# 1. Coloque o modelo treinado como best.pt
cp /caminho/do/modelo/best.pt .

# 2. Execute
source venv/bin/activate
python detector-EPI-v2.py
```

---

## 🔧 ALTERNATIVA: Treinar Seu Próprio Modelo

Siga as instruções em: [INSTRUCOES_TREINAR_MODELO.md](INSTRUCOES_TREINAR_MODELO.md)

---

## ❌ POR QUE O CÓDIGO ATUAL NÃO FUNCIONA BEM?

- YOLOv8 padrão detecta 80 classes (pessoa, carro, gato, etc)
- **NÃO** detecta capacetes específicos
- A detecção por cor/forma é imprecisa
- **SOLUÇÃO**: Use modelo treinado para capacetes (hardhat)

---

## 📊 MODELOS DISPONÍVEIS

| Modelo | Precisão | Link |
|--------|----------|------|
| Hard Hat Workers | 95%+ | https://universe.roboflow.com/roboflow-universe/hard-hat-workers |
| PPE Detection | 90%+ | https://universe.roboflow.com/roboflow-universe/ppe-detection |
| Construction Safety | 92%+ | https://universe.roboflow.com/construction/construction-safety-gsnvb |
