# 🚀 GUIA COMPLETO - Usar Modelo do Roboflow

## ✅ Passo a Passo (5 minutos)

### 1️⃣ Pegar API Key do Roboflow

1. Acesse: https://app.roboflow.com/
2. Crie conta grátis (ou faça login)
3. Vá em: **Settings** (canto superior direito) → **API Keys**
4. Copie sua **Private API Key**

### 2️⃣ Encontrar Modelo de Capacete

1. Acesse: https://universe.roboflow.com/
2. Na busca, digite: **"hard hat detection"**
3. Escolha um modelo com boa precisão (>85%)
   - Exemplos:
     - "Hard Hat Workers Detection"
     - "Construction Safety Detection"
     - "PPE Detection"

4. Clique no modelo escolhido
5. Vá na aba **"Model"** ou **"Deploy"**
6. Copie o **Model ID**
   - Formato: `workspace-name/project-name/version`
   - Exemplo: `hard-hat-workers/yolov8/3`

### 3️⃣ Configurar o Código

Edite o arquivo `detector-EPI-roboflow.py`:

```python
# Linha 14: Cole seu Model ID
MODELO_ID = "hard-hat-workers/yolov8/3"  # ⚠️ SUBSTITUA AQUI

# Linha 17: Cole sua API Key
API_KEY = "sua_api_key_aqui"  # ⚠️ SUBSTITUA AQUI
```

### 4️⃣ Executar

```bash
source venv/bin/activate
python detector-EPI-roboflow.py
```

---

## 📋 Exemplo Completo de Configuração

```python
# detector-EPI-roboflow.py (linhas 14-17)

MODELO_ID = "joseph-nelson/hard-hat-sample/3"
API_KEY = "abc123xyz789"  # Sua chave real aqui
```

---

## ❓ Onde Encontrar as Informações

### Model ID - Opção 1: Na URL
Quando você abre o modelo no Roboflow, olhe a URL:
```
https://universe.roboflow.com/joseph-nelson/hard-hat-sample/model/3
                             ↑           ↑              ↑
                          workspace   project       versão
```
Model ID = `joseph-nelson/hard-hat-sample/3`

### Model ID - Opção 2: Na Página do Modelo
Na seção **"Deploy"** → **"Hosted API"** → tem o model_id ali

### API Key
```
Settings → Roboflow API → Private API Key → Copiar
```

---

## 🎯 Modelos Recomendados

| Nome | Model ID (exemplo) | Precisão |
|------|-------------------|----------|
| Hard Hat Sample | joseph-nelson/hard-hat-sample/3 | 94% |
| Hard Hat Workers | roboflow-universe/hard-hat-workers/2 | 92% |
| Construction Safety | construction-site-safety/yolov8/5 | 89% |
| PPE Detection | ppe-detection-v2/yolov8/1 | 91% |

*(IDs podem variar - use o que aparece no seu projeto)*

---

## ✅ Checklist

- [ ] Criei conta no Roboflow
- [ ] Peguei minha API Key
- [ ] Encontrei um modelo de capacete
- [ ] Copiei o Model ID
- [ ] Editei detector-EPI-roboflow.py com MODELO_ID e API_KEY
- [ ] Executei: `python detector-EPI-roboflow.py`
- [ ] Verificar resultados na pasta `resultados/`

---

## ⚠️ Problemas Comuns

### Erro: "API Key inválida"
- Verifique se copiou a chave completa
- Use a **Private API Key**, não a Public

### Erro: "Model not found"
- Verifique o Model ID está no formato correto
- Certifique-se que o modelo é público ou você tem acesso

### Erro: "No such file or directory: inference"
- Execute: `pip install inference-sdk supervision`

---

## 🎉 Pronto!

Depois de configurar, suas imagens serão processadas automaticamente!

Os resultados aparecem em: **resultados/**
