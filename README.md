# Segmentação de Corpo Inteiro - TikTok Dataset

Este repositório contém o código e os resultados da atividade de segmentação de imagens de corpo inteiro extraídas de vídeos do TikTok.

## Objetivo
O objetivo deste projeto é comparar diferentes abordagens para segmentar pessoas em imagens com fundos variados, utilizando 2615 amostras de imagens e suas respectivas máscaras manuais (ground truth).

A segmentação foi dividida em três métodos:
1. **Método 1 (Literatura): Segment Anything Model (FastSAM)**
2. **Método 2 (Literatura): Limiarização Clássica (Otsu)**
3. **Método 3 (Método Próprio): GrabCut + Refinamento Morfológico**

---

## 🚀 Como Executar o Pipeline

O código foi projetado para rodar todas as três abordagens em um único script, avaliar os resultados matematicamente e gerar imagens comparativas automaticamente.

### 1. Pré-requisitos
Você precisará do Python 3.8+ e das seguintes bibliotecas:
```bash
pip install torch torchvision ultralytics opencv-python scikit-learn matplotlib Pillow
```

### 2. Estrutura de Pastas Esperada
Certifique-se de que os dados estão organizados da seguinte forma junto ao script:
```text
/
├── images/               # As 2615 imagens originais (.png)
├── masks/                # As 2615 máscaras ground-truth (.png)
├── collages/             # (Opcional) As colagens visuais
├── segmentation_pipeline.py  # O script principal
└── README.md
```

### 3. Rodando o Código
Basta executar o script principal. Ele processará as imagens (usando uma amostra para economizar tempo de processamento) e salvará os resultados:
```bash
python segmentation_pipeline.py
```

---

## 🧠 Explicação dos Métodos

Abaixo explicamos o que cada método faz nos bastidores de forma simplificada:

### Método 1: FastSAM (A Inteligência Artificial "Faz-tudo")
*O que é:* O FastSAM é baseado na tecnologia de vanguarda criada pela Meta (antigo Facebook), chamada "Segment Anything Model". É uma Inteligência Artificial poderosa treinada para entender qualquer objeto em uma imagem.
*Como usamos:* Pedimos à IA para "segmentar tudo" que ela via na imagem e, em seguida, programamos o computador para selecionar a maior mancha encontrada, assumindo que a maior coisa na imagem seja a pessoa dançando.

### Método 2: Limiarização de Otsu (O Matemático Clássico)
*O que é:* Uma técnica antiga e muito rápida dos anos 1970. Ele tira toda a cor da imagem (deixa em preto e branco) e calcula matematicamente qual é o tom exato de cinza que melhor divide o "fundo" escuro do "frente" claro.
*Como usamos:* Após aplicar esse limiar matemático, usamos filtros morfológicos (que funcionam como lixas digitais) para tentar remover os "chuviscos" de fundo e preencher os buracos no corpo da pessoa.

### Método 3: GrabCut + Morfologia (Nosso Método Próprio)
*O que é:* O GrabCut é um algoritmo interativo fantástico. Normalmente, um usuário desenha um retângulo em volta da pessoa, e a IA tenta adivinhar o contorno exato.
*Como usamos:* Automatizamos esse processo. O nosso script desenha um "retângulo imaginário" no meio de todas as imagens, considerando que as pessoas no TikTok estão geralmente centralizadas. Depois que o GrabCut estima o corpo da pessoa, aplicamos nosso próprio pipeline de refinamento (aumento, preenchimento de furos e erosão) para garantir que a máscara selecionada seja inteiriça e sem vazamentos pelo cenário.

---

## 📊 Resultados e Métricas

Na pasta `results/` gerada após a execução, você encontrará:
1. `metrics_comparison.csv`: Os números puros detalhados.
2. `metrics_barplot.png`: Um gráfico de barras comparando a performance das três abordagens.
3. `visual_comparison.png`: Imagens lado-a-lado comparando a foto real com a máscara que o nosso código achou versus a máscara perfeita feita à mão.
4. `report.txt`: O relatório formal resumindo tudo.

### Conclusões Breves
O corpo humano não é um formato uniforme (as mãos movem, as pernas cruzam), e o fundo das casas no TikTok (móveis, quadros) confunde muito o algoritmo matemático (Otsu). Nosso método de **GrabCut+Morfologia** teve o melhor equilíbrio por usar a suposição espacial ("a pessoa está no centro") aliada ao tratamento digital agressivo das bordas.
O FastSAM sem uma instrução explícita de "onde" clicar na tela (prompt via bounding box) acabou selecionando o quadro geral do cenário como o maior objeto.
