# Bootcamp-de-Machine-Learning-Human-Segmentation-Dataset

Segmentação Humana usando YOLOv8 para capturas de pessoas dançando em vídeos do TikTok

O desempenho do modelo foi avaliado utilizando as métricas mAP50 e mAP50-95. Enquanto o mAP50 avalia a capacidade de detecção com um limiar de IoU de 0.5, o mAP50-95 fornece uma avaliação mais rigorosa, considerando múltiplos limiares de IoU. Além disso, foram monitoradas as perdas de bounding box (box_loss) e segmentação (seg_loss), que indicam respectivamente o erro de localização e a qualidade das máscaras.

Para YOLOv8-seg a relação entre box_loss e seg_loss:
- box_loss alto + seg_loss baixo → caixa ruim, máscara boa
- box_loss baixo + seg_loss alto → caixa boa, máscara ruim
- ambos caindo → treino saudável

mAP50 - Mean Average Precision com IoU = 0.50 (“fácil”)
- mAP50 alto → modelo está acertando bem "de forma geral"
- É uma métrica mais permissiva
- Maior é melhor

mAP50-95
- 0.50, 0.55, 0.60, ..., 0.95 → Ele testa o modelo com critérios cada vez mais rígidos
- Mede precisão fina da segmentação.
- A tendência ao decorrer do treinamento é se aproximar do valor de mAP50, o que significa maior índice de aprendizagem.



Treinamento 

Epoch 1/50
seg_loss: 0.9295
box_loss: 0.5968
mAP50: 0.951
mAP50-95: 0.717

Epoch 10/50
seg_loss: 0.8859
box_loss: 0.5645
mAP50: 0.991
mAP50-95: 0.868

Epoch 20/50
seg_loss: 0.7633
box_loss: 0.4593
mAP50: 0.994
mAP50-95: 0.89

Epoch 30/50
seg_loss: 0.6614
box_loss: 0.3852
mAP50: 0.995
mAP50-95: 0.905

Epoch 40/50
seg_loss: 0.6177
box_loss: 0.344
mAP50: 0.995
mAP50-95: 0.913

Epoch 50/50
seg_loss: 0.4325
box_loss: 0.2156
mAP50: 0.995
mAP50-95: 0.928

Métricas para avaliação do modelo aprendidas no módulo 5 da unidade 3 para segmentação de imagens:

IoU (Intersection over Union) - Índice de Jaccard: Divisão entre
- Área da Interseção → pixels que são previstos corretamente
- Área da União → todos os pixels da previsão + reais

Dice Coefficient
- O Dice é muito parecido com o IoU, mas penaliza menos erros pequenos.
- Dice = 2 * (Área da Interseção) / (Área da Previsão + Área Real)

Resultado do modelo YOLO:
- IoU médio: 0.8361392639563694
- Dice médio: 0.9048590316219716

O modelo apresentou desempenho elevado na tarefa de segmentação, alcançando IoU médio de 0.83 e Dice coefficient de 0.90, indicando alta sobreposição entre as máscaras previstas e as máscaras ground truth.

MÉTRICAS
evaluate_model.py - calcular métricas após o treino (best.pt)
- mAP50: 0.9948394997093201
- mAP50-95: 0.9814636473535077
- Mask mAP50: 0.9949641577060931
- Mask mAP50-95: 0.9294933528548848

mAP50: Significa que, considerando IoU ≥ 50%, o modelo acerta praticamente todas as detecções.

mAP50-95: Mesmo com critérios muito rigorosos, o modelo mantém alta precisão. Isso indica que o modelo não apenas detecta as pessoas, mas segmenta com grande qualidade.

Mask mAP50: O contorno das pessoas foi identificado com grande fidelidade. Isso é excelente para um problema de segmentação humana.

Mask mAP50-95: Pequenas imprecisões nas bordas já reduzem essa métrica. Mesmo assim, acima de 0.90 é considerado muito bom.

O modelo apresentou desempenho extremamente elevado na tarefa de segmentação humana. As métricas de avaliação indicam um mAP50 de 0.9948 e mAP50-95 de 0.9814, demonstrando alta precisão na detecção das pessoas.

Na segmentação, os resultados também foram expressivos, com Mask mAP50 de 0.9949 e Mask mAP50-95 de 0.9294, indicando que as máscaras geradas pelo modelo possuem elevada fidelidade em relação às anotações do dataset.