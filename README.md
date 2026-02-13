# Bootcamp-de-Machine-Learning-Human-Segmentation-Dataset

Análise de Duplicatas do Dataset

Objetivo
Verificar a existência de registros duplicados no arquivo df.csv, garantindo que não haja repetição de dados que possa comprometer a qualidade da análise ou o treinamento do modelo.

Estrutura Avaliada
O arquivo CSV contém as seguintes colunas:

* images → caminhos das imagens originais
* masks → caminhos das máscaras de segmentação
* collages → caminhos das imagens de visualização

Procedimento Realizado
Foi executado um script em Python utilizando a biblioteca pandas para:

* Listar as colunas presentes no CSV
* Verificar duplicatas de linhas completas
* Verificar duplicatas individualmente em cada coluna
* Realizar verificação cruzada considerando todos os caminhos juntos

Resultados Obtidos

* Linhas totalmente duplicadas no CSV: 0
* Duplicatas na coluna images: 0
* Duplicatas na coluna masks: 0
* Duplicatas na coluna collages: 0
* Duplicatas considerando todas as colunas combinadas: 0

Conclusão
Nenhuma duplicata foi identificada no dataset. Cada registro possui caminhos únicos para imagens, máscaras e collages.

O dataset apresenta consistência estrutural e está adequado para as próximas etapas de análise, pré-processamento e treinamento de modelos de segmentação, sem risco de viés causado por dados repetidos.

