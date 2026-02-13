# Bootcamp-de-Machine-Learning-Human-Segmentation-Dataset

Análise de Consistência dos Metadados

Objetivo
Verificar a integridade estrutural do dataset, avaliando a existência de valores ausentes no arquivo df.csv e a consistência dimensional entre imagens, máscaras e colagens.

Verificação de Valores Ausentes
Foi realizada a checagem de valores nulos nas colunas do CSV:

images: 0 valores ausentes

masks: 0 valores ausentes

collages: 0 valores ausentes

Conclusão: O arquivo de metadados está completo, sem registros faltantes.

Verificação de Consistência de Dimensões
Foi realizada a comparação entre as dimensões (largura e altura) das imagens originais e seus respectivos arquivos de máscara e colagem.

Resultados:

Total de arquivos verificados: 7845

Total de inconsistências de dimensão: 2615

As inconsistências identificadas ocorreram exclusivamente nas imagens da pasta collages.

Exemplo de inconsistência encontrada:
Arquivo: collages/0_00030.jpg
Dimensão esperada (imagem original): (540, 960)
Dimensão encontrada (colagem): (1626, 960)

Padrão Observado
As imagens da pasta collages apresentam largura aproximadamente três vezes maior que a imagem original, mantendo a mesma altura. Isso indica que as colagens são composições visuais (provavelmente imagem original + máscara + visualização combinada), e não arquivos destinados ao treinamento direto do modelo.

Conclusão Geral

O dataset não possui valores ausentes.

Imagens e máscaras possuem dimensões consistentes entre si.

As inconsistências identificadas referem-se apenas às colagens, o que é esperado devido à sua natureza de visualização.

O dataset encontra-se estruturalmente consistente e adequado para treinamento de modelos de segmentação, desde que apenas as pastas images e masks sejam utilizadas no pipeline de treinamento.