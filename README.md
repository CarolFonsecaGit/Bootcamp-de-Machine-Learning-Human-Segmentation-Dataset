# Bootcamp-de-Machine-Learning-Human-Segmentation-Dataset

Análise de Qualidade das Imagens

Objetivo:
Verificar a integridade dos arquivos do dataset Human Full Body Segmentation (TikTok), garantindo que todas as imagens estejam acessíveis e não apresentem corrupção.

Estrutura Avaliada
O dataset contém três categorias principais de arquivos:

* images/ → imagens originais
* masks/ → máscaras de segmentação
* collages/ → imagens de visualização

Procedimento Realizado
Foi executado um script em Python que percorreu todas as entradas listadas no arquivo df.csv. Para cada caminho registrado, o script:

* Verificou se o arquivo existia fisicamente no diretório esperado
* Tentou abrir a imagem utilizando a biblioteca Pillow
* Aplicou o método img.verify() para detectar possíveis corrupções

Resultados Obtidos

* Images analisadas: 2615
* Masks analisadas: 2615
* Collages analisadas: 2615
* Total geral de arquivos verificados: 7845
* Arquivos corrompidos encontrados: 0

Conclusão
Todos os 7845 arquivos foram abertos e verificados com sucesso, sem detecção de corrupção ou ausência de arquivos. O dataset encontra-se íntegro e consistente, estando apto para as próximas etapas de análise exploratória, pré-processamento e treinamento de modelos de segmentação.

