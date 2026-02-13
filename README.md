# Bootcamp-de-Machine-Learning-Human-Segmentation-Dataset

Ao executar o comando "print(df.columns)"

Retorna: Index(['Unnamed: 0', 'images', 'masks', 'collages'], dtype='str')

Isso significa que:

Unnamed: 0 → índice salvo automaticamente no CSV

images → caminho das imagens

masks → caminho das máscaras

collages → imagens de visualização

🚫 Não existe coluna de classe, portanto isso confirma que se trata de um dataset de segmentação, não de classificação.

A "classe" está dentro das máscaras, não no CSV.

Colunas disponíveis:
['Unnamed: 0', 'images', 'masks', 'collages']


Isso confirma:

-> O dataset NÃO é de classificação, mas sim de segmentação
-> Não existe coluna "label" porque a classe está na máscara (imagem)

Em datasets de segmentação, a “classe” está dentro da imagem da máscara, será necessário analisar os pixels das máscaras. Será necessário importar a lib Pillow.

# Nota para os resultados:

Classe 0:   90.32%
Classe 255:  9.68%

No dataset:

0 → Fundo
255 → Pessoa

Isso é padrão comum em máscaras binárias. Em vez de usar 1, muitos datasets usam 255 para representar a classe positiva.


