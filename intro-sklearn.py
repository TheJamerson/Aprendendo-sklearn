
# Programa para Introdução a Biblioteca SciKit-Learn

from sklearn.naive_bayes import MultinomialNB

dados = [ 
    [1, 50, 1],
    [0, 1, 0],
    [0, 60, 1],
    [0, 100, 1],
    [1, 70, 0],
    [0, 3, 1], 
    [1, 1, 0],
    [1, 200, 1],
    [0, 2, 1],
    [1, 1, 1]
]

resultados = [1, 0, 1, 1, 1, 0, 0, 1, 1, 1]

clf = MultinomialNB()

treino_dados = dados[:7]
treino_resultados = resultados[:7]

teste_dados = dados[-3:]
teste_resultados = resultados[-3:]

clf.fit(treino_dados, treino_resultados)

previsao = clf.predict(teste_dados)

print(previsao)
print(teste_resultados) 

    
# Mais atualizações em breve!
