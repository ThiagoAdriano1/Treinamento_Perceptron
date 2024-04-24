# Dada uma rede Perceptron com três terminais de entrada, utilizando pesos iniciais W1 = 0,4; w2=−0,6; w3= 0,6 e limiar = 0,5, responda:
# Ensinar a rede com os dados (001, –1) e (110,+1), usando uma taxa de aprendizado η = 0,4;

import numpy as np

# Definindo a função de ativação

def step_function(x):
    return 1 if x>= 0 else -1

# inicializando os pesos limiar

pesos = np.array([0.4, -0.6, 0.6])
limiar = 0.5
taxa_aprendizado = 0.4

# Dados de treinamento

entradas = np.array([[0, 0, 1], [1, 1, 0]])
saidas = np.array([-1, 1])

# Treinamento da rede Perceptron

for i in range(len(entradas)):
    entrada = entradas[i]
    saida_esperada = saidas[i]
    saida_calculada = step_function(np.dot(entrada, pesos)-limiar)
    erro = saida_esperada - saida_calculada
    pesos = pesos + taxa_aprendizado * erro * entrada

# Exibindo os pesos atualizados

print(f'Pesos Atualizados: {pesos}')
