# Leitura da base de dados
base = read.csv('census.csv')

# Apagar a coluna X
base$X = NULL

# Tratamento dos campos categoricos
### Para esse algoritmo nao e necessario fazer a mudanca de variavel categorica 
### para variavel numerica, na verdade e mais se nao fizer, pois a leitura e 
### desenvolvimento das regras e mais facil com as proprias string
#base$workclass = factor(base$workclass, levels = c(' Federal-gov', ' Local-gov', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay'), labels = c(1, 2, 3, 4, 5, 6, 7))
#base$education = factor(base$education, levels = c(' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college'), labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
#base$marital.status = factor(base$marital.status, levels = c(' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'), labels = c(1, 2, 3, 4, 5, 6, 7))
#base$occupation = factor(base$occupation, levels = c(' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving'), labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
#base$relationship = factor(base$relationship, levels = c(' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife'), labels = c(1, 2, 3, 4, 5, 6))
#base$race = factor(base$race, levels = c(' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White'), labels = c(1, 2, 3, 4, 5))
#base$sex = factor(base$sex, levels = c(' Female', ' Male'), labels = c(0, 1))
#base$native.country = factor(base$native.country, levels = c(' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia'), labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41))
#base$income = factor(base$income, levels = c(' <=50K', ' >50K'), labels = c(0, 1))

# Escalonamento
### Escalonamento tambem nao e necessario e aconselhavel nao fazer, 
### pois, caso faça, os numeros ficam muito pequenos e nao da diferenca nos resultados
#base[, 1] = scale(base[, 1])
#base[, 3] = scale(base[, 3])
#base[, 5] = scale(base[, 5])
#base[, 11:13] = scale(base[, 11:13])

# Divisao entre treinamento e teste
# Lembrando que algoritmo de teste e muito lento, por isso fazer a divisao no 5%
library(caTools)
set.seed(1)
divisao = sample.split(base$income, SplitRatio = 0.05)
base_treinamento = subset(base, divisao == TRUE)
base_teste = subset(base, divisao == FALSE)

# Inciando o aprendizado por regras com a biblioteca RougtSets
#install.packages('RoughSets')
library(RoughSets)

# Convertendo as bases para a biblioteca poder ler
# Transformando o dataframe em decision table
dt_treinamento = SF.asDecisionTable(dataset = base_treinamento, decision.attr = 15)
dt_teste = SF.asDecisionTable(dataset = base_teste, decision.attr = 15)
##decisionTable (tabela que eu quero, posicao do atributo classe)


# Antes de inciar a classificacao temos que separar em intervalos os atributos
# o cn2Rules não funciona valores numericos por isso a divisao
# Lembrando que fazemos a discretizacao que é passar de numericos para categoricos

intervalos = D.discretization.RST(dt_treinamento, nOfIntervals = 4) #numero de intervalos eu escolho
##applyDecTable já faz a separacao nos intervalos por mim
dt_treinamento = SF.applyDecTable(dt_treinamento, intervalos)
dt_teste = SF.applyDecTable(dt_teste, intervalos)
# Agora podemos criar o classificador
classificador = RI.CN2Rules.RST(dt_treinamento, K = 1) #k e grau de complexidade das regras 
print(classificador)

# Analisando com predict
previsoes = predict(classificador, newdata = dt_teste[-15])

# Fazendo a comparação agora
matriz_confusao = table(dt_teste[, 15], unlist(previsoes))
##unlist conserto o erro de previsoes nao estar no mesmo padrao que o dt
print(matriz_confusao)
library(caret)
confusionMatrix(matriz_confusao)
##vemos que regras não é tão bom assim
