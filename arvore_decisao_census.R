# Leitura da base de dados
base = read.csv('census.csv')

# Apagar a coluna X
base$X = NULL

# Tratamento dos campos categ?ricos
base$workclass = factor(base$workclass, levels = c(' Federal-gov', ' Local-gov', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay'), labels = c(1, 2, 3, 4, 5, 6, 7))
base$education = factor(base$education, levels = c(' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college'), labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
base$marital.status = factor(base$marital.status, levels = c(' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'), labels = c(1, 2, 3, 4, 5, 6, 7))
base$occupation = factor(base$occupation, levels = c(' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving'), labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
base$relationship = factor(base$relationship, levels = c(' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife'), labels = c(1, 2, 3, 4, 5, 6))
base$race = factor(base$race, levels = c(' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White'), labels = c(1, 2, 3, 4, 5))
base$sex = factor(base$sex, levels = c(' Female', ' Male'), labels = c(0, 1))
base$native.country = factor(base$native.country, levels = c(' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia'), labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41))
base$income = factor(base$income, levels = c(' <=50K', ' >50K'), labels = c(0, 1))

# Escalonamento
base[, 1] = scale(base[, 1])
base[, 3] = scale(base[, 3])
base[, 5] = scale(base[, 5])
base[, 11:13] = scale(base[, 11:13])

# Divisao entre treinamento e teste
library(caTools)
set.seed(1)
divisao = sample.split(base$income, SplitRatio = 0.85)
base_treinamento = subset(base, divisao == TRUE)
base_teste = subset(base, divisao == FALSE)


# Criando acesso a arvore de decisao
library(rpart)
library(rpart.plot)
classificador = rpart(formula = income ~ ., data = base_treinamento)
print(classificador)
##lembrando o formula = atributo classe ~ atributos previsores

rpart.plot(classificador)

# Criando o previsor agora com os testes
previsoes = predict(classificador, newdata = base_teste[-15], type = 'class')
##type class é pra deixar os dados em 0,1 e não em probabilidade

# Chamando a matriz de confusao para avaliar a performance
matriz_confusao = table(base_teste[,15], previsoes)
print(matriz_confusao)

# Chamando o accuracy
library(caret)
confusionMatrix(matriz_confusao)



# Fazendo a poda da arvore
##ela vai procurar onde estao os maiores erros e cortar
poda = classificador$cptable[which.min(classificador$cptable[, "xerror"]), "CP"]
print(classificador$cptable)
##procurou onde foi o split com menor erro

prune(classificador, poda)
##prune = poda
# Mesmo assim não melhorou =(
















