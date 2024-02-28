import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

# Importando a base de dados
base = pd.read_csv('../recursos/insurance.csv')

# Visualizando as informações da base
# print(base.shape)
# print(base.head())

# Removendo colunas que não são necessárias
base = base.drop(columns=['Unnamed: 0'])
# print(base.head(2))

# Separar a variavel dependende das variaveis preditoras
y = base.iloc[:, 7].values # Separa a coluna Accident
X = base.iloc[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values

# print(y)
# print(X)

labelencoder = LabelEncoder()
for i in range(X.shape[1]):
    if X[:,i].dtype == 'object':
        X[:,i] = labelencoder.fit_transform(X[:,i])

# print(X)

# Separar os dados em treino e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X,y,test_size=0.3,random_state=1)

modelo = GaussianNB()
modelo.fit(X_treinamento, y_treinamento)

previsoes = modelo.predict(X_teste)
# print(previsoes)

accuracy = accuracy_score(y_teste, previsoes)
precision = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')
print('-'*80)
print(f'Acuracia: {(accuracy * 100):.2f}%\t|\tPrecisão: {(precision * 100):.2f}%\t|\tRecall: {(recall * 100):.2f}%,\t|\tF1: {(f1 * 100):.2f}%')

report = classification_report(y_teste, previsoes)
print(report)

# Gerando a matriz de confusão
confusao = ConfusionMatrix(modelo, classes=['None', 'Severe', 'Mild','Moderate'])
confusao.fit(X_treinamento, y_treinamento)
confusao.score(X_teste, y_teste)
confusao.poof()
plt.show()
