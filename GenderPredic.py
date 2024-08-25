#Gender

#Implementación de una técnica de aprendizaje máquina sin el uso de un framework
#========================================================================================================#
#========================================================================================================#
#Se importan las librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import random


#========================================================================================================#
#=======================================FUNCIONES========================================================#
#========================================================================================================#
def sigmoid_function(X):
  return 1/(1+math.e**(-X))

def log_regression4(X, y, alpha, epochs):
  y_ = np.reshape(y, (len(y), 1)) # shape (150,1)
  N = len(X)
  theta = np.random.randn(len(X[0]) + 1, 1) #* initialize theta
  X_vect = np.c_[np.ones((len(X), 1)), X] #* Add x0 (column of 1s)
  avg_loss_list = []
  loss_last_epoch = 9999999
  for epoch in range(epochs):
    sigmoid_x_theta = sigmoid_function(X_vect.dot(theta)) # shape: (150,5).(5,1) = (150,1)
    grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_) # shapes: (5,150).(150,1) = (5, 1)
    best_params = theta
    theta = theta - (alpha * grad)
    hyp = sigmoid_function(X_vect.dot(theta)) #
    avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp)
    avg_loss_list.append(avg_loss)
    loss_step = abs(loss_last_epoch - avg_loss) #*
    loss_last_epoch = avg_loss #*

  return best_params

def train_test_split(X, Y, test_size):        
  Y=Y.to_frame('Y')#Se pasa de serie a columna para unirla
  data=pd.concat([X,Y],axis=1)

  data=data.sample(frac=1).reset_index(drop=True)#Se cambia el orden de los datos, se devuelven todas las filas
  #, se resetea el índice y se elimina el antiguo índice
  numero_para_test=int(len(data) * test_size) #Numero de filas que se tomarán

  test=data[:numero_para_test]
  train=data[numero_para_test:]

  X_train=train.drop(columns='Y')
  y_train=train['Y']
  X_test=test.drop(columns='Y')
  y_test=test['Y']

  return X_train, X_test, y_train, y_test


#=======================================================================================================#




#Llamamos los datos
df=pd.read_csv('student_performance_data.csv')
#Vamos a usar los features numéricos y se convertiran en numéricos los que no lo son.
df = pd.get_dummies(df, columns=['Major'], drop_first=True)

#Hacemos columnas binarias
df['Gender'] = df['Gender'].replace({'Female': 1, 'Male': 0})#
df['PartTimeJob'] = df['PartTimeJob'].replace({'Yes': 1, 'No': 0})#
df['ExtraCurricularActivities'] = df['ExtraCurricularActivities'].replace({'Yes': 1, 'No': 0})#
df['Major_Business'] = df['Major_Business'].astype(int)
df['Major_Education'] = df['Major_Education'].astype(int)
df['Major_Engineering'] = df['Major_Engineering'].astype(int)
df['Major_Science'] = df['Major_Science'].astype(int)


y_hombre = (df["Gender"] == 0).astype(int) # return 1 si es hombre, and 0 if not
y_mujer = (df["Gender"] == 1).astype(int)


X=df.drop(columns=['StudentID'])

# Listq de ys
y_niveles = [y_hombre, y_mujer]
y_niveles = {'Hombre':y_hombre,
                'Mujer':y_mujer}
predicted_probs = {'Hombre':0.0,
                   'Mujer':0.0}
actual_y = {'Hombre':0,
            'Mujer':0}


for key, y_nivel in y_niveles.items():
  # Split dataset (training and testing sets)
  X_train, X_test, y_train, y_test = train_test_split(X, y_nivel, test_size=0.2)

  # Scale X
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  # Train model
  epochs = 1000
  alpha = 1
  best_params = log_regression4(X_train, y_train, alpha, epochs)
  # Make predictions on test set
  index_ = 5
  X_to_predict = [list(X_test[index_])]
  # print(X_to_predict)
  X_to_predict = np.c_[np.ones((len(X_to_predict), 1)), X_to_predict] # add x0 for bias
  # print(X_to_predict)
  pred_probability = sigmoid_function(X_to_predict.dot(best_params))
  predicted_probs[key] = pred_probability[0][0]
  print('Our model calculated probability of sample being {}, is: {}%'.format(key, round(pred_probability[0][0]*100,2)))
  actual_y[key] = y_test.iloc[index_]

max_key = max(predicted_probs, key=predicted_probs.get)
print('\n', predicted_probs)
print('\nModel Prediction: {}'.format(max_key))
max_actual_y = max(actual_y, key=actual_y.get)
print('Real value is: {}'.format(max_actual_y))

