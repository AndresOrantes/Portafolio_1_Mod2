#GPA, pero esta vez predice tu calificación en base a tu
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
#Creación de una columna categórica para el GPA. Según investigué un GPA mayor a 3.7 a más de 90.
def categorizacion_gpa(gpa):
    if gpa >= 3.7:
        return 0 #Calificaci'on alta
    elif gpa >= 2.7:
        return 1#Media
    else:
        return 2#Baja
df['GPA_Categoria'] = df['GPA'].apply(categorizacion_gpa)#Hace una nueva columna categórica que es numérica.

#Hacemos columnas binarias
df['PartTimeJob'] = df['PartTimeJob'].replace({'Yes': 1, 'No': 0})#
df['ExtraCurricularActivities'] = df['ExtraCurricularActivities'].replace({'Yes': 1, 'No': 0})#
df['Gender'] = df['Gender'].replace({'Female': 1, 'Male': 0})#

y_alto = (df["GPA_Categoria"] == 0).astype(int) # 
y_medio = (df["GPA_Categoria"] == 1).astype(int)
y_bajo = (df["GPA_Categoria"] == 2).astype(int)


X=df.drop(columns=['GPA_Categoria','GPA','StudentID','Gender','Major','ExtraCurricularActivities','PartTimeJob'])

# Listq de ys
y_niveles = [y_alto, y_medio, y_bajo]
y_niveles = {'GPA Alto':y_alto,
                'GPA Medio':y_medio,
                'GPA Bajo':y_bajo}
predicted_probs = {'GPA Alto':0.0,
                   'GPA Medio':0.0,
                   'GPA Bajo':0.0}
actual_y = {'GPA Alto':0,
            'GPA Medio':0,
            'GPA Bajo':0}

valido=False
"""while valido==0 :
  predecir = input("Ingrese los valores para las características separados por comas: ")
  predecir = [float(x) for x in predecir.split(',')]
  if len(predecir) != X.shape[1]:
    print(f"Por favor, ingrese {X.shape[1]} características.")
  else:
    valido=True"""


#Es más sencillo para todos si lo hago así
while valido==False:
  Edad =int(input("Ingresa tu edad: "))
  Horas=int(input("Ingresa las horas que invertiste en tu estudio: "))
  Asistencias=int(input("Ingresa tu asistencia(0-100): "))
  if Edad!=None and Horas!=None and Asistencias!=None:
    predecir=(Edad, Horas, Asistencias)
    valido=True
  else:
    print('Revisa y vuelve a ingresar tus datos')


escalarysesgo=False
for key, y_nivel in y_niveles.items():
  # Split dataset (training and testing sets)
  #X_train, X_test, y_train, y_test = train_test_split(X, y_nivel, test_size=0.2)
  X_train=X #Pues no creo que sea necesario la separación.
  y_train=y_nivel
  # Scale X
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  # Train model
  epochs = 1000
  alpha = 1
  best_params = log_regression4(X_train, y_train, alpha, epochs)

  if escalarysesgo==False:
    # Paréntesis: se escalan los datos ingresados, pero se tienen que escalar con el standard scaler,
    #y el StandardScaler está dentro del for, por lo que se me ocurrió hacer esto, para que se prediga una vez
    predecir = sc.fit_transform([predecir]) 
    predecir = np.c_[np.ones((len(predecir), 1)), predecir] #Se añade x0 para el sesgo
    escalarysesgo=True

  # Predicción y su probabilidad
  pred_probability = sigmoid_function(predecir.dot(best_params)) 
  predicted_probs[key] = pred_probability[0][0]
  print('Our model calculated probability of sample being {}, is: {}%'.format(key, round(pred_probability[0][0]*100,2)))
  #print('El modelo calculó que la probabilidad de que la muestra sea', key, 'es:',round(pred_probability[0][0]*100,2))
max_key = max(predicted_probs, key=predicted_probs.get)
print("=========================================================================")
print("=========================================================================")
print("=========================================================================")
print("=========================================================================")
print('\La predicción indica que es probable que tengas un {}'.format(max_key))
