"""
##  NAME
       K_nearest_neighbors.py
##  VERSION
        [2.0]
##  AUTHOR
	Luz Rosario Guevara Cruz <lguevara@lcg.unam.mx>
	Ignacio Emmanuel Ramirez Bernabé <>
##  GITHUB REPOSITORY
	https://github.com/lro-guevara/Fungi-clustering
##  DATE
    2022/02/28
##  DESCRIPTION
    This program analyzes a dataset from UCI mushroom, train and
    evaluate the data to predict results.
##  CATEGORY
    Data analysis, machine learning.
##  USAGE
	Bioinformatic usage in Python 3.0
##  ARGUMENTS
    No arguments are required.
##Required library
    import numpy
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score, classification_report, \
    confusion_matrix, plot_confusion_matrix

##  INPUT
    <data.txt> Contains the 22 atributes of the 8124 instances found
    in "agaricus-lepiota.data"
    <target.txt> Contains the classes of the instances found in first
    column of"agaricus-lepiota.data".

OUTPUT
    <predicted_classes.png> Matrix confusion of the data.
    <classification_report_k.txt> Specific data about prediction
    with different k.

"""

import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score, classification_report, \
    confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt


# Crear lista de los datos de las caracteristicas
data = []
with open("data.txt") as file:
    # Obtener cada valor de cada caracteristica
    for line in file:
        line = line.rstrip().split(",")
        # Transformar el tipo de dato de caracter a int
        for i, characteristic in enumerate(line):
            line[i] = ord(characteristic)
        # Anadir los valores a la lista
        data.append(line)

# Transformar en array los valores de las caracteristicas
data = numpy.array(data)

# Crear lista de los datos de las clases
classes = []
with open("classes.txt") as file:
    # Obtener las clases sin espacios o tabs
    for line in file:
        line = line.rstrip()
        # Guardar en la lista
        classes.append(line)

# Transformar en array los valores de las clases
classes = numpy.array(classes)

# Separar los datos para los sets de datos de entrenamiento y evaluacion
x_train, x_test, y_train, y_test = train_test_split(data, classes,
                                                    test_size=0.30,
                                                    random_state=0)

print(len(x_train), len(x_test), len(y_train), len(y_test))

# Clasificación K Nearest neighbors
k = 1

# Definir el clasificador
classifier = KNeighborsClassifier(n_neighbors=k)

# Entrenar el clasificador con lo datos de entrenamiento y los
# valores de las clases
classifier.fit(x_train, y_train)

# Realizar la prediccion con el clasificador con los datos de evaluacion
y_predict = classifier.predict(x_test)

# Imprimir medidas de rendimiento del clasificador
print("Accuracy: {}".format(accuracy_score(y_test, y_predict)))
print("Precision: {}".format(precision_score(y_test, y_predict,
                                             average="macro")))
print("Recall: {}".format(recall_score(y_test, y_predict,
                                       average="macro")))
print("F-score: {}".format(f1_score(y_test, y_predict,
                                    average="macro")))

# Imprimir el reporte de las metricas de clasificacion
target_names = ['Edible', 'Poisonous']
print(classification_report(y_test, y_predict,
                            target_names=target_names))

# Guardar el reporte de las metricas de clasificacion
with open('classification_report_k1.txt', 'w') as f:
    f.write(classification_report(y_test, y_predict,
                                  target_names=target_names))

# Imprimir la matriz de confusion
print(confusion_matrix(y_test, y_predict))

# Generar la figura de la matriz de confusion
plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues,
                      display_labels=['Edible', 'Poisonous'])

# Guardar la figura de la matriz de confusion
plt.savefig('predicted_classes_k1.png')
