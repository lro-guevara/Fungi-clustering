import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score, classification_report, \
    confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt


data = []
with open("data.txt") as file:
    for line in file:
        line = line.rstrip().split(",")
        for i, characteristic in enumerate(line):
            line[i] = ord(characteristic)
        data.append(line)

data = numpy.array(data)

target = []
with open("target.txt") as file:
    for line in file:
        line = line.rstrip()
        target.append(line)

target = numpy.array(target)

# Separamos el dataset en dos: entrenamiento y evaluación
x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                    test_size=0.30,
                                                    random_state=0)

print(len(x_train), len(x_test), len(y_train), len(y_test))


# Clasificación K Nearest neighbors
k = 1

# Definición del clasificador
classifier = KNeighborsClassifier(n_neighbors=k)

# Entrenamiento del clasificador con lo datos de entrenamiento y valores
# de clase para cada ejemplo
classifier.fit(x_train, y_train)

# Predicción con el clasificador entrenado en los datos de evaluación
y_predict = classifier.predict(x_test)

# Medidas de rendimiento del clasificador
print("Accuracy: {}".format(accuracy_score(y_test, y_predict)))
print("Precision: {}".format(precision_score(y_test, y_predict,
                                             average="macro")))
print("Recall: {}".format(recall_score(y_test, y_predict,
                                       average="macro")))
print("F-score: {}".format(f1_score(y_test, y_predict,
                                    average="macro")))


# Imprimir reporte de las metricas de clasificacion
target_names = ['Poisonous', 'Edible']
print(classification_report(y_test, y_predict,
                            target_names=target_names))

with open('classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_predict,
                                  target_names=target_names))

# Imprimir matriz de confusion
print(confusion_matrix(y_test, y_predict))

plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues,
                      display_labels=['Poisonous', 'Edible'])

plt.savefig('predicted_classes.png')
