import sklearn as skl
print(skl.__version__)

# Importamos el dataset
import sklearn.datasets
dataset = sklearn.datasets.load_iris()

from sklearn.model_selection import train_test_split

# Renombramos los valores para que X sean los atributos e Y sean las respuestas del sistema
X = dataset.data
y = dataset.target

# Realizamos la partición de nuestro dataset en un conjunto de entrenamiento y otro de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

# Creamos el clasificador SVM lineal
classifier =  SVC(kernel="linear", C=0.025)

# Realizamos el entrenamiento
classifier.fit(X_train, y_train)

# Obtenemos el accuracy de nuestro modelo para el conjunto de test
print(classifier.score(X_test, y_test))

# Importamos la función de entrenamiento y validación cruzada
from sklearn.model_selection import cross_val_score
# Usamos otro clasificador no entrenamos
classifier_cross =  SVC(kernel="linear", C=0.025)
nScores = cross_val_score(classifier_cross, X, y, cv=10)
# Nos devuelve un array de tipo Numpy. Podemos usar el método mean para obtener la media de los valores devuev
print("Valor medio de clasificacion", nScores.mean())
