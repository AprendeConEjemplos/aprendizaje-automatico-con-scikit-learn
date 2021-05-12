"""Cargamos la librería necesaria"""

import pandas as pd
import sklearn as skl

"""Cargamos el fichero del dataset"""

url = "https://raw.githubusercontent.com/AprendeConEjemplos/aprendizaje-automatico-con-scikit-learn/main/04_Preprocesamiento/Stars.csv"
dataframe = pd.read_csv(url)
print(dataframe)

print(dataframe.describe())

dataset = dataframe.drop("Type", axis=1)
label = dataframe["Type"].copy()

dataframe_op1 = dataframe.dropna(subset=["Temperature"])    # Opción 1, eliminamos las instancias con valores nulos
dataframe_op2 = dataframe.drop("Temperature", axis=1)       # Opción 2, eliminamos el atributo que contiene valores nulos
mean_temp = dataframe["Temperature"].mean()
dataframe_op3 = dataframe["Temperature"].fillna(mean_temp)  # Opción 3, asignamos el valor medio en los valores nulos

"""Iniciamos el preprocesamiento de los atributos con valores de texto"""

color_cat = dataframe[['Color']]
spectral_cat = dataframe[['Spectral_Class']]
print(color_cat.head(10))
print(spectral_cat.head(10))

"""Importamos la funcionalidad de Scikit-learn"""

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
color_cat_encoded = ordinal_encoder.fit_transform(color_cat)
print(color_cat_encoded[:10])

print(ordinal_encoder.categories_)

"""Importamos lo necesario para realizar el One Hot Encoding"""

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
color_cat_one_hot = one_hot_encoder.fit_transform(color_cat)
print(color_cat_one_hot)
print(color_cat_one_hot.toarray().shape)
print(color_cat_one_hot.toarray())

"""Ejemplos de normalización de valores de atributos

"""

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
l_values = dataframe[['L']]
scaled_values = min_max_scaler.fit(l_values)
print(min_max_scaler.transform(l_values)[0:10])

"""Ejemplo de estandarización"""

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
l_values = dataframe[['L']]
scaled_values = standard_scaler.fit(l_values)
print(standard_scaler.transform(l_values)[0:10])

"""Creación del pipeline completo"""

from sklearn.compose import ColumnTransformer
num_attrs = ["Temperature", "L", "R"]
text_attrs = ["Color", "Spectral_Class"]

pipeline = ColumnTransformer([
                              ("numeric", StandardScaler(), num_attrs),
                              ("text", OneHotEncoder(), text_attrs)
])
preprocessed_dataset = pipeline.fit_transform(dataset)

"""Creamos los conjutos de entrenamiento y test"""

from sklearn.model_selection import train_test_split
# Realizamos la partición de nuestro dataset en un conjunto de entrenamiento y otro de test (20%)
X_train, X_test, y_train, y_test = train_test_split(preprocessed_dataset, label, test_size=0.2, random_state=42)

"""Entrenamos y mostramos el resultado"""

from sklearn.svm import SVC

# Creamos el clasificador SVM lineal
classifier =  SVC()

# Realizamos el entrenamiento
classifier.fit(X_train, y_train)

# Obtenemos el accuracy de nuestro modelo para el conjunto de test
print(classifier.score(X_test, y_test))