import pandas as pd
import os

# Obtener el directorio actual de trabajo
current_directory = os.getcwd()

# Proporcionar la ruta completa al archivo 'dataset.csv'
csv_file_path = os.path.join(current_directory, 'dataset.csv')

# Verificar si el archivo existe en la ruta proporcionada
if os.path.exists(csv_file_path):
    # Cargar el conjunto de datos desde el archivo CSV en un DataFrame de pandas
    movies = pd.read_csv(csv_file_path)

    # Mostrar las primeras 10 filas del DataFrame
    print(movies.head(10))

    # Generar estadísticas descriptivas del DataFrame
    print(movies.describe())

    # Imprimir un resumen conciso del DataFrame (información sobre columnas, tipos de datos, valores no nulos, etc.)
    print(movies.info())

    # Calcular la suma de valores nulos para cada columna en el DataFrame
    print(movies.isnull().sum())

    # Obtener los nombres de las columnas del DataFrame
    print(movies.columns)

    # Seleccionar y mantener solo las columnas especificadas ('id', 'title', 'overview', 'genre') del DataFrame
    movies = movies[['id', 'title', 'overview', 'genre']]

    # Agregar una nueva columna 'tags' concatenando las columnas 'overview' y 'genre'
    movies['tags'] = movies['overview'] + ' ' + movies['genre']

    # Crear un nuevo DataFrame 'new_data' eliminando las columnas 'overview' y 'genre' de 'movies'
    new_data = movies.drop(columns=['overview', 'genre'])

    # Importar los módulos necesarios de la biblioteca NLTK para el procesamiento de texto
    import nltk
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Descargar recursos de NLTK para tokenización, lematización y stopwords
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')


    # Definir una función para limpiar datos de texto
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)


    # Aplicar la función clean_text a la columna 'tags' de 'new_data' y almacenar el resultado en 'tags_clean'
    new_data['tags_clean'] = new_data['tags'].apply(clean_text)

    # Asegurarse de que scikit-learn esté instalado
    try:
        import sklearn
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        os.system('pip install scikit-learn')
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

    # Inicializar un objeto CountVectorizer con un máximo de 10,000 características y stop words en inglés
    cv = CountVectorizer(max_features=10000, stop_words='english')

    # Ajustar el CountVectorizer a la columna 'tags_clean' y transformar los datos de texto en una representación vectorial numérica
    vector = cv.fit_transform(new_data['tags_clean'].values.astype('U')).toarray()

    # Comprobar la forma del vector resultante
    print(vector.shape)

    # Calcular la similitud del coseno entre los vectores
    similarity = cosine_similarity(vector)

    # Imprimir un resumen conciso del DataFrame 'new_data'
    print(new_data.info())

    # Calcular las puntuaciones de similitud para la tercera película con todas las demás películas, ordenarlas y almacenar el resultado
    distance = sorted(list(enumerate(similarity[2])), reverse=True, key=lambda vector: vector[1])

    # Imprimir los títulos de las cinco películas más similares a la tercera película
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)


    # Definir una función para recomendar las 5 películas más similares para un título de película dado
    def recommend(movies):
        index = new_data[new_data['title'] == movies].index[0]
        distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
        for i in distance[0:5]:
            print(new_data.iloc[i[0]].title)


    # Llamar a la función recommend con "Iron Man" como argumento
    recommend("Iron Man")

    # Importar el módulo pickle para la serialización de objetos en Python
    import pickle

    # Serializar el DataFrame 'new_data' y guardarlo en un archivo
    pickle.dump(new_data, open('movies_list.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

    # Deserializar el archivo 'movies_list.pkl' de nuevo en un objeto Python
    loaded_data = pickle.load(open('movies_list.pkl', 'rb'))

    # Importar el módulo os para interactuar con el sistema operativo
    import os

    # Imprimir el directorio de trabajo actual
    print(os.getcwd())
else:
    print(f"El archivo {csv_file_path} no existe.")
