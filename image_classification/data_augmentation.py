import os
from keras.preprocessing.image import ImageDataGenerator
import shutil
import random

from keras.utils import img_to_array, load_img

# Rutas de directorios de entrada, salida de entrenamiento y salida de prueba
directorio_entrada = '/home/mcc/PycharmProjects/Transformer-Silhouettes_v1/image_classification/Dataset_124_subjects_angles/'
directorio_entrenamiento = '/home/mcc/PycharmProjects/Transformer-Silhouettes_v1/image_classification/Dataset_Augmented_124_angles/train/'
directorio_prueba = '/home/mcc/PycharmProjects/Transformer-Silhouettes_v1/image_classification/Dataset_Augmented_124_angles/test/'

# Porcentaje de datos para entrenamiento y prueba (80/20)
porcentaje_entrenamiento = 0.8

# Crear directorios de entrenamiento y prueba si no existen
if not os.path.exists(directorio_entrenamiento):
    os.makedirs(directorio_entrenamiento)
if not os.path.exists(directorio_prueba):
    os.makedirs(directorio_prueba)

# Obtener la lista de clases (nombres de las carpetas en el directorio de entrada)
clases = os.listdir(directorio_entrada)

# Iterar sobre las clases
for clase in clases:
    # Ruta completa de la clase de entrada y salida para entrenamiento y prueba
    ruta_clase_entrada = os.path.join(directorio_entrada, clase)
    ruta_clase_entrenamiento = os.path.join(directorio_entrenamiento, clase)
    ruta_clase_prueba = os.path.join(directorio_prueba, clase)

    # Crear directorios de salida para entrenamiento y prueba si no existen
    if not os.path.exists(ruta_clase_entrenamiento):
        os.makedirs(ruta_clase_entrenamiento)
    if not os.path.exists(ruta_clase_prueba):
        os.makedirs(ruta_clase_prueba)

    # Obtener la lista de imágenes en la clase de entrada
    imagenes = os.listdir(ruta_clase_entrada)

    # Calcular el número de imágenes para entrenamiento y prueba
    num_entrenamiento = int(len(imagenes) * porcentaje_entrenamiento)
    num_prueba = len(imagenes) - num_entrenamiento

    # Barajar las imágenes aleatoriamente
    random.shuffle(imagenes)

    # Iterar sobre las imágenes para entrenamiento
    for imagen in imagenes[:num_entrenamiento]:
        # Ruta completa de la imagen de entrada y salida para entrenamiento
        ruta_imagen_entrada = os.path.join(ruta_clase_entrada, imagen)
        ruta_imagen_salida = os.path.join(ruta_clase_entrenamiento, imagen)

        # Copiar la imagen original al directorio de entrenamiento
        shutil.copy(ruta_imagen_entrada, ruta_imagen_salida)

        # Configurar el generador de aumentos de datos para entrenamiento
        datagen_entrenamiento = ImageDataGenerator(
            rescale=1./255,
            #rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

        # Cargar la imagen
        imagen_cargada = load_img(ruta_imagen_entrada)

        # Convertir la imagen en un arreglo numpy
        x = img_to_array(imagen_cargada)
        x = x.reshape((1,) + x.shape)

        # Generar aumentos de datos y guardar las imágenes aumentadas para entrenamiento
        i = 0
        for batch in datagen_entrenamiento.flow(x, batch_size=1, save_to_dir=ruta_clase_entrenamiento, save_prefix='augmented', save_format='png'):
            i += 1
            if i >= 1:  # Número de imágenes aumentadas a generar por imagen original
                break

    # Iterar sobre las imágenes para prueba
    for imagen in imagenes[num_entrenamiento:]:
        # Ruta completa de la imagen de entrada y salida para prueba
        ruta_imagen_entrada = os.path.join(ruta_clase_entrada, imagen)
        ruta_imagen_salida = os.path.join(ruta_clase_prueba, imagen)

        # Copiar la imagen original al directorio de prueba
        shutil.copy(ruta_imagen_entrada, ruta_imagen_salida)