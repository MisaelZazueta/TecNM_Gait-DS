import os
import cv2
import numpy as np
from sklearn.cluster import SpectralClustering


def gei_Generator(ruta_entrada, directorio_salida, nombre):
    # Directorio donde se encuentran las siluetas
    directorio_siluetas = ruta_entrada

    # Leer las siluetas del directorio y extraer la región de la silueta
    siluetas_region = []
    for archivo in os.listdir(directorio_siluetas):
        ruta_archivo = os.path.join(directorio_siluetas, archivo)
        silueta = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)

        # Encontrar los contornos de la silueta
        _, binaria = cv2.threshold(silueta, 0, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contornos) > 0:
            # Obtener el contorno más grande (asumiendo que es la silueta principal)
            contorno_silueta = max(contornos, key=cv2.contourArea)

            # Obtener el rectángulo delimitador de la silueta
            x, y, w, h = cv2.boundingRect(contorno_silueta)

            # Extraer la región de la silueta y redimensionar a una forma común
            silueta_region = silueta[y:y + h, x:x + w]
            silueta_region_resized = cv2.resize(silueta_region, (300, 600))  # Ajustar el tamaño según sea necesario
            siluetas_region.append(silueta_region_resized)

    # Convertir las siluetas a un formato compatible con Spectral Clustering
    siluetas_flattened = [silueta.flatten() for silueta in siluetas_region]

    # Realizar Spectral Clustering para agrupar las siluetas
    num_clusters = 3  # Número de conjuntos deseados
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=0)
    etiquetas = spectral_clustering.fit_predict(siluetas_flattened)

    # Crear una imagen de energía vacía
    imagen_energia = np.zeros_like(siluetas_region[0], dtype=np.float32)

    # Calcular la energía para cada silueta
    for i, silueta in enumerate(siluetas_region):
        # Convertir la silueta en una máscara binaria
        mascara = cv2.threshold(silueta, 0, 255, cv2.THRESH_BINARY)[1]

        # Calcular la energía de la silueta (puedes ajustar esta parte según tus necesidades)
        energia_silueta = cv2.distanceTransform(mascara, cv2.DIST_L2, 3)

        # Aplicar la transformación logarítmica a la energía de la silueta
        energia_silueta = np.log(energia_silueta + 1)

        # Agregar la energía de la silueta a la imagen de energía correspondiente al conjunto
        imagen_energia += energia_silueta

    # Escalar los valores de la imagen de energía para el rango 0-255
    imagen_energia = cv2.normalize(imagen_energia, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Guardar la imagen de energía en el directorio de salida
    nombre_archivo = nombre
    ruta_salida = os.path.join(directorio_salida, nombre_archivo)
    cv2.imwrite(ruta_salida, imagen_energia)

    print('Se ha creado la imagen de energía para todas las siluetas seleccionadas.')
    print('Proceso finalizado.')