import os
import shutil
import cv2


j = 676940
danadas = []
# Define a function to check if an image contains a silhouette
def contains_silhouette(image_path, input_directory):
    k = 0
    while True:
        try:
            # Read the image
            image = cv2.imread(image_path)
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply a threshold to create a binary image
            _, binary_image = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            # Find contours in the binary image
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Check if any contours were found
            return len(contours) > 0
        except:
            cambiar_imagen_danada(image_path, input_directory)
            k = k + 1
            if k == 2:
                danadas.append(image_path)
                return True

def input_image(input_directory):
    # Iterate through the files in the input directory
    i = 0
    for filename in os.listdir(input_directory):
        if filename.endswith('.png'):
            input_path = os.path.join(input_directory, filename)
            # Check if the image contains a silhouette
            if not contains_silhouette(input_path, input_directory):
                os.remove(input_path)
                i = i + 1
                global j
                j = j + 1
                print("Removed: ", input_path, "No. ", str(i), "Totales: ", str(j))
    print(input_directory, ": Finished")


def buscar_y_eliminar_directorios(ruta_principal, palabras_clave):
    for ruta_actual, directorios, archivos in os.walk(ruta_principal):
        for directorio in directorios:
            for palabra_clave in palabras_clave:
                if palabra_clave in directorio:
                    ruta_completa = os.path.join(ruta_actual, directorio)
                    input_image(ruta_completa)

def cambiar_imagen_danada(path_imagen_danada, input_directory):
    nuevo_path = path_imagen_danada.replace("TecNM-GaitDS-Clean", "TecNM-GaitDS")
    os.remove(path_imagen_danada)
    shutil.copy(nuevo_path, input_directory)
    print("Imagen cambiada. ", path_imagen_danada)

for i in range(111, 122):
    ruta_principal = r'C:\\Users\\Maquinot\\Desktop\\Misael_Zazueta\\TecNM-GaitDS\\'
    '''
    if i > 99:
        ruta_principal = ruta_principal + str(i)
    if i < 100 and i > 9:
        ruta_principal = ruta_principal + '0' + str(i)
    if i > 0 and i < 10:
        ruta_principal = ruta_principal + '00' + str(i)
    '''
    palabras_clave = ["silhouettes"]
    buscar_y_eliminar_directorios(ruta_principal, palabras_clave)

print(danadas)
