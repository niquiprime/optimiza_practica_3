import pygame
import os

def cargar_imagenes():
    # Obtén la ruta de las imágenes (asegúrate de tener las imágenes en el mismo directorio que este script)
    ruta_imagen_jpg = os.path.join("./sprites/creeper.jpeg")
    ruta_imagen_png = os.path.join("./sprites/steve.png")
    
    # Cargar imágenes
    creeper = pygame.image.load(ruta_imagen_jpg)
    steve = pygame.image.load(ruta_imagen_png)
    
    # Devolver las imágenes cargadas
    return creeper, steve