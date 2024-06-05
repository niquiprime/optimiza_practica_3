import numpy as np
import random
import pygame
import sys
from datetime import datetime

# Parámetros del Algoritmo
grid_width = 10  # Ancho de la grilla
grid_height = 8  # Alto de la grilla
population_size = grid_height + 3  # Asegurar que la población inicial sea mayor que el largo de una columna
cell_size = 60
max_moves = 100

# Parámetro opcional para la semilla
use_seed = False  # Cambia a True si deseas fijar una semilla
seed = 42 if use_seed else int(datetime.now().timestamp())

# Fijar la semilla para reproducibilidad (opcional)
random.seed(seed)
np.random.seed(seed)

# Inicialización de Pygame
pygame.init()

# Tamaño de la pantalla
screen_width = grid_width * cell_size
screen_height = grid_height * cell_size
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Algoritmo Genético - Población Inicial")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Movimientos posibles N, NE, E, SE, S, SO, O, NO, Quedarse quieto
moves = [
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0)
]

class Individual:
    def __init__(self, x, y, prob_moves):
        self.x = x
        self.y = y
        self.prob_moves = prob_moves
        self.move_count = 0  # Contador de movimientos

    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, (self.y * cell_size + cell_size // 2, self.x * cell_size + cell_size // 2), cell_size // 3)

    def move(self):
        if self.move_count < max_moves and self.y < grid_width - 1:  # Comprobar si el individuo puede moverse
            move = np.random.choice(range(9), p=self.prob_moves)
            dx, dy = moves[move]
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < grid_height and 0 <= new_y < grid_width:
                self.x, self.y = new_x, new_y
                self.move_count += 1  # Incrementar el contador de movimientos

def initialize_population(size):
    population = []
    positions = set()
    while len(population) < size:
        x, y = random.randint(0, grid_height-1), random.randint(0, 1)
        if (x, y) not in positions:
            prob_moves = np.random.dirichlet(np.ones(9))
            population.append(Individual(x, y, prob_moves))
            positions.add((x, y))
    return population

population = initialize_population(population_size)

def draw_grid(screen):
    for x in range(0, screen_width, cell_size):
        for y in range(0, screen_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            if x // cell_size == grid_width - 1:  # Última columna
                pygame.draw.rect(screen, GREEN, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Líneas de la grilla

def draw_population(screen, population):
    for individual in population:
        individual.draw(screen)

# Bucle principal de Pygame
running = True
move_iterations = 0  # Contador de iteraciones de movimiento

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill(WHITE)
    draw_grid(screen)
    draw_population(screen, population)
    pygame.display.flip()

    if move_iterations < max_moves:
        # Mover la población
        for individual in population:
            individual.move()
        move_iterations += 1  # Incrementar el contador de iteraciones de movimiento
    
    #pygame.time.wait(1000)

pygame.quit()
sys.exit()
