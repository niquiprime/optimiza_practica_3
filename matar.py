import numpy as np
import random
import pygame
import sys
from datetime import datetime

# Parámetros del Algoritmo Genético
num_genes = 8  # Número de genes, uno por cada dirección de movimiento
mutation_rate = 0.5
n_individuals = 10  # Número de individuos
n_selection = 2  # Número de individuos seleccionados para reproducción
n_generations = 20  # Número de generaciones

# Parámetros de la Simulación
grid_width = 10  # Ancho de la grilla
grid_height = 8  # Alto de la grilla
cell_size = 60
max_moves = 100

# Parámetro opcional para la semilla
use_seed = True  # Cambia a True si deseas fijar una semilla
seed = 1718226075 if use_seed else int(datetime.now().timestamp())

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
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Movimientos posibles: NO, N, NE, O, E, SO, S, SE
moves = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
]

class Individual:
    def __init__(self, x, y, num_genes):
        self.id = id(self)
        self.x = x
        self.y = y
        self.genes = [random.random() for _ in range(num_genes)]
        self.genes = normalizar(self.genes)
        self.special_attribute = False
        self.move_count = 0  # Contador de movimientos
        self.reached_goal = False  # Indicador de si alcanzó la meta
        self.steps_to_goal = None  # Pasos para llegar a la meta

    def draw(self, screen):
        color = RED if self.special_attribute else BLUE
        pygame.draw.circle(screen, color, (self.y * cell_size + cell_size // 2, self.x * cell_size + cell_size // 2), cell_size // 3)
        
    def move(self, occupied_positions, population):
        if not self.reached_goal and self.move_count < max_moves and self.y < grid_width - 1:  # Comprobar si el individuo puede moverse
            move = np.random.choice(range(num_genes), p=self.genes)
            dx, dy = moves[move]
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < grid_height and 0 <= new_y < grid_width:
                # Verificar si hay otro individuo en la nueva posición
                for other_individual in population:
                    if other_individual.x == new_x and other_individual.y == new_y:
                        # Verificar si el individuo actual tiene el atributo especial (asesino)
                        if self.special_attribute:
                            self.eliminate(other_individual, occupied_positions, population)
                        # No importa si se puede mover o no si ya hay otro individuo en esa posición
                        return

                if (new_x, new_y) not in occupied_positions:
                    if (self.x, self.y) in occupied_positions:
                        occupied_positions.remove((self.x, self.y))  # Liberar la posición actual
                    self.x, self.y = new_x, new_y
                    occupied_positions.add((self.x, self.y))  # Marcar la nueva posición como ocupada
                    self.move_count += 1  # Incrementar el contador de movimientos
                else:
                    self.move_count += 1  # Si no se puede mover, incrementar el contador de movimientos sin cambiar de posición

                if self.y == grid_width - 1:  # Alcanzó la última columna y esta vivo
                    self.reached_goal = True
                    self.steps_to_goal = self.move_count

    def eliminate(self, other_individual, occupied_positions, population):
        """Elimina al otro individuo de la población y de la grilla."""
        if (other_individual.x, other_individual.y) in occupied_positions:
            occupied_positions.remove((other_individual.x, other_individual.y))
        population.remove(other_individual)
        other_individual.reached_goal = False
        print(f"Individuo {other_individual.id} eliminado por el individuo {self.id}")
        
        
    def reset(self):
        """Reiniciar la posición y estado del individuo."""
        self.x = random.randint(0, grid_height - 1)
        self.y = random.randint(0, 1)
        self.move_count = 0
        self.reached_goal = False
        self.steps_to_goal = None
    


def normalizar(array):
    sumatoria = sum(array)
    for i in range(len(array)):
        array[i] = array[i] / sumatoria
    # Si la suma de los valores normalizados no es 1, se le asigna la diferencia al de mayor peso para no afectar significativamente la probabilidad
    max_value = max(array)
    diff = 1 - sum(array)
    array[array.index(max_value)] += diff
    return array

class DNA:
    def __init__(self, num_genes, mutation_rate, n_individuals, n_selection, n_generations, fitness_func, verbose=True):
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.n_selection = n_selection
        self.n_generations = n_generations
        self.verbose = verbose
        self.fitness_func = fitness_func
        self.num_genes = num_genes
        self.history = []  # Variable para almacenar los genes de cada generación

    def create_individual(self):
        """Crea un individuo con probabilidades de movimiento aleatorias y atributo especial en False."""
        x, y = random.randint(0, grid_height - 1), random.randint(0, 1)
        return Individual(x, y, self.num_genes)
    
    def create_population(self):
        """Crea una población inicial de individuos."""
        return [self.create_individual() for _ in range(self.n_individuals)]

    def selection(self, population):
        """Selecciona los mejores individuos en función de su aptitud."""
        scores = [(self.fitness_func(individual), individual) for individual in population]
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [individual for _, individual in scores[:self.n_selection]]
        return selected

    def reproduction(self, population, selected):
        """Realiza la reproducción entre individuos seleccionados."""
        new_population = []
        for _ in range(self.n_individuals):
            if len(selected) > 1:
                parent1, parent2 = random.sample(selected, 2)
                crossover_point = np.random.randint(1, self.num_genes)
                child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
                child = Individual(parent1.x, parent1.y, self.num_genes)
                child.genes = normalizar(child_genes)
                new_population.append(child)
            elif len(selected) == 1:
                parent = selected[0]
                child = Individual(parent.x, parent.y, self.num_genes)
                child.genes = normalizar(parent.genes)
                new_population.append(child)
            else:
                new_population = self.create_population()
                break
        return new_population

    def mutation(self, population):
        """Realiza la mutación en la población según la tasa de mutación."""
        for individual in population:
            if random.random() < self.mutation_rate:
                mutation_point = np.random.randint(self.num_genes)
                max_value = max(individual.genes)
                individual.genes[mutation_point] = max_value / 2
                individual.special_attribute = True
                individual.genes = normalizar(individual.genes)
        return population

    def run_geneticalgo(self):
        """Ejecuta el algoritmo genético durante el número de generaciones especificado."""
        population = self.create_population()

        for generation in range(self.n_generations + 1):
            if self.verbose:
                print(f'Generación {generation}:  {len(population)} individuos')
                for ind in population:
                    print(f'Asesino: {ind.special_attribute} | ID: {ind.id}')
            # Limpiar la grilla y reiniciar individuos para la nueva generación
            for individual in population:
                individual.reset()

            # Simulación de la generación actual
            move_iterations = 0  # Contador de iteraciones de movimiento
            occupied_positions = set((ind.x, ind.y) for ind in population)  # Conjunto de posiciones ocupadas
            reached_goal_individuals = []  # Individuos que alcanzaron la meta en esta generación

            while move_iterations < max_moves:
                screen.fill(WHITE)
                draw_grid(screen)
                draw_population(screen, population)
                pygame.display.flip()

                # Mover la población
                for individual in population:
                    individual.move(occupied_positions, population)
                    if individual.reached_goal and individual not in reached_goal_individuals:
                        reached_goal_individuals.append(individual)
                move_iterations += 1  # Incrementar el contador de iteraciones de movimiento

                pygame.time.wait(100)

            # Guardar los genes de la población actual en la historia
            self.history.append([ind.genes for ind in population])
            selected = self.selection(population)
            population = self.reproduction(population, selected)
            population = self.mutation(population)
            
            
             # Imprimir los individuos que llegaron a la meta en esta generación
            if reached_goal_individuals:
                print(f"Individuos que alcanzaron la meta en la generación {generation}: {len(reached_goal_individuals)} individuos")
                for ind in reached_goal_individuals:
                    print(f"ID: {ind.id} | Pasos para llegar a la meta: {ind.steps_to_goal} | ¿Asesino?: {'Si' if ind.special_attribute else 'No'}")
            else:
                print(f"Ningún individuo alcanzó la meta en la generación {generation}")

            

        return population

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

def fitness(individual):
    if individual.reached_goal:
        return (grid_width) / (individual.steps_to_goal + 1)  # Recompensar llegar a la meta en menos movimientos
    else:
        return 0  # Penalizar la distancia desde la última columna

def main():
    model = DNA(
        num_genes=num_genes,
        mutation_rate=mutation_rate,
        n_individuals=n_individuals,
        n_selection=n_selection,
        n_generations=n_generations,
        fitness_func=fitness,
        verbose=True
    )

    population = model.run_geneticalgo()

    print("Población final:")
    for individual in population:
        print(f"ID: {individual.id} | ¿Asesino?: {'Si' if individual.special_attribute else 'No'}")

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
