import numpy as np
import random
import pygame
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# Parámetros de la Simulación
grid_width = 20  # Ancho de la grilla
grid_height = 10  # Alto de la grilla
cell_size = 60
max_moves = 100

# Parámetros del Algoritmo Genético
num_genes = 8  # Número de genes, uno por cada dirección de movimiento
mutation_rate = 0.05
n_individuals = grid_height + 5  # Número de individuos
n_padres = 2  # Número de individuos seleccionados para reproducción
n_generations = 2000  # Número de generaciones

# Parámetro opcional para la semilla
use_seed = False  # Cambia a True si deseas fijar una semilla
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
        self.vivo = True
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
        other_individual.vivo = False
        #print(f"Individuo {other_individual.id} eliminado por el individuo {self.id}")
        
        
    def reset(self):
        """Reiniciar la posición y estado del individuo."""
        self.x = np.random.randint(0, grid_height)
        self.y = np.random.randint(0, 2)
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
    def __init__(self, num_genes, mutation_rate, n_individuals, n_padres, n_generations, fitness_func, verbose=True):
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.n_padres = n_padres
        self.n_generations = n_generations
        self.verbose = verbose
        self.fitness_func = fitness_func
        self.num_genes = num_genes
        self.history = []  # Variable para almacenar los genes de cada generación
        self.fitness_history = []  # Variable para almacenar el fitness de cada generación

    def create_individual(self):
        """Crea un individuo con probabilidades de movimiento aleatorias y atributo especial en False."""
        x, y = np.random.randint(0, grid_height), np.random.randint(0, 2)
        return Individual(x, y, self.num_genes)
    
    def create_population(self):
        """Crea una población inicial de individuos."""
        return [self.create_individual() for _ in range(self.n_individuals)]

    def selection(self, population):
        # Filtrar individuos que están vivos y han alcanzado la meta
        valid_individuals = [ind for ind in population if ind.vivo and ind.reached_goal]
        
        if not valid_individuals:
            print("No hay individuos válidos para seleccionar")
            return []  # No hay individuos válidos para seleccionar

        if len(valid_individuals) < self.n_padres:
            print("No hay suficientes individuos válidos para seleccionar el número deseado de padres")
            return valid_individuals  # Retornar todos los individuos válidos disponibles

        fitness_values = [self.fitness_func(ind) for ind in valid_individuals]
        
        if max(fitness_values) == 0:
            print("Todos los individuos tienen fitness 0, seleccionando aleatoriamente")
            probabilidades = [1 / len(valid_individuals) for _ in valid_individuals]
        else:
        #    print("Numero de individuos validos", len(valid_individuals))
            # Normalizar los valores de fitness para obtener probabilidades
            #sortear los fitness
        #    fitness_values.sort(reverse=True)
        #    total_fitness = sum(fitness_values)
        #    probabilities = [fitness / total_fitness for fitness in fitness_values]

        # Asegurar que las probabilidades sumen 1
        #total_prob = sum(probabilities)
        #if total_prob != 1:
        #    probabilities = [p / total_prob for p in probabilities]
        
        #selected_indices = np.random.choice(len(valid_individuals), size=min(self.n_padres, len(valid_individuals)), p=probabilities, replace=False)
        #selected = [valid_individuals[i] for i in selected_indices]
        
        #for i in range(len(selected)):
        #    print(f'Individuo {selected[i].id} seleccionado con probabilidad {probabilities[i]}')
####
            #print("Numero de individuos validos", len(valid_individuals))
            fitness_values.sort(reverse=True)
            #Ordenar individuos por su fitness deben ir ordenados del mas apto al menos apto
            valid_individuals.sort(key=lambda x: self.fitness_func(x), reverse=True)
            #Obtener probabilidades (1-p)**n para n individuos validos
            if max(fitness_values) >= 1:
                factor_p = 0.9
            else:
                factor_p = max(fitness_values)
            probabilidades = []
            for i in  range(len(valid_individuals)):
                probabilidades.append((1-factor_p)**i+1) #empezar en n =1
            normalizar(probabilidades)
            #print("Normalizada",sum(probabilidades))
            selected_indices = np.random.choice(len(valid_individuals), size=min(self.n_padres, len(valid_individuals)), p=probabilidades, replace=False)
            selected = [valid_individuals[i] for i in selected_indices]
            #Imprimir los individuos seleccionados y los indices
            for i in range(len(selected)):
                print(f'Individuo {selected[i].id} seleccionado con probabilidad {probabilidades[selected_indices[i]]}')
            print(f"factor p de reproduccion de este episodio: {factor_p}")
            return selected

    def reproduction(self, population, selected):
        """Realiza la reproducción entre individuos seleccionados."""
        new_population = []
        if not selected:
            print("No hay individuos seleccionados para reproducción, creando nueva población")
            return self.create_population()  # Crear nueva población si no hay individuos seleccionados

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
                #for ind in population:
                #    print(f'ID: {ind.id} | ¿Asesino?: {'Si' if ind.special_attribute else 'No'}')
            # Limpiar la grilla y reiniciar individuos para la nueva generación
            for individual in population:
                individual.reset()

            # Simulación de la generación actual
            move_iterations = 0  # Contador de iteraciones de movimiento
            occupied_positions = set((ind.x, ind.y) for ind in population)  # Conjunto de posiciones ocupadas
            reached_goal_individuals = []  # Individuos que alcanzaron la meta en esta generación

            while move_iterations < max_moves:
                if generation % 100 == 0:
                    # Actualizar la pantalla solo en múltiplos de 100
                    draw_grid(screen)
                    draw_population(screen, population)
                    pygame.time.wait(50)
                    pygame.display.update()

                # Mover la población
                for individual in population:
                    individual.move(occupied_positions, population)
                    if individual.reached_goal and individual not in reached_goal_individuals:
                        reached_goal_individuals.append(individual)
                move_iterations += 1  # Incrementar el contador de iteraciones de movimiento
                #mostrar solo generacion 100 y 200
                pygame.time.wait(0)

            # Calcular y almacenar el fitness promedio de la generación solo para los individuos que alcanzaron la meta
            fitness_promedio = max([self.fitness_func(ind) for ind in population if ind.reached_goal and ind.vivo], default=0)
            self.fitness_history.append(fitness_promedio)
            vivos_reached_goal = [ind for ind in reached_goal_individuals if ind.vivo]

            # Guardar los genes de la población actual en la historia
            self.history.append([ind.genes for ind in population])
            if  len(vivos_reached_goal) == 0:
                selected = []
            else:
                selected = self.selection(population)
            population = self.reproduction(population, selected)
            population = self.mutation(population)
            
            # Imprimir los individuos vivos que llegaron a la meta en esta generación
            if vivos_reached_goal:
                print(f"Individuos vivos que alcanzaron la meta en la generación {generation}: {len(vivos_reached_goal)} individuos")
                for ind in vivos_reached_goal:
                    print(f"ID: {ind.id} | Pasos para llegar a la meta: {ind.steps_to_goal} |Fitness:{self.fitness_func(ind)} | ¿Asesino?: {'Si' if ind.special_attribute else 'No'}")
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
    if individual.reached_goal and individual.vivo:
        fitness = (grid_width) / (individual.steps_to_goal)
        if fitness >=1:
            fitness = 1
        return fitness  # Recompensar llegar a la meta en menos movimientos
    else:
        return 0  # Penalizar la distancia desde la última columna

def main():
    model = DNA(
        num_genes=num_genes,
        mutation_rate=mutation_rate,
        n_individuals=n_individuals,
        n_padres=n_padres,
        n_generations=n_generations,
        fitness_func=fitness,
        verbose=True
    )

    population = model.run_geneticalgo()
    # Graficar el fitness promedio por generación
    plt.plot(model.fitness_history)
    plt.xlabel('Generación')
    plt.ylabel('Fitness Promedio')
    plt.title('Evolución del Fitness Promedio')
    plt.show()
    #print("Población final:")
    #for individual in population:
    #    print(f"ID: {individual.id} | ¿Asesino?: {'Si' if individual.special_attribute else 'No'}")

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
