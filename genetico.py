import numpy as np
import random
import pygame
import sys
import math
from datetime import datetime
import matplotlib.pyplot as plt
from cargar_img import cargar_imagenes


# Parámetros de la Simulación
grid_width = 20  # Ancho de la grilla
grid_height = 10  # Alto de la grilla
cell_size = 60
max_moves = 100

# Parámetros del Algoritmo Genético
num_genes = 8  # Número de genes, uno por cada dirección de movimiento
mutation_rate = 0.1
n_individuals = grid_height + 5  # Número de individuos
n_padres = 2  # Número de individuos seleccionados para reproducción
n_generations = 1000  # Número de generaciones

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
screen = pygame.display.set_mode((screen_width + 200, screen_height))
pygame.display.set_caption("Algoritmo Genético")


# Cargar imágenes
creeper, steve = cargar_imagenes()

# redimensionar imagenes a tamaño de celda
creeper = pygame.transform.scale(creeper, (cell_size // 2, cell_size // 2))
steve = pygame.transform.scale(steve, (cell_size // 2, cell_size // 2))


# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GREEN = (0, 100, 0)

# Movimientos posibles: NO, N, NE, O, E, SO, S, SE
moves = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
]

# Colores asociados a los movimientos: NO, N, NE, O, E, SO, S, SE
move_colors = [
    ORANGE, DARK_GREEN, CYAN, YELLOW, MAGENTA, GREEN, BLUE, RED
]

# Nombres de los movimientos
move_names = [
    "NO", "N", "NE", "O", "E", "SO", "S", "SE"
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
        self.color = BLUE if not self.special_attribute else RED  # Color inicial
        self.aura_color = BLUE
        self.image = steve

    def draw(self, screen):
        # Dibujar el "aura"
        aura_radius = cell_size // 2
        center_x = self.y * cell_size + cell_size // 2
        center_y = self.x * cell_size + cell_size // 2
        pygame.draw.circle(screen, self.aura_color, (center_x, center_y), aura_radius, 5)

        # Dibujar la imagen del individuo
        if self.special_attribute:
            self.image = creeper
            screen.blit(self.image, (self.y * cell_size + cell_size // 4, self.x * cell_size + cell_size // 4))
        else:
            self.image = steve
            screen.blit(self.image, (self.y * cell_size + cell_size // 4, self.x * cell_size + cell_size // 4))
            
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
                    self.color = move_colors[move]  # Cambiar el color en función del movimiento
                    self.aura_color = move_colors[move]  # Cambiar el color del aura en función del movimiento
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
        self.history = []
        self.historial_vivos = []
        self.historial_asesinados = []
        self.historial_no_llegaron_meta = []
        self.historial_muertes = []

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
            selected_indices = np.random.choice(len(valid_individuals), size=min(self.n_padres, len(valid_individuals)), p=probabilidades, replace=False)
            selected = [valid_individuals[i] for i in selected_indices]
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
                    draw_legend(screen, generation)
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
            # Calcular y almacenar proporcion de vivos, asesinados, que no llegaron a la meta y muertos de la generacion
            prop_vivos = len([ind for ind in population if ind.vivo and ind.reached_goal]) / n_individuals
            prop_asesinados = (n_individuals - len([ind for ind in population if ind.vivo])) / n_individuals
            prop_no_llegaron_meta = len([ind for ind in population if not ind.reached_goal]) / n_individuals
            prop_muertes = prop_asesinados + prop_no_llegaron_meta
            self.historial_vivos.append(prop_vivos)
            self.historial_asesinados.append(prop_asesinados)
            self.historial_no_llegaron_meta.append(prop_no_llegaron_meta)
            self.historial_muertes.append(prop_muertes)

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

    screen.fill(BLACK)

    for x in range(0, screen_width, cell_size):
        for y in range(0, screen_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            if x // cell_size == grid_width - 1:  # Última columna
                pygame.draw.rect(screen, GREEN, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Líneas de la grilla

def draw_underline(screen, pos, width):
    # Función para dibujar subrayado
    pygame.draw.line(screen, WHITE, pos, (pos[0] + width, pos[1]), 2)


def draw_legend(screen, generation):
    font = pygame.font.SysFont('Arial', 20)
    legend_x = screen_width + 20  # Ajuste horizontal para alinear a la derecha
    legend_y = 20  # Posición vertical inicial de la leyenda

    for move_name, color in zip(move_names, move_colors):
        text = font.render(f'{move_name}:', True, WHITE)
        screen.blit(text, (legend_x, legend_y))
        pygame.draw.circle(screen, color, (legend_x + 100, legend_y + 10), 10)  # Ajuste horizontal para el círculo
        legend_y += 30  # Espacio entre líneas de la leyenda

        font = pygame.font.SysFont('Arial', 20)

    # Renderizar y dibujar el texto con subrayado para "Generación actual"
    generation_text = font.render('Generación actual:', True, WHITE)
    screen.blit(generation_text, (legend_x, legend_y))
    draw_underline(screen, (legend_x, legend_y + generation_text.get_height()), generation_text.get_width())

    # Renderizar y dibujar el número de generación
    generation_number = font.render(f'{generation}', True, WHITE)
    screen.blit(generation_number, (legend_x, legend_y + generation_text.get_height() + 10))  # Ajuste vertical

    # Renderizar y dibujar el texto con subrayado para "Semilla"
    seed_text = font.render(f'Semilla:', True, WHITE)
    screen.blit(seed_text, (legend_x, legend_y + generation_text.get_height() + 40))  # Ajuste vertical
    draw_underline(screen, (legend_x, legend_y + generation_text.get_height() + seed_text.get_height() + 40), seed_text.get_width())

    # Renderizar y dibujar el número de semilla
    seed_number = font.render(f'{seed}', True, WHITE)
    screen.blit(seed_number, (legend_x, legend_y + generation_text.get_height() + seed_text.get_height() + 50))  # Ajuste vertical

    # Agregar imagen de Steve y Creeper, indicando el atributo especial, donde creeper es el asesino
    steve_text = font.render('Individuo:', True, WHITE)
    screen.blit(steve_text, (legend_x, legend_y + generation_text.get_height() + seed_text.get_height() + 100))  # Ajuste vertical
    screen.blit(steve, (legend_x + 100, legend_y + generation_text.get_height() + seed_text.get_height() + 80))  # Ajuste vertical

    creeper_text = font.render('Asesino:', True, WHITE)
    screen.blit(creeper_text, (legend_x, legend_y + generation_text.get_height() + seed_text.get_height() + 140))  # Ajuste vertical
    screen.blit(creeper, (legend_x + 100, legend_y + generation_text.get_height() + seed_text.get_height() + 130))  # Ajuste vertical

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
# Función para agrupar valores obtiene la media de esa division
def agrupar_valores(valores, num_divisiones):
    agrupados = []
    tam_grupo = len(valores) // num_divisiones
    for i in range(num_divisiones):
        grupo = valores[i*tam_grupo:(i+1)*tam_grupo]
        agrupados.append(np.mean(grupo))
    return agrupados


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
    generaciones = np.linspace(0, n_generations, 100)
    model.run_geneticalgo()
    # Agrupar los valores en 100 divisiones
    num_divisiones = 100
    vivos_agrupados = agrupar_valores(model.historial_vivos, num_divisiones)
    muertos_agrupados = agrupar_valores(model.historial_muertes, num_divisiones)
    asesinatos_agrupados = agrupar_valores(model.historial_asesinados, num_divisiones)
    meta_agrupados = agrupar_valores(model.historial_no_llegaron_meta, num_divisiones)
    # Graficar la proporcion de vivos por generacion
    plt.figure(figsize=(12, 8))
    plt.plot(generaciones, vivos_agrupados, 'r', label='Proporcion vivos')
    plt.plot(generaciones, muertos_agrupados, 'y', label='Proporcion muertos')
    plt.plot(generaciones,asesinatos_agrupados, 'b', label='Proporcion asesinatos')
    plt.plot(generaciones, meta_agrupados, 'g', label='Proporcion llegaron a la meta')
    plt.legend()
    plt.show()
# Crear figura y ejes
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Plotear cada variable en un subgráfico diferente
    axes[0].plot(model.historial_vivos, 'r', label='Proporción vivos')
    axes[0].legend()
    axes[0].set_ylabel('Proporción')

    axes[1].plot(model.historial_muertes, 'y', label='Proporción muerte')
    axes[1].legend()
    axes[1].set_ylabel('Proporción')

    axes[2].plot(model.historial_asesinados, 'b', label='Proporción Asesinatos')
    axes[2].legend()
    axes[2].set_ylabel('Proporción')

    axes[3].plot(model.historial_no_llegaron_meta, 'g', label='Proporción que no llegaron a la meta')
    axes[3].legend()
    axes[3].set_xlabel('Generación')
    axes[3].set_ylabel('Proporción')

    # Título de la figura
    fig.suptitle('Evolución del algoritmo genético')

    # Ajustar el diseño para evitar superposición
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Mostrar los gráficos
    plt.show()

if __name__ == "__main__":
    main()