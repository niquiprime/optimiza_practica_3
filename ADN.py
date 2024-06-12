
import numpy as np
import random

class DNA:
    def __init__(self, target, mutation_rate, n_individuals, n_selection, n_generations, verbose = True):
        self.target = target
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.n_selection = n_selection
        self.n_generations = n_generations
        self.verbose = verbose
        self.pasos = np.random.randint(1, 9)


    def crear_individuo(self, min = 20, max = 100):
        return normalizar([np.random.randint(min, max) for _ in range(len(self.target))])

    def create_population(self):
        return [self.crear_individuo() for _ in range(self.n_individuals)]

    def fitness(self, individual):
        pasos = self.pasos
        minimo_pasos = 0
        fitness = minimo_pasos - pasos
        
        
        return fitness
    
    def selection(self, population):

        scores = [(self.fitness(i), i) for i in population ]
        scores = [(self.fitness(i), i) for i in population]
        scores = [i[1] for i in sorted(scores)]
        #print("Numero random",scores)

        return scores[len(scores)-self.n_selection:]
    
    def reproduction(self, population, selected):

        point = 0
        father = []

        for i in range(len(population)):
            point = np.random.randint(1, len(self.target) - 1)
            father = random.sample(selected, 2)

            population[i][:point] = father[0][:point]
            population[i][point:] = father[1][point:]
            
            population[i] = normalizar(population[i])
           
        return population
    
    def mutation(self, population):
        
        for i in range(len(population)):
            if random.random() <= self.mutation_rate:
                point = np.random.randint(len(self.target))
                new_value = np.random.randint(20, 100)

                while new_value == population[i][point]:
                    new_value = np.random.randint(20, 100)
                
                population[i][point] = new_value
            population[i] = normalizar(population[i])
            return population
    
    def run_geneticalgo(self):
        population = self.create_population()

        for i in range(self.n_generations):

            if self.verbose:
                print('___________')
                print('Generacion: ', i)
                #print('Poblacion', population, self.pasos)
                print('Poblacion\n', population, "\nPasos: ", self.fitness(population[0]), self.pasos)
                #print(max(population[0]),sum(population[0]))
                print()

            selected = self.selection(population)
            population = self.reproduction(population, selected)
            population = self.mutation(population)
def normalizar(array):
    sumatoria = sum(array)
    for i in range(0,len(array)):
        array[i] = (array[i]/sumatoria)
    #print(array)
    return array
def main():
    target = [1,2,3,4,5,6,7,8]
    model = DNA(
        target = target,
        mutation_rate = 0.5,
        n_individuals =2,
        n_selection = 2,
        n_generations = 100,
        verbose=True)
    model.run_geneticalgo()

    
if __name__ == '__main__':
    main()