import copy
import glob
import pickle
import random
import subprocess
import matplotlib.pyplot as plt

from evaluation import evaluate, ev
fbow_util_path = "/home/decamargo/Documents/FBoW/build/utils/fbow_create_vocabulary"
feature_path = "/home/decamargo/Documents/output_features"
params = []
crossover_prob = 0.78
mutation_prob = 0.08
termination_accuracy = 0.9
max_generations = 100
all_results = {}
chromosome_length = 0
save_path = "./genetic_save.pkl"

class Param:
    def __init__(self, name, minValue, maxValue):
        self.name = name
        self.minValue = minValue
        self.maxValue = maxValue
        self.range = maxValue - minValue
        self.numberOfBits = len("{0:b}".format(self.range))


class Chromosome:
    def __init__(self, chromosome, is_initialized=False):
        # create a random chromosome
        if not is_initialized:
            self.chromosome = []
            for i in range(chromosome):
                self.chromosome.append(random.randint(0, 1))
        else:
            self.chromosome = chromosome


class Individual:
    def __init__(self, chromosome, is_initialized):
        if is_initialized:
            self.fitness = 0
            self.result_dict = {}
            self.chromosome = Chromosome(chromosome, True)
        else:
            self.fitness = 0
            self.result_dict = {}
            chromosome_valid = False
            while not chromosome_valid:
                self.chromosome = Chromosome(chromosome_length)
                chromosome_valid = self.is_chromosome_valid()

    def decode_chromosome(self):
        decoded_dict = {}
        current_bit = 0
        for p in params:
            bit_string = ""
            for i in range(current_bit, current_bit + p.numberOfBits):
                bit_string += str(self.chromosome.chromosome[i])
            current_bit += p.numberOfBits
            decoded_dict[p.name] = (int(bit_string, 2)) + p.minValue
        return decoded_dict

    def is_chromosome_valid(self):
        current_bit = 0
        for p in params:
            bit_string = ""
            for i in range(current_bit, current_bit + p.numberOfBits):
                bit_string += str(self.chromosome.chromosome[i])
            current_bit += p.numberOfBits
            if int(bit_string, 2) > p.maxValue-p.minValue:
                return False
        return True

    def get_one_string(self):
        s = ""
        for i in self.chromosome.chromosome:
            s += str(i)
        return s

    def create_voc(self, name, param_dict):
        params = ""
        for k in param_dict:
           params += " " + "-" + k + " " + str(param_dict[k])
        cmd = fbow_util_path + " " + feature_path + " ./" + name + params
        subprocess.run(cmd, shell=True)
        return "./" + name


class Generation:
    def __init__(self, population, chromosome_length):
        self.population = population
        self.population_fitness = 0
        self.chromosome_length = chromosome_length

    def init_generation(self, population_size):
        population = []
        for p in range(population_size):
            population.append(Individual(self.chromosome_length, False))
        self.population = population

    def get_score(self):
        score = 0.0
        for i in self.population:
            decoded_dict = i.decode_chromosome()
            bin_string = i.get_one_string()
            name = bin_string + ".voc"
            # we are checking if this combination of parameters has already been tested to not build the same vocab again
            if bin_string not in all_results:
                # create voc return path to voc
                voc = i.create_voc(name, decoded_dict)
                fitness, res_dict = evaluate(voc, decoded_dict["norm"], decoded_dict["weight"])
                i.fitness = fitness
                all_results[bin_string] = i.fitness
                i.result_dict = res_dict
            else:
                fitness = all_results[bin_string]
            score += fitness
        self.population_fitness = score / len(self.population)
        return self.population_fitness


class Population:
    def __init__(self, population_size, chromosome_length):
        self.population_size = population_size
        self.highest_individual = Individual(chromosome_length, False)
        self.current_generation = 0
        self.highest_fit_gen_ind = 0
        self.highest_generation_fitness = 0
        self.highest_individual_fitness = 0
        self.chromosome_length = chromosome_length
        self.generations = []
        first_gen = Generation([], self.chromosome_length)
        first_gen.init_generation(self.population_size)
        self.generations.append(first_gen)

    def termination_condition_fulfilled(self):
        if self.current_generation >= max_generations:
            return True
        if self.highest_individual_fitness >= termination_accuracy:
            return True
        return False

    def evaluate_curr_gen(self):
        curr_gen_fitness = self.generations[self.current_generation].get_score()
        if curr_gen_fitness > self.highest_generation_fitness:
            self.highest_generation_fitness = curr_gen_fitness
            self.highest_fit_gen_ind = self.current_generation
        self.generations[self.current_generation].population.sort(key=lambda x: x.fitness, reverse=True)
        highest_ind = self.generations[self.current_generation].population[0]
        if highest_ind.fitness > self.highest_individual_fitness:
            self.highest_individual = highest_ind
            self.highest_individual_fitness = highest_ind.fitness
        self.cleanup_vocabs()

    def create_new_gen(self):
        curr_gen = copy.deepcopy(self.generations[self.current_generation])
        new_population = []
        parents = []
        f = 0
        for k in curr_gen.population:
            f += k.fitness * k.fitness

        for m in range(len(curr_gen.population)):
            w = random.uniform(0.0, f)
            d = 0.0
            for i in curr_gen.population:
                d += i.fitness * i.fitness
                if d >= w:
                    # parent found
                    parents.append(i)
                    break

        # parents found
        offsprings = self.generate_offsprings(parents)
        for o in offsprings:
            m = self.mutation(o)
            new_population.append(m)
        new_generation = Generation(new_population, curr_gen.chromosome_length)
        print("Generation " + str(self.current_generation))
        print("Highest individual fitness: " + str(self.highest_individual_fitness))
        print("Avg. generation fitness: " + str(self.generations[self.current_generation].population_fitness))
        self.generations.append(new_generation)
        self.current_generation += 1
        self.save()


    def generate_offsprings(self, parents):
        new_population = []
        for i in range(0, len(parents), 2):
            child1, child2 = self.generate_offspring(parents[i], parents[i + 1])
            new_population.append(child1)
            new_population.append(child2)
        return new_population

    def generate_offspring(self, parent1, parent2):
        chrom_length = len(parent1.chromosome.chromosome)
        ret1_valid = False
        ret2_valid = False
        if random.uniform(0.0, 1.0) <= crossover_prob:
            for j in range (chrom_length):
                ran = random.randint(1,chrom_length-2)
                if not ret1_valid:
                    child1_chrom = []
                if not ret2_valid:
                    child2_chrom = []
                for i in range(ran):
                    if not ret1_valid:
                        child1_chrom.append(parent1.chromosome.chromosome[i])
                    if not ret2_valid:
                        child2_chrom.append(parent2.chromosome.chromosome[i])
                for b in range(ran, chrom_length):
                    if not ret1_valid:
                        child1_chrom.append(parent1.chromosome.chromosome[b])
                    if not ret2_valid:
                        child2_chrom.append(parent2.chromosome.chromosome[b])
                if not ret1_valid:
                    child1 = Individual(child1_chrom, True)
                if not ret2_valid:
                    child2 = Individual(child2_chrom, True)
                if not ret1_valid and child1.is_chromosome_valid():
                    ret1_valid = True
                    ret1 = copy.deepcopy(child1)
                if not ret2_valid and child2.is_chromosome_valid():
                    ret2_valid = True
                    ret2 = copy.deepcopy(child2)
            if ret1_valid:
                child1 = ret1
            else:
                child1 = parent1
                print("c1 offspring failed")

            if ret2_valid:
                child2 = ret2
            else:
                child2 = parent2
                print("c2 offspring failed")

            return child1, child2
        else:
            return parent1, parent2

    def mutation(self, individual):
        mutated_ind = copy.deepcopy(individual)
        for i in range(len(individual.chromosome.chromosome)):
            if random.uniform(0, 1) < mutation_prob:
                if mutated_ind.chromosome.chromosome[i] == 0:
                    mutated_ind.chromosome.chromosome[i] = 1
                else:
                    mutated_ind.chromosome.chromosome[i] = 0
        # only mutate if still valid chromosome
        if mutated_ind.is_chromosome_valid():
            return mutated_ind
        else:
            print("mutation failed")
            return individual

    def cleanup_vocabs(self):
        vocs = glob.glob("*.voc")
        fittest_voc_name = self.highest_individual.get_one_string() + ".voc"
        for v in vocs:
            if not v == fittest_voc_name:
                cmd = "rm " + v
                subprocess.run(cmd, shell=True)

    def save(self):
        print("--------------------")
        print("SAVING")
        with open(save_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("DONE SAVING")
        print("--------------------")


# when new generation has been build:
# 1. go through each individual, build vocab and evaluate it to get fitness score
# 2. compare individual fitnessscore to highest ind fitnessscore -> if highest save ind
# 3. get avg generation fitnessscore -> if higher save generation
# 4. delete vocabs of generation if not saved
# 5. check if term criteria reached.
# 6. create new generation

def run_genetic_agorithm(population_size):
    global chromosome_length
    for p in params:
        chromosome_length += p.numberOfBits
    pop = Population(population_size, chromosome_length)
    while not pop.termination_condition_fulfilled():
        pop.evaluate_curr_gen()
        pop.create_new_gen()


def plot_results(pkl_path):
    with open(save_path, 'rb') as handle:
        b = pickle.load(handle)
    pop_fitness = []
    for gen in b.generations:
        pop_fitness.append(gen.population_fitness)
    plt.plot(pop_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Avg. fitness")
    plt.show()
    ind_fitness = []
    highest_score = 0.0
    for gen in b.generations:
        for ind in gen.population:
            if ind.fitness > highest_score:
                highest_score = ind.fitness
        ind_fitness.append(highest_score)
    plt.plot(ind_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Highest individual fitness")
    plt.show()


if __name__ == '__main__':
    # we get a freshly generated voc and evaluate it here
    #with open(save_path, 'rb') as handle:
    #    b = pickle.load(handle)
    k = Param("k", 2, 14)
    l = Param("l", 1, 7)
    weight = Param("weight", 0, 3)
    norm = Param("norm", 0, 2)
    distance = Param("distance", 0, 1)
    params.append(k)
    params.append(l)
    params.append(weight)
    params.append(norm)
    params.append(distance)
    population_size = 10
    run_genetic_agorithm(population_size)
    #plot_results(save_path)

