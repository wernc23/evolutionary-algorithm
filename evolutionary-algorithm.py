import random
import numpy as np
import matplotlib.pyplot as plt


#########################################################
# PARAMETERS                                            #
#########################################################
popSize = 100
chromLength = 300
iteration_max = 1000
crossover_rate = 0.7
mutation_rate = 0.1
tournamentSize = 2

fitness = np.empty([popSize])
costVector = np.empty([chromLength])


#########################################################
# Load network                                          #
#########################################################
def loadNetwork():
    fname = "network.txt"
    input = np.loadtxt(fname)
    for i in range(0, chromLength):
        costVector[i] = input[i][2]


#########################################################
# FITNESS EVALUATION                                    #
#########################################################
def evaluateFitness(chromosomes, best, bestChrom):
    costFullyConnectedNetwork = 30098.059999999983
    fitness_total = 0.0
    fitness_average = 0.0

    # For all chromosomes
    for i in range(0, popSize):
        fitness[i] = 0

    # For all chromosomes
    for i in range(0, popSize):
        cost = 0
        for j in range(0, chromLength):
            if chromosomes[i][j] == 1:
                cost = cost+costVector[j]
        fitness[i] = 1-(cost/costFullyConnectedNetwork)
        fitness_total = fitness_total+fitness[i]
    fitness_average = fitness_total/popSize

    for i in range(0, popSize):
        if fitness[i] >= best:
            best = fitness[i]
            bestChrom = chromosomes[i]

    return best, fitness_average, fitness_total, bestChrom


#########################################################
# PERFORMANCE GRAPH                                     #
#########################################################
def plotChart(best, avg):
    plt.plot(best, label='best')
    plt.plot(avg, label='average')
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.legend()
    plt.xlim(1, iteration_max-1)
    plt.ylim(0.0, 1.0)
    plt.show()


# This function is responsible for creating the initial chromosome population
# of size 'chromLength'
# It appends a new chromosome for each index 0 ... popSize using the np.random.randint function
# the np.random.randit function will create an array of length chromLength and populate the elements
# of a random int between 0 and 2, so 0 or 1.
def generateInitialPopulation():
    count = 0
    chromosomes = []
    # create each chromosome
    while (count < popSize):
        # create array of length chromLength with elements either 0 or 1
        chromosomes.append(np.random.randint(0, 2, chromLength))
        count = count + 1
    return chromosomes

# This function is responsible for creating and returning two parent chromosome use for
# crossover and mutation.
# @param chromosomes {Array[][]} entire population of chromosomes


def selectChromosomesForCrossover(chromosomes):
    # get two parents
    parent1 = tournamentSelection(chromosomes)
    parent2 = tournamentSelection(chromosomes)
    return parent1, parent2

# This function is responsible for running the tournament. It iterates from 0 to the tournmanetSize
# and finds the fittest chromosome which is returned from this function
# @param chromosomes {Array[][]} entire population of chromosomes


def tournamentSelection(chromosomes):
    bestFitness = 0.0
    bestFitnessIndex = -1
    count = 0
    # while count is less than k (tournamentSize)
    while (count < tournamentSize):
        # select random index
        selectedIndex = random.randint(0, popSize - 1)
        # if fitness is better than bestFitness
        if(fitness[selectedIndex] > bestFitness):
            # bestFitness is the fitness index of selectedIndex
            bestFitness = fitness[selectedIndex]
            # remember selectedIndex so that we can return chromosome found with best fitness
            bestFitnessIndex = selectedIndex
        count = count + 1
    return chromosomes[bestFitnessIndex]

# This function is responsible for swapping the middle part between two points in the chromosome parents.
# It gets two random numbers for the two point swap and creates a new chromosome from the subarrays that
# were spliced.


def crossoverSwap(chrom1, chrom2):
    # get two random points
    point1 = random.randint(0, chromLength - 1)
    point2 = random.randint(0, chromLength - 1)
    # get high and low for swapping technique
    if(point1 < point2):
        low = point1
        high = point2
    else:
        high = point1
        low = point2

    # create new array by concating subarrays together
    newChrom1 = np.concatenate(
        (
            chrom1[0:low],
            chrom2[low: high],
            chrom1[high: len(chrom1)]
        )
    )
    newChrom2 = np.concatenate(
        (
            chrom2[0:low],
            chrom1[low: high],
            chrom2[high: len(chrom2)]
        )
    )
    return newChrom1, newChrom2

# This function is responsible for deciding if there should be a crossover or not
# depending on a random number generated and the crossover rate


def crossover(chrom1, chrom2):
    # Get number between 0 and 1
    crossoverRandomNumber = random.uniform(0, 1)
    # if that number is less than or equal to the crossover rate, crossover should occur
    if(crossoverRandomNumber <= crossover_rate):
        # swap subarray of chromosome
        chrom1, chrom2 = crossoverSwap(chrom1, chrom2)
    return chrom1, chrom2

# This function is responsible for the mutuation process.
# It creates a random number generated and checks if that number is less than or equal to the
# mutation rate. If it is then mutation occours. It creates another random number between
# 0 ... chromLength - 1 and flips the bit located in that index of the chromosome.


def mutation(chromes):
    count = 0
    # for each chromosome passed in
    while (count < len(chromes)):
        # get random number between 0 and 1
        crossoverRandomNumber = random.uniform(0, 1)
        # if number is less than or equal to mutation rate
        if(crossoverRandomNumber <= mutation_rate):
            # mutate chromosome by flipping random bit
            randomBit = random.randint(0, chromLength - 1)
            if(chromes[count][randomBit] == 0):
                chromes[count][randomBit] = 1
            else:
                chromes[count][randomBit] = 0
        count = count + 1
    return chromes

# This function is responsible for calling all the functions that handle the
# process of selecting the parent chromosomes, handling crossover, handling mutation,
# and returning the new population.


def newPopulationSequence(chromosomes):
    newCromosomes = []
    count = 0
    while(count < (popSize / 2)):
        # Select Chromosomes
        chrom1, chrom2 = selectChromosomesForCrossover(chromosomes)
        # Crossover
        crossoverChrom1, crossoverChrom2 = crossover(chrom1, chrom2)
        # Mutation
        mutatedChromes = mutation(
            [
                crossoverChrom1,
                crossoverChrom2
            ]
        )
        # append new chromosomes
        mutatedCount = 0
        while (mutatedCount < len(mutatedChromes)):
            newCromosomes.append(mutatedChromes[mutatedCount])
            mutatedCount = mutatedCount + 1

        # increase iteration
        count = count + 1

    return newCromosomes


#########################################################
# MAIN                                                  #
#########################################################
if __name__ == '__main__':
    best = 0.0
    average = 0.0
    iteration = 0
    bestChrom = []
    bestM = np.empty([iteration_max], dtype=np.float32)
    averageM = np.empty([iteration_max], dtype=np.float32)
    chromosomes = np.empty([popSize], dtype=np.byte)
    print("GENETIC ALGORITHM APPLIED TO OVERLAY NETWORK OPTIMIZATION")
    # load network
    loadNetwork()
    # Initialize population of chromosomes
    chromosomes = generateInitialPopulation()
    while (iteration < iteration_max-1):
        chromosomes = newPopulationSequence(chromosomes)
        best, average, fitness_total, bestChrom = evaluateFitness(
            chromosomes, best, bestChrom)
        bestM[iteration] = best
        averageM[iteration] = average
        iteration = iteration + 1
        print("iteration: ", iteration)
        print("best fitness: ", best)

    print("best fitness: ", best)
    print("average fitness: ", average)
    print("bestChrom: ", bestChrom)

    plotChart(bestM, averageM)
