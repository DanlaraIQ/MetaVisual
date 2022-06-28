import random
import numpy as np
import math
# from solution import solution
import time

# -- PSO


def pso(lb, ub, dim, PopSize, iters, fobj):
    # PSO parameters
    mejor_ruta = []
    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    vel = np.zeros((PopSize, dim))
    pBestScore = np.zeros(PopSize)
    pBestScore.fill(float("inf"))
    pBest = np.zeros((PopSize, dim))
    gBest = np.zeros(dim)
    gBestScore = float("inf")
    datos_grafica_movimiento = []
    pos = np.random.uniform(0, 1, (PopSize, dim)) * (ub - lb) + lb
    tiempo_inicial = time.time()
    for l in range(0, iters):
        una_linea = pos.ravel()
        datos_grafica_movimiento.extend(una_linea)
        for i in range(0, PopSize):
            pos[i, :] = np.clip(pos[i, :], lb, ub)
            # Calculate objective function for each particle
            fitness = fobj(pos[i, :])

            if(pBestScore[i] > fitness):
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :]

            if(gBestScore > fitness):
                gBestScore = fitness
                gBest = pos[i, :]

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iters)

        for i in range(0, PopSize):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = w * vel[i, j] + c1 * r1 * \
                    (pBest[i, j] - pos[i, j]) + c2 * r2 * (gBest[j] - pos[i, j])

                if(vel[i, j] > Vmax):
                    vel[i, j] = Vmax

                if(vel[i, j] < -Vmax):
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]
        mejor_ruta.append(gBestScore)
    mejor_eval = gBestScore
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, mejor_eval, mejor_ruta, gBest

# - DE


def de(lb, ub, dim, PopSize, iters, fobj):
    valorMejor = []
    mutation_factor = 0.5
    crossover_ratio = 0.7
    stopping_func = None
    best = float('inf')

    # initialize population
    population = []
    datos_grafica_movimiento = []

    population_fitness = np.array([float("inf") for _ in range(PopSize)])

    for p in range(PopSize):
        sol = []
        for d in range(dim):
            d_val = random.uniform(lb[d], ub[d])
            sol.append(d_val)

        population.append(sol)

    population = np.array(population)

    # calculate fitness for all the population
    for i in range(PopSize):
        fitness = fobj(population[i, :])
        population_fitness[p] = fitness

        # is leader ?
        if fitness < best:
            best = fitness
            leader_solution = population[i, :]

    # start work

    t = 0
    tiempo_inicial = time.time()
    while t < iters:
        una_linea = population.ravel()
        datos_grafica_movimiento.extend(una_linea)
        # should i stop
        if stopping_func is not None and stopping_func(best, leader_solution, t):
            break

        # loop through population
        for i in range(PopSize):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in range(PopSize) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(dim):
                d_val = population[id_1, d] + mutation_factor * (
                    population[id_2, d] - population[id_3, d]
                )

                # 2. Recombination
                rn = random.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = np.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = fobj(mutant_sol)
            # s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < best:
                    best = mutant_fitness
                    leader_solution = mutant_sol

        # increase iterations
        t = t + 1

        valorMejor.append(best)

    # return solution
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, best, valorMejor, leader_solution

# - GWO


def gwo(lb, ub, dim, SearchAgents_no, Max_iter, fobj):
    valorMejor = []
    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")
    datos_grafica_movimiento = []

    # Initialize the positions of search agents
    Positions = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb

    # Main loop
    tiempo_inicial = time.time()
    for l in range(0, Max_iter):
        una_linea = Positions.ravel()
        datos_grafica_movimiento.extend(una_linea)
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)

            # Calculate objective function for each search agent
            fitness = fobj(Positions[i, :])

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if (fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()

        a = 2 - l * ((2) / Max_iter)  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a  # Equation (3.3)
                C1 = 2 * r2  # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])  # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha  # Equation (3.6)-part 1

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a  # Equation (3.3)
                C2 = 2 * r2  # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])  # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta  # Equation (3.6)-part 2

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a  # Equation (3.3)
                C3 = 2 * r2  # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])  # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta  # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
        valorMejor.append(Alpha_score)
    tiempo = time.time() - tiempo_inicial
    return datos_grafica_movimiento, Alpha_score, valorMejor, Alpha_pos
