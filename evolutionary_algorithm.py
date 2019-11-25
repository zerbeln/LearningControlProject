#!/usr/bin/env python

import numpy as np
import random


class EvoAlg:

    def __init__(self, p):
        self.parent_psize = p.parent_pop_size  # Number of policies in parent pop
        self.offspring_psize = p.offspring_pop_size  # Number of policies in offspring pop
        self.total_pop_size = p.parent_pop_size + p.offspring_pop_size  # Total number of policies from each pop
        self.mut_chance = p.prob_mutation  # Probability that a specific weight will be mutated
        self.mut_rate = p.mutation_rate  # Maximum amount of change allowed by mutation
        self.eps = p.epsilon  # For e-greedy selection
        n_inputs = p.num_inputs; n_outputs = p.num_outputs; n_nodes = p.num_hidden
        self.policy_size = (n_inputs + 1)*n_nodes + (n_nodes + 1) * n_outputs  # Number of weights for NN

        self.pops = np.zeros((self.total_pop_size, self.policy_size))
        self.parent_pop = np.zeros((p.parent_pop_size, self.policy_size))
        self.offspring_pop = np.zeros((p.offspring_pop_size, self.policy_size))
        self.parent_fitness = np.zeros(self.parent_psize)
        self.offspring_fitness = np.zeros(self.offspring_psize)
        self.fitness = np.zeros(self.total_pop_size)

    def reset_populations(self):  # Re-initializes EA population for new run
        """
        Create new populations (for beginning of stat run)
        :return: None
        """

        self.pops = np.zeros((self.total_pop_size, self.policy_size))
        self.parent_pop = np.zeros((self.parent_psize, self.policy_size))
        self.offspring_pop = np.zeros((self.offspring_psize, self.policy_size))
        self.fitness = np.zeros(self.total_pop_size)
        self.parent_fitness = np.zeros(self.parent_psize)
        self.offspring_fitness = np.zeros(self.offspring_psize)

        # Create intitial population of NN weights using a normal distribution
        for policy_id in range(self.parent_psize):
            for w in range(self.policy_size):
                weight = np.random.normal(0, 1)
                self.parent_pop[policy_id, w] = weight
        for policy_id in range(self.offspring_psize):
            for w in range(self.policy_size):
                weight = np.random.normal(0, 1)
                self.offspring_pop[policy_id, w] = weight

        self.combine_pops()

    def mutate(self):  # Mutate policy based on probability
        """
        Mutate offspring populations
        :return: None
        """

        for policy_id in range(self.offspring_psize):
            for w in range(self.policy_size):
                rnum = random.uniform(0, 1)
                if rnum < self.mut_chance:
                    weight = self.offspring_pop[policy_id, w]
                    mutation = np.random.normal(0, self.mut_rate)*weight
                    self.offspring_pop[policy_id, w] += mutation

    def epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents from which an offspring population will be created
        :return: None
        """

        for policy_id in range(self.parent_psize):
            rnum = random.uniform(0, 1)
            if rnum > self.eps:  # Choose best policy
                pol_index = np.argmax(self.fitness)
                self.parent_pop[policy_id] = self.pops[pol_index].copy()
                self.parent_fitness[policy_id] = self.fitness[pol_index]
            else:
                parent = random.randint(0, (self.total_pop_size-1))  # Choose a random parent
                self.parent_pop[policy_id] = self.pops[parent].copy()
                self.parent_fitness[policy_id] = self.fitness[parent]

    def down_select(self):  # Create a new offspring population
        """
        Select parent,s create offspring population, and perform mutation operations
        :return: None
        """
        self.combine_pops()
        self.epsilon_greedy_select()  # Select K solutions using epsilon greedy
        self.offspring_pop = self.parent_pop.copy()  # Produce K offspring
        self.mutate()  # Mutate offspring population

    def combine_pops(self):  # Merge pop and fitness arrays for selection process
        """
        Combine parent and offspring populations into single population array
        :return: None
        """

        self.fitness = np.concatenate((self.parent_fitness, self.offspring_fitness), axis=None)
        self.pops = np.concatenate((self.parent_pop, self.offspring_pop), axis=0)
