import pandas as pd
import numpy as np
import ast
import json
import random
import time
import re

from MetaCVI import Meta_CVI
from MetaAlgorithm import Algorithm
from MetaConfig import Config
from cvi import Validation
from scoop import futures

from sklearn import metrics
from deap import base, creator, tools, algorithms
from ClusGeneticMethods import GenBirch, GenDBSCAN, GenKMeans, GenOptics, GenAffinityPropagation, GenMeanshift, GenAgglomerative, GenSpectral
from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
	AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
	OPTICS, Birch

class AutoClustering:

	# Parameters required
	# data, time budget, population size, generation size,
	# crossover + mutation offspring size

	def __init__(self, filename, generations, population, pre_config=False, pre_cvi=False, time=None):
		self.filename = filename
		self.data = pd.read_csv(filename).iloc[:, :-1]
		self.time = time

		# Recommend CVI or Configuration
		if pre_config:
			self.pre_config = True
			config = Config(filename, "distance")
			self.config, self.cvi, algorithm = config.search()
			self.model = self.get_model(algorithm)
			self.cvi = ast.literal_eval(self.cvi)
			tf = self.config[1:-1]
			pattern = re.split(r',\s*(?![^()]*\))', tf)
			formatted_config = list()
			for p in pattern:
				p = eval(p)
				formatted_config.append(p)
			self.config = formatted_config
		elif pre_cvi:
			self.pre_config = False
			algorithm = Algorithm(filename, "distance").search()
			# algorithm = "spectral"
			self.cvi = Meta_CVI(filename, "distance").search(algorithm)
			self.model = self.get_model(algorithm)
		else:
			self.pre_config = False
			self.cvi = [['i_index', 1], ['modified_hubert_t', 1], ['banfeld_raferty', -1]]
			self.model = self.get_model(random.choice(['kmeans', 'ag', 'birch', 'spectral', 'meanshift', 'optics', 'ap', 'db']))

		# print(self.config)

		# Creator
		fitness_weights = (np.float(self.cvi[0][1]), np.float(self.cvi[1][1]), np.float(self.cvi[2][1]))
		creator.create("FitnessMulti", base.Fitness, weights=fitness_weights)
		creator.create("Individual", list, fitness=creator.FitnessMulti)

		# Toolbox initialization
		self.toolbox = base.Toolbox()
		if self.pre_config:
			self.toolbox.register("individual", self.init_individual, creator.Individual)
			self.toolbox.register("population", self.init_population, list, self.toolbox.individual)
		else:
			self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.gen_population, n=1)
			self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		self.toolbox.register("evaluate", self.fitness_function)
		self.toolbox.register("mate", self.crossover)
		self.toolbox.register("mutate", self.mutate)
		self.toolbox.register("select", tools.selNSGA2)
		# self.toolbox.register("map", futures.map)

		# Evolution Algorithm parameter initialization
		self.generations = generations
		self.pop_size = population

	# Get configuration parameters
	def get_parameters(self):
		return self.model.algorithm, self.cvi
	# Generate random population
	def gen_population(self):
		return self.model.generate_pop(self)

	# Initialize population
	def init_population(self, pcls, ind_init):
		if self.pre_config:
			contents = self.config
			return pcls(ind_init(c) for c in contents)
		else:
			population = []
			for i in range(0, 10):
				population.append(self.gen_population())
			return population

	# Initialize individual
	def init_individual(self, icls, content):
		if self.pre_config:
			return icls(content)
		else:
			return self.gen_population()

	# Evaluate individual fitness
	def fitness_function(self, individual):
		try:
			clustering = individual[0].fit(self.data)
			labels = clustering.labels_
		except:
			return (0,0,0)
		try:
			validate = Validation(np.asmatrix(self.data).astype(np.float), np.asarray(self.data), labels)
			metric_values = validate.run_list([self.cvi[0][0], self.cvi[1][0], self.cvi[2][0]])
			return metric_values[self.cvi[0][0]], metric_values[self.cvi[1][0]], metric_values[self.cvi[2][0]]
		except:
			return (0,0,0)

	# Crossover individual, call related model
	def crossover(self, ind1, ind2):
		return self.model.crossover(ind1, ind2)

	# Mutate individual, call related model
	def mutate(self, individual):
		return self.model.mutate(individual)

	# Get Clustering Algorithm model(s)
	def get_model(self, algorithm):
		model__labels = {'ap': GenAffinityPropagation(),
						 'db': GenDBSCAN(),
						 'ag': GenAgglomerative(),
						 'optics': GenOptics(),
						 'birch': GenBirch(),
						 'spectral': GenSpectral(),
						 'meanshift': GenMeanshift()}
		return model__labels[algorithm]

	# Search best parameter configurations
	def search(self):
		if self.pre_config:
			pop = self.toolbox.population()
		else:
			pop = self.toolbox.population(n=self.pop_size)

		hof = tools.ParetoFront()

		# self.random_search(pop, self.toolbox,  generations=self.generations, halloffame=hof)
		algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.3, ngen=self.generations, verbose=True, halloffame=hof)
		# if self.time is not None:
		# self.eaSimpleTimed(pop, self.toolbox, cxpb=0.7, mutpb=0.3, budget=self.time, verbose=True, halloffame=hof)
		return pop, hof

	# Random search baseline algorithm
	def random_search(self, population, toolbox, generations, halloffame=None, stats=None):

		logbook = tools.Logbook()
		logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in population if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		if halloffame is not None:
			halloffame.update(population)

		record = stats.compile(population) if stats else {}
		logbook.record(gen=0, nevals=len(invalid_ind), **record)

		for gen in range(generations):
			# Select the next generation individuals
			offspring = toolbox.select(population, len(population))

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			# Update the hall of fame with the generated individuals
			if halloffame is not None:
				halloffame.update(offspring)

			# Replace the current population by the offspring
			population[:] = offspring

			# Append the current generation statistics to the logbook
			record = stats.compile(population) if stats else {}
			logbook.record(gen=gen, nevals=len(invalid_ind), **record)

		return population, logbook

	def eaSimpleTimed(self, population, toolbox, cxpb, mutpb, budget, stats=None,
				 halloffame=None, verbose=__debug__):

		logbook = tools.Logbook()
		logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in population if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		if halloffame is not None:
			halloffame.update(population)

		record = stats.compile(population) if stats else {}
		logbook.record(gen=0, nevals=len(invalid_ind), **record)
		if verbose:
			print
			logbook.stream

		# Begin the [TIMED] generational process
		start_time = time.time()
		gen = 0
		# print("Start: ", start_time)
		# print("Print: ", budget)
		# for gen in range(1, ngen + 1):
		while time.time() - start_time <= budget:
			# Track number of generations
			gen += 1
			# Select the next generation individuals
			offspring = toolbox.select(population, len(population))

			# Vary the pool of individuals
			offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			# Update the hall of fame with the generated individuals
			if halloffame is not None:
				halloffame.update(offspring)

			# Replace the current population by the offspring
			population[:] = offspring

			# Append the current generation statistics to the logbook
			record = stats.compile(population) if stats else {}
			logbook.record(gen=gen, nevals=len(invalid_ind), **record)
			if verbose:
				print
				logbook.stream

		return population, logbook
