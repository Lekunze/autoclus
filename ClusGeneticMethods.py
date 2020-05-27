from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
	AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
	OPTICS, Birch
import random
import numpy as np
import copy

'''
each class is designed to support the functions needed for genetic optimization 
'''


class GenKMeans:

	def __init__(self):
		self.params = ["n_clusters", "algorithm", "init", "n_init"]
		self.algorithm = "KMeans"

	@staticmethod
	def generate_pop(self):
		n_clusters = random.randint(2, 100)
		init = random.choice(['k-means++', 'random'])
		algorithm = random.choice(['auto', 'full', 'elkan'])
		n_init = random.choice(list(range(10, 25)))
		population = KMeans(n_clusters=n_clusters, algorithm=algorithm, init=init, n_init=n_init)

		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = len(self.params)
		tpop = copy.deepcopy(pop)

		for i in range(1, 3):
			pos = random.randint(0, len_params - 1)
			if pos <= 3:
				tpop[0].n_init = p.n_init
			if pos <= 2:
				tpop[0].init = p.init
			if pos <= 1:
				tpop[0].algorithm = p.algorithm
			if pos == 0:
				tpop[0].n_clusters = p.n_clusters

		return tpop,

	def crossover(self, pop, pop2):
		len_params = len(self.params)
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(1, 3):
			pos = random.randint(0, len_params - 1)
			if pos <= 3:
				tpop[0].n_init, tpop2[0].n_init = tpop2[0].n_init, tpop[0].n_init
			if pos <= 2:
				tpop[0].init, tpop2[0].init = tpop2[0].init, tpop[0].init
			if pos <= 1:
				tpop[0].algorithm, tpop2[0].algorithm = tpop2[0].algorithm, tpop[0].algorithm
			if pos == 0:
				tpop[0].n_clusters, tpop2[0].n_clusters = tpop2[0].n_clusters, tpop[0].n_clusters

		return tpop, tpop2


class GenMeanshift:

	def __init__(self):
		self.params = ["cluster_all", "bin_seeding", "init", "n_init"]
		self.algorithm = "Meanshift"

	@staticmethod
	def generate_pop(self):
		cluster_all = random.choice([True, False])
		bin_seeding = random.choice([True, False])
		bandwidth = random.choice([None, 1, 2, 3, 4, 5])
		max_iter = random.choice([200, 300, 400, 500, 600, 700])
		population = MeanShift(cluster_all=cluster_all, bin_seeding=bin_seeding, bandwidth=bandwidth, max_iter=max_iter)
		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = len(self.params)
		tpop = copy.deepcopy(pop)

		for i in range(3):
			pos = random.randint(0, len_params - 1)
			if pos <= 3:
				tpop[0].max_iter = p.max_iter
			if pos <= 2:
				tpop[0].bandwidth = p.bandwidth
			if pos <= 1:
				tpop[0].bin_seeding = p.bin_seeding
			if pos == 0:
				tpop[0].cluster_all = p.cluster_all

		return tpop,

	def crossover(self, pop, pop2):

		len_params = len(self.params)
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(3):
			pos = random.randint(0, len_params - 1)
			if pos <= 3:
				tpop[0].max_iter, tpop2[0].max_iter = tpop2[0].max_iter, tpop[0].max_iter
			if pos <= 2:
				tpop[0].bandwidth, tpop2[0].bandwidth = tpop2[0].bandwidth, tpop[0].bandwidth
			if pos <= 1:
				tpop[0].bin_seeding, tpop2[0].bin_seeding = tpop2[0].bin_seeding, tpop[0].bin_seeding
			if pos == 0:
				tpop[0].cluster_all, tpop2[0].cluster_all = tpop2[0].cluster_all, tpop[0].cluster_all

			return tpop, tpop2


class GenDBSCAN:

	def __init__(self):
		self.params = ["eps", "min_samples", "metric", "algorithm", "leaf_size", "p"]
		self.algorithm = "DBSCAN"

	@staticmethod
	def generate_pop(self):

		eps = random.choice([0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
		min_samples = random.choice([5, 10, 15, 20, 30, 50, 100, 150, 200])
		metric = random.choice(['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'])
		algorithm = random.choice(["auto", "ball_tree", "kd_tree", "brute"])
		leaf_size = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200])
		p = random.choice([1, 2, 3])
		population = DBSCAN(eps=eps, metric=metric, min_samples=min_samples, algorithm=algorithm, leaf_size=leaf_size,
							p=p)

		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = len(self.params)
		tpop = copy.deepcopy(pop)

		for i in range(3):
			pos = random.randint(0, len_params - 1)
			if pos <= 5:
				tpop[0].eps = p.eps
			if pos <= 4:
				tpop[0].metric = p.metric
			if pos <= 3:
				tpop[0].min_samples = p.min_samples
			if pos <= 2:
				tpop[0].algorithm = p.algorithm
			if pos <= 1:
				tpop[0].leaf_size = p.leaf_size
			if pos == 0:
				tpop[0].p = p.p

		return tpop,

	def crossover(self, pop, pop2):
		len_params = len(self.params)
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(4):
			pos = random.randint(0, len_params - 1)

			if pos <= 5:
				tpop[0].eps, tpop2[0].eps = tpop2[0].eps, tpop[0].eps
			if pos <= 4:
				tpop[0].metric, tpop2[0].metric = tpop2[0].metric, tpop[0].metric
			if pos <= 3:
				tpop[0].min_samples, tpop2[0].min_samples = tpop[0].min_samples, tpop2[0].min_samples
			if pos <= 2:
				tpop[0].algorithm, tpop2[0].algorithm = tpop2[0].algorithm, tpop[0].algorithm
			if pos <= 1:
				tpop[0].leaf_size, tpop2[0].leaf_size = tpop2[0].leaf_size, tpop[0].leaf_size
			if pos == 0:
				tpop[0].p, tpop2[0].p = tpop2[0].p, tpop[0].p

		return tpop, tpop2


class GenAffinityPropagation:

	def __init__(self):
		self.params = ["damping", "max_iter", "affinity"]
		self.len_params = len(self.params)  # number of parameters
		self.algorithm = "Affinity Propagation"

	@staticmethod
	def generate_pop(self):
		damping = random.uniform(0.5, 1)
		max_iter = random.randint(100, 300)
		affinity = random.choice(['euclidean', 'precomputed'])
		population = AffinityPropagation(damping=damping, max_iter=max_iter, affinity=affinity)
		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = self.len_params
		tpop = copy.deepcopy(pop)

		for i in range(2):
			pos = random.randint(0, len_params - 1)
			if pos <= 2:
				tpop[0].damping = p.damping
			if pos <= 1:
				tpop[0].max_iter = p.max_iter
			if pos == 0:
				tpop[0].affinity = p.affinity

		return tpop,

	def crossover(self, pop, pop2):
		p = copy.deepcopy(pop2)
		len_params = self.len_params
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(2):
			pos = random.randint(0, len_params - 1)
			if pos <= 2:
				tpop[0].damping, tpop2[0].damping = tpop2[0].damping, tpop[0].damping
			if pos <= 1:
				tpop[0].max_iter, p.max_iter = tpop2[0].max_iter, tpop[0].max_iter
			if pos == 0:
				tpop[0].affinity, tpop2[0].affinity = tpop2[0].affinity, tpop[0].affinity

		return tpop, tpop2


class GenSpectral:

	def __init__(self):
		self.params = ["n_clusters", "eigen_solver",
					   "n_init", "gamma", "affinity"]
		self.len_params = len(self.params)
		self.algorithm = "Spectral Clustering"

	@staticmethod
	def generate_pop(self):
		n_clusters = random.randint(2, 100)
		eigen_solver = random.choice([None, 'arpack', 'lobpcg', 'amg'])
		n_init = random.randint(1, 20)
		gamma = random.uniform(0.5, 3)
		affinity = random.choice(['nearest_neighbors', 'rbf'])
		population = SpectralClustering(n_clusters=n_clusters,
										eigen_solver=eigen_solver,
										n_init=n_init,
										gamma=gamma,
										affinity=affinity)
		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = self.len_params
		tpop = copy.deepcopy(pop)

		for i in range(4):
			pos = random.randint(0, len_params - 1)
			if pos <= 4:
				tpop[0].n_clusters = p.n_clusters
			if pos <= 3:
				tpop[0].eigen_solver = p.eigen_solver
			if pos <= 2:
				tpop[0].n_init = p.n_init
			if pos <= 1:
				tpop[0].gamma = p.gamma
			if pos == 0:
				tpop[0].affinity = p.affinity

		return tpop,

	def crossover(self, pop, pop2):

		len_params = self.len_params
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(4):
			pos = random.randint(0, len_params - 1)
			if pos <= 4:
				tpop[0].n_clusters, tpop2[0].n_clusters = tpop2[0].n_clusters, tpop[0].n_clusters
			if pos <= 3:
				tpop[0].eigen_solver, tpop2[0].eigen_solver = tpop2[0].eigen_solver, tpop2[0].eigen_solver
			if pos <= 2:
				tpop[0].n_init, tpop2[0].n_init = tpop2[0].n_init, tpop[0].n_init
			if pos <= 1:
				tpop[0].gamma, tpop2[0].gamma = tpop2[0].gamma, tpop[0].gamma
			if pos == 0:
				tpop[0].affinity, tpop2[0].affinity = tpop2[0].affinity, tpop[0].affinity

		return tpop, tpop2


class GenAgglomerative:

	def __init__(self):
		self.params = ["n_clusters", "linkage", "affinity"]
		self.len_params = len(self.params)
		self.algorithm = "Agglomerative Clustering"

	@staticmethod
	def generate_pop(self):
		n_clusters = random.randint(2, 100)
		linkage = random.choice(['ward', 'complete', 'average', 'single'])
		affinity = random.choice(['euclidean', 'l1', 'l2',
								  'manhattan', 'cosine'])
		population = AgglomerativeClustering(n_clusters=n_clusters,
											 linkage=linkage,
											 affinity=affinity)
		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = self.len_params
		tpop = copy.deepcopy(pop)

		for i in range(2):
			pos = random.randint(0, len_params - 1)
			if pos <= 2:
				tpop[0].n_clusters = p.n_clusters
			if pos <= 1:
				tpop[0].linkage = p.linkage
			if pos == 0:
				tpop[0].affinity = p.affinity

		return tpop,

	def crossover(self, pop, pop2):
		len_params = self.len_params
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(2):
			pos = random.randint(0, len_params - 1)
			if pos <= 2:
				tpop[0].n_clusters, tpop2[0].n_clusters = tpop2[0].n_clusters, tpop[0].n_clusters
			if pos <= 1:
				tpop[0].linkage, tpop2[0].linkage = tpop2[0].linkage, tpop[0].linkage
			if pos == 0:
				tpop[0].affinity, tpop2[0].affinity = tpop2[0].affinity, tpop[0].affinity

		return tpop, tpop2


class GenOptics:

	def __init__(self):
		self.params = ["min_samples", "max_eps", "metric",
					   "cluster_method", "algorithm"]
		self.len_params = len(self.params)
		self.algorithm = "OPTICS"

	@staticmethod
	def generate_pop(self):
		min_samples = random.uniform(0, 1)
		max_eps = random.choice([np.inf, random.uniform(1, 100)])
		metric = random.choice(['cityblock', 'cosine', 'euclidean',
								'l1', 'l2', 'manhattan', 'braycurtis',
								'canberra', 'chebyshev', 'correlation',
								'dice', 'hamming', 'jaccard', 'kulsinski',
								'mahalanobis', 'minkowski', 'rogerstanimoto',
								'russellrao', 'seuclidean', 'sokalmichener',
								'sokalsneath', 'sqeuclidean', 'yule'])
		cluster_method = random.choice(['xi', 'dbscan'])
		algorithm = random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])

		population = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric,
							cluster_method=cluster_method, algorithm=algorithm)
		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = self.len_params
		tpop = copy.deepcopy(pop)

		for i in range(4):
			pos = random.randint(0, len_params - 1)

			if pos <= 4:
				tpop[0].min_samples = p.min_samples
			if pos <= 3:
				tpop[0].max_eps = p.max_eps
			if pos <= 2:
				tpop[0].metric = p.metric
			if pos <= 1:
				tpop[0].cluster_method = p.cluster_method
			if pos == 0:
				tpop[0].algorithm = p.algorithm

		return tpop,

	def crossover(self, pop, pop2):

		len_params = self.len_params
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(4):
			pos = random.randint(0, len_params - 1)

			if pos <= 4:
				tpop[0].min_samples, tpop2[0].min_samples = tpop2[0].min_samples, tpop[0].min_samples
			if pos <= 3:
				tpop[0].max_eps, tpop2[0].max_eps = tpop2[0].max_eps, tpop[0].max_eps
			if pos <= 2:
				tpop[0].metric, tpop2[0].metric = tpop2[0].metric, tpop[0].metric
			if pos <= 1:
				tpop[0].cluster_method, tpop2[0].cluster_method = tpop2[0].cluster_method, tpop[0].cluster_method
			if pos == 0:
				tpop[0].algorithm, tpop[0].algorithm = tpop2[0].algorithm, tpop[0].algorithm

		return tpop, tpop2


class GenBirch:

	def __init__(self):
		self.params = ["threshold", "branching_factor", "compute_labels", "copy"]
		self.len_params = len(self.params)
		self.algorithm = "Birch Clustering"

	@staticmethod
	def generate_pop(self):
		threshold = random.uniform(0.2, 2)
		branching_factor = random.randint(1, 100)
		compute_labels = random.choice([True, False])
		copy = random.choice([True, False])
		population = Birch(threshold=threshold, branching_factor=branching_factor,
								 compute_labels=compute_labels, copy=copy)
		return population

	def mutate(self, pop):
		p = self.generate_pop(self)
		len_params = self.len_params
		tpop = copy.deepcopy(pop)

		for i in range(3):
			pos = random.randint(0, len_params - 1)

			if pos <= 3:
				tpop[0].threshold = p.threshold
			if pos <= 2:
				tpop[0].branching_factor = p.branching_factor
			if pos <= 1:
				tpop[0].compute_labels = p.compute_labels
			if pos == 0:
				tpop[0].copy = p.copy

		return tpop,

	def crossover(self, pop, pop2):

		len_params = self.len_params
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(3):
			pos = random.randint(0, len_params - 1)

			if pos <= 3:
				tpop[0].threshold, tpop2[0].threshold = tpop2[0].threshold, tpop[0].threshold
			if pos <= 2:
				tpop[0].branching_factor, tpop2[0].branching_factor = tpop2[0].branching_factor, tpop[0].branching_factor
			if pos <= 1:
				tpop[0].compute_labels, tpop2[0].compute_labels = tpop2[0].compute_labels, tpop[0].compute_labels
			if pos == 0:
				tpop[0].copy, tpop2[0].copy = tpop2[0].copy, tpop[0].copy

			return tpop, tpop2
