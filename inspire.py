import numpy as np
from skopt import Optimizer
from scipy.stats import invweibull
from skopt.learning.gaussian_process import kernels
from skopt.learning.gaussian_process.gpr import GaussianProcessRegressor
import matplotlib.pyplot as plt
from operator import itemgetter
from tools import flatten

def minCapacities(sinrs, bandwidths, areas=None):
	min_sinrs = np.array([np.min(s) for s in sinrs])
	capacities = np.multiply(bandwidths, np.log(1 + min_sinrs))

	return capacities / areas if areas is not None else capacities

def areaCapacities(integrals, bandwidths, areas=None):
	capacities = np.multiply(bandwidths, integrals)

	return capacities / areas if areas is not None else capacities

def globalReward(individualRewards):
	return np.sum(individualRewards)

class Agent:
	def __init__(self, dimensions, nu=1.5):
		self._dimensions = dimensions
		self._xx = None
		self._yy = None
		self._lip_estimate_l1 = None
		self._lip_estimate_l2 = None
		self._n_obs = 0
		self._models = []

		self._kernel = kernels.Matern(nu=nu, length_scale_bounds=(1e-9, 1e9)) + kernels.WhiteKernel(noise_level_bounds=(1e-9, 1e9))
		self._opt = Optimizer(self._dimensions, base_estimator=GaussianProcessRegressor(self._kernel, n_restarts_optimizer=15), acq_func="EI", acq_optimizer="lbfgs", n_initial_points=10)

	def normalize(self, x):
		normalized = []
		for i,xi in enumerate(x):
			normalized.append((xi - self._dimensions[i][0]) / (self._dimensions[i][1] - self._dimensions[i][0]))

		return np.array(normalized)
		
	def predict(self):
		return self._opt.ask()

	def tell(self, x, y):
		self._opt.tell(x, y)
		np_x = np.array(x)
		np_y = np.array([y])

		# if self._opt.models != []:
		# 	print(self._opt.models[-1].kernel_)
		if self._xx is None:
			self._xx = np.array([np_x])
			self._yy = np.array([np_y])
		else:
			self._xx = np.append(self._xx, [np_x], axis=0)
			self._yy = np.append(self._yy, [np_y], axis=0)

		self._n_obs += 1

		# Add slopes to the slope objects
		slopes_l1 = []
		slopes_l2 = []
		x_normalized = self.normalize(x)
		for i in range(self._n_obs - 1):
			xx_normalized = self.normalize(self._xx[i])
			dx_l1 = np.linalg.norm(x_normalized - xx_normalized, 1)
			dx_l2 = np.linalg.norm(x_normalized - xx_normalized, 2)
			if dx_l1 != 0:
				dy = np.abs(y - self._yy[i, 0])
				slopes_l1.append(dy / dx_l1)
				slopes_l2.append(dy / dx_l2)

		if len(slopes_l1) > 0:
			m_l1 = max(slopes_l1) if max(slopes_l1) > 0 else 1e-3
			m_l2 = max(slopes_l2) if max(slopes_l2)  > 0 else 1e-3
			self._lip_estimate_l1 = m_l1 if self._lip_estimate_l1 is None else max(m_l1, self._lip_estimate_l1)
			self._lip_estimate_l2 = m_l2 if self._lip_estimate_l2 is None else max(m_l2, self._lip_estimate_l2)

class Inspire:
	def __init__(self, h_dimensions, conflicts, nu=1.5):
		self._agents = []
		self._h_dimensions = h_dimensions
		self._agent_indices = []
		self._dim_indices = []
		self._agent_dim_indices = []
		self._n_dim = 0

		# Each area has an agent
		for idx_source,area_conflicts in enumerate(conflicts):
			prescription_location = []
			# Extract the conflict of the area
			conflicts[idx_source][idx_source] = 1.0
			area_conflicts[idx_source] = 1.0

			prescription_location = np.where(np.array(area_conflicts) != 0)[0]

			# These are the areas in conflict with the current area
			self._agent_indices.append(prescription_location)

			# Extract the correct dimensions
			agent_dim_indices = []
			dimensions = []
			idx_dim = 0
			for idx,dims in enumerate(h_dimensions):
				dim_indices = []
				for dim in dims:
					if idx in prescription_location:
						dimensions.append(dim)
						agent_dim_indices.append(idx_dim)

					dim_indices.append(idx_dim)
					idx_dim += 1
				self._dim_indices.append(dim_indices)

			# Number of dimensions
			if self._n_dim == 0:
				self._n_dim = idx_dim

			# Indices of dimensions managed by the agent
			self._agent_dim_indices.append(agent_dim_indices)
			self._agents.append(Agent(dimensions, nu=nu))

		self._conflicts = conflicts

	def homemade_median(self, sample, weights=None):
		if weights is None:
			weights = [1 for s in sample]

		sorted_lists = [list(x) for x in zip(*sorted(zip(sample, weights), key=itemgetter(0)))]
		sorted_sample = sorted_lists[0]
		sorted_weights = sorted_lists[1]

		w_cumsum = [sum([w for w in sorted_weights[:i+1]]) for i in range(len(sorted_weights))]
		cutoff = w_cumsum[-1] / 2.0

		for i,s in enumerate(sorted_sample):
			if w_cumsum[i] >= cutoff:
				return s

		print("ERROR:")
		print("Weights:", sorted_weights)
		print("Cumsum:", w_cumsum)
		print("Cutoff:", cutoff)

		return None

	def consensus(self, sample, order, involved_agents=None):
		if order == 2:
			# Estimation of the Lipschitz constant of the agent rewards
			l_constants = [self._agents[agent_idx]._lip_estimate_l2 for agent_idx in involved_agents]
			l_constants = [l if l is not None else 1 for l in l_constants]
			return np.average(sample, weights=l_constants)
		if order == 1:
			# Estimation of the Lipschitz constant of the agent rewards
			l_constants = [self._agents[agent_idx]._lip_estimate_l1 for agent_idx in involved_agents]
			l_constants = [l if l is not None else 1 for l in l_constants]
			return self.homemade_median(sample, weights=l_constants)

		print("ERROR:")
		print("Method:", order)

		return None

	def log(self, s, log):
		if log:
			print(s)

	def predict(self, show_log=False):
		# Gather the prescriptions
		prescriptions = []
		for agent in self._agents:
			prescriptions.append(agent.predict())

		np.set_printoptions(linewidth=np.inf)

		print("Prescriptions:")
		summary = np.zeros((len(self._agents), self._n_dim))
		summary -= 1
		for idx_agent,_ in enumerate(self._agents):
			for idx_for_agent,idx_dim in enumerate(self._agent_dim_indices[idx_agent]):
				summary[idx_agent][idx_dim] = prescriptions[idx_agent][idx_for_agent]
		print(summary)

		print("Normalized weights for dimensions:")
		summary = np.zeros((len(self._agents), self._n_dim))
		l_constants = [agent._lip_estimate_l1 for agent in self._agents]
		l_constants = [l if l is not None else 1 for l in l_constants]
		for idx_agent,_ in enumerate(self._agents):
			for idx_for_agent,idx_dim in enumerate(self._agent_dim_indices[idx_agent]):
				summary[idx_agent][idx_dim] = l_constants[idx_agent]
		print(summary / np.sum(summary, axis=0))

		# Compute the consensus for each dimension
		final_configuration_l1 = []
		final_configuration_l2 = []

		# For each agent
		to_compare_l1 = [[] for _ in range(len(self._agents))]
		to_compare_l2 = [[] for _ in range(len(self._agents))]
		f_dims = flatten(self._h_dimensions)
		for idx_agent,_ in enumerate(self._agents):
			focus_indices = self._dim_indices[idx_agent]
			config_agent_l1 = []
			config_agent_l2 = []

			self.log(f"Agent {idx_agent} is responsible for dimensions {focus_indices}", show_log)
			# For each dimension governed by the agent
			for focus_idx in focus_indices:
				self.log(f"\tDimension {focus_idx}:", show_log)
				sample = []
				# For each prescriptor for this dimension
				for idx_target in self._agent_indices[idx_agent]:
					# Retrieve the value of this dimension in the prescription
					idx_target_dim = self._agent_dim_indices[idx_target].index(focus_idx)
					sample.append(prescriptions[idx_target][idx_target_dim])

					self.log(f"\t\tAgent {idx_target} also prescribes for this dimension at index {idx_target_dim} with value {prescriptions[idx_target][idx_target_dim]}", show_log)

				# Once all values are retrieved, compute the "median"
				bounds = f_dims[focus_idx]
				config_agent_l1.append(min(bounds[1], max(bounds[0], self.consensus(sample, 1, involved_agents=self._agent_indices[idx_agent]))))
				config_agent_l2.append(min(bounds[1], max(bounds[0], self.consensus(sample, 2, involved_agents=self._agent_indices[idx_agent]))))
				self.log(f"\t\tThe sample is {sample} and the final value is {config_agent_l1[-1]}", show_log)

			# Add the consensus to the final configuration
			for idx_target in self._agent_indices[idx_agent]:
				to_compare_l1[idx_target].append(config_agent_l1)
				to_compare_l2[idx_target].append(config_agent_l2)

			final_configuration_l1.append(config_agent_l1)
			final_configuration_l2.append(config_agent_l2)

		cost_l1, cost_l2 = 0, 0
		for i,p in enumerate(prescriptions):
			cost = 1.0 if self._agents[i]._lip_estimate_l1 is None else self._agents[i]._lip_estimate_l1 * np.linalg.norm(np.array(p) - np.array(flatten(to_compare_l1[i])), 1)
			# print(f"Comparing {p} and {to_compare_l1[i]} with lipschitz = {self._agents[i]._lip_estimate_l1} : {cost}")
			cost_l1 += cost

			cost = 1.0 if self._agents[i]._lip_estimate_l2 is None else self._agents[i]._lip_estimate_l2 * np.linalg.norm(np.array(p) - np.array(flatten(to_compare_l2[i])))
			# print(f"Comparing {p} and {to_compare_l2[i]} with lipschitz = {self._agents[i]._lip_estimate_l2} : {cost}")
			cost_l2 += cost

		print(f"L1: {cost_l1}, L2: {cost_l2}")

		return final_configuration_l1 if cost_l1 < cost_l2 else final_configuration_l2

	def tell(self, configuration, selfish_rewards, show_log=False):
		indiv_rewards = []
		for idx_agent,_ in enumerate(self._agents):
			x = []
			y = 0
			# Gather all the dimensions the agent know of, compute the reward and the configuration of the neighborhood
			self.log(f"Neighbors of Agent {idx_agent}: {self._agent_indices[idx_agent]}", show_log)
			for idx_neighbor in self._agent_indices[idx_agent]:
				self.log(f"\tNeighborhood of Agent {idx_neighbor}: {len(self._agent_indices[idx_neighbor])}", show_log)
				y += selfish_rewards[idx_neighbor] / len(self._agent_indices[idx_neighbor])
				for v in configuration[idx_neighbor]:
					x.append(v)

			indiv_rewards.append(y)
			self.log(f"Agent {idx_agent}: {y}", show_log)

			self._agents[idx_agent].tell(x, -y)

		return np.array(indiv_rewards)
				
	def getAgentsRewards(self):
		indiv_rews = []
		for agent in self._agents:
			indiv_rews.append(-agent._yy[:, 0])

		return indiv_rews