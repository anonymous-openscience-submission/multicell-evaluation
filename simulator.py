import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pprint import pprint
from topos import *


class NoisyDecomposition(ABC):
	def __init__(self, std_noises, max_vars=None):
		self._std_noises = std_noises
		self._min_var = np.multiply(std_noises, std_noises)

		delta = 0
		for i,std in enumerate(self._std_noises):
			for j,std2 in enumerate(self._std_noises):
				if i != j:
					delta += std * std2

		v_minus = np.linalg.norm(std_noises) ** 2
		v_plus = (np.sqrt(v_minus) + 2 * np.sqrt(delta)) ** 2
		
		self._max_var = np.array([v_plus / len(std_noises)] * len(std_noises))

	def noisy_sample(self, x):
		sample = self.sample(x)
		# Add some noise
		noise_vector = np.array([np.random.normal(0, std) for std in self._std_noises])
		return sample + noise_vector

	def variance_lower_bound(self):
		return self._min_var
	
	def variance_upper_bound(self):
		return self._max_var

	@abstractmethod
	def getConfiguration(self):
		...

	@abstractmethod
	def getRandomConfiguration(self):
		...

	@abstractmethod
	def factor_variable_matrix(self):
		...

	@abstractmethod
	def sample(self, x):
		...

	@abstractmethod
	def getIntervals(self):
		...

class Simulator(NoisyDecomposition):
	def __init__(self, topo, alpha, noises):
		"""Constructor of the simulator

		Args:
				topo (AbstractTopo): the topology to simulate
		"""
		super().__init__(noises)
		self._topo = topo
		self._alpha = alpha

	def draw(self, showAdjacencies=[], showConflicts=[], ax=None, ax_sinr=None, ax_heatmap=None, ax_heatmap_card=None, fig=None, title=None, ylabel=None, rectangle=None):
		"""Draw the simulated environment on a given Matplotlib axis

		Args:
				showAdjacencies (list, optional): the adjacencies to show, as a triplet (int, int, str (color)). Defaults to [].
				showConflicts (list, optional): the conflicts to show, as a quadruplet:
				(int, int, str ("inner" or "outter"), str (color)). Defaults to [].
		"""

		ax_is_none = ax is None and ax_sinr is None and ax_heatmap is None and ax_heatmap_card is None
		if ax_is_none:
			fig, axes = plt.subplots(nrows=2, ncols=2)
			ax = axes[0][0]
			ax_sinr = axes[0][1]
			ax_heatmap = axes[1][0]
			ax_heatmap_card = axes[1][1]

		if ax is not None:
			ax.clear()
			if title is None:
				ax.set_title("Cellular Network")
			else:
				ax.set_title(title)

			self._topo.draw(ax)
			self._topo.draw(ax, usersOnly=True)

			for x,y,c in showAdjacencies:
				self._topo.showAdjacency(x, y, ax, c)

			for x,y,a,c in showConflicts:
				self._topo.showConflicts(x, y, a, ax, c)

			if rectangle is None:
				ax.set_xlim([0, self._topo._w])
				ax.set_ylim([0, self._topo._h])
			else:
				ax.set_xlim(rectangle[0])
				ax.set_ylim(rectangle[1])

			if ylabel is not None:
				ax.set_ylabel(ylabel)

		if ax_sinr is not None:
			self.sinrHeatmap(ax=ax_sinr, fig=fig)
			ax_sinr.set_title("Users SINR")
		if ax_heatmap is not None:
			self.capacityHeatmap(ax=ax_heatmap, fig=fig)
			ax_heatmap.set_title("Users Capacity (without Resource Sharing)")
		if ax_heatmap_card is not None:
			ax_heatmap_card.set_title("Users Capacity (with Resource Sharing)")
			self.capacityHeatmap(ax=ax_heatmap_card, fig=fig, card=True)
		if ax_is_none:
			plt.show()

	def getSummary(self):
		return self._topo.getSummary()

	def getConflicts(self):
		return np.copy(self._topo._conflicts)

	def getAdjacencies(self):
		return np.copy(self._topo._adjacency).astype(int)

	def getDimWiseAdjacencies(self):
		adj = self._topo._adjacency + np.eye(self._topo._n)
		dims = self.getConfigurationBounds()
		idxs = []
		counter = 0
		for dagent in dims:
			didx = []
			for _ in dagent:
				didx.append(counter)
				counter += 1
			idxs.append(didx)

		n_dims = idxs[-1][-1] + 1
		adj_dims = np.zeros((self._topo._n, n_dims))
		for i,ci in enumerate(adj):
			for j,cij in enumerate(ci):
				if cij == 1:
					for j_dim in idxs[j]:
						adj_dims[i][j_dim] = 1
		
		return adj_dims

	def setGlobalConfiguration(self, config):
		self._topo.setGlobalConfiguration(config)

	def getGlobalConfiguration(self):
		return self._topo.getGlobalConfiguration()

	def getConfiguration(self):
		return self.getGlobalConfiguration()

	def getRandomConfiguration(self):
		return None

	def factor_variable_matrix(self):
		return self.getDimWiseAdjacencies().astype(int)

	def sample(self):
		return self.simulate()

	def getIntervals(self):
		return self.getConfigurationBounds()

	def getConfigurationBounds(self):
		return self._topo.getConfigurationBounds()

	def sinrHeatmap(self, ax=None, fig=None):
		self._topo.startTransmissions()
		sinrs = self._topo.getEndUsersSINRs()
		sinr_values = []
		for i,cell_sinrs in enumerate(sinrs):
			for j,values in enumerate(cell_sinrs):
				sinr_values.append([values[0], values[1], values[2]])

		sinr_values = np.array(sinr_values)
		sinr_max = np.max(sinr_values[:, 2])
		sinr_min = np.min(sinr_values[:, 2])
		cmap = 'viridis'
		norm = LogNorm(vmin=sinr_min, vmax=sinr_max)

		ax.scatter(sinr_values[:, 0], sinr_values[:, 1], c=sinr_values[:, 2], s=11, cmap=cmap, norm=norm)
		
		if fig is not None:
			fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
			
		ax.set_xlim([0, self._topo._w])
		ax.set_ylim([0, self._topo._h])

	def capacityHeatmap(self, ax=None, fig=None, card=False, vmin=None, vmax=None):
		self._topo.startTransmissions()
		sinrs = self._topo.getEndUsersSINRs()
		cards = self._topo.getCardinalities()
		capacities = []
		bws = self._topo.getBandwidths()
		for i,bw in enumerate(bws):
			card_value = cards[i] if card else 1.0
			for j,values in enumerate(sinrs[i]):
				capacities.append([values[0], values[1], bw * np.log2(1.0 + values[2]) / card_value])

		capacities = np.array(capacities)
		cmap = 'viridis'
		cap_max = np.max(capacities[:, 2]) if vmax is None else vmax
		cap_min = np.min(capacities[:, 2]) if vmin is None else vmin
		norm = LogNorm(vmin=cap_min, vmax=cap_max)

		ax.scatter(capacities[:, 0], capacities[:, 1], c=capacities[:, 2], s=11, cmap=cmap, norm=norm)
		
		if fig is not None:
			fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
			
		ax.set_xlim([0, self._topo._w])
		ax.set_ylim([0, self._topo._h])

	def basicUserMetric(self, scheduled=False, zone=None):
		self._topo.startTransmissions()
		bandwidths = self._topo.getBandwidths()
		areas = self._topo.getCardinalities()
		if scheduled:
			area_capacities = []
			for i,s in enumerate(self._topo.getEndUsersSINRs()):
				capacities_unscheduled = []
				for v in s:
					capacities_unscheduled.append(bandwidths[i] * np.log2(1.0 + v[2]) if areas[i] > 0 else 0)
				schedule = scheduler(capacities_unscheduled, self._alpha)
				capacities_scheduled = []
				for sch,v in zip(schedule, capacities_unscheduled):
					capacities_scheduled.append(sch * v)
				area_capacities.append(capacities_scheduled)
		else:
			area_capacities = [[bandwidths[i] * np.log2(1.0 + v[2]) if areas[i] > 0 else 0 for v in s] for i,s in enumerate(self._topo.getEndUsersSINRs())]

		cell_aggregation = []
		if self._topo._params._many_areas:
			for i in range(0, len(areas), 2):
				cell_aggregation.append([(areas[i] if self._topo._denormalize else 1.0, area_capacities[i]), (areas[i+1] if self._topo._denormalize else 1.0, area_capacities[i+1])])
		else:
			for i in range(0, len(areas)):
				cell_aggregation.append([(areas[i] if self._topo._denormalize else 1.0, area_capacities[i])])

		# Filter out the cells outside of zone
		if zone is not None:
			cell_positions = self._topo.BSPositions()
			filtered_data = []
			for cell_position,metrics in zip(cell_positions, cell_aggregation):
				if cell_position[0] >= zone[0][0] and cell_position[0] <= zone[0][1] and cell_position[1] >= zone[1][0] and cell_position[1] <= zone[1][1]:
					filtered_data.append(metrics)

			cell_aggregation = filtered_data

		return cell_aggregation
		
	def simulate(self):# , specific_cells=None):
		"""Simulate the communications and extract relevant performance measures
		"""
		metric = self.basicUserMetric()
		# print(metric)
		dict_constant = {
			0: 1e3,
			0.25: 1e4,
			0.5: 1e4,
			0.75: 1e5,
			1: 1e6,
			2: 1e7,
			3: 1e9
		} if self._topo._denormalize else {
			0: 1e1,
			0.25: 1e2,
			0.5: 1e2,
			0.75: 1e3,
			1: 1e3,
			2: 1e5,
			3: 1e7
		}
		return np.array([sum([w * scheduled_alpha_fairness(v, self._alpha) for w,v in m]) for m in metric]) / dict_constant[self._alpha]