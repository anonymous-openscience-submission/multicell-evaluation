from cells import *
import networkx as nx
import matplotlib.pyplot as plt
from tools import *
import json
from multiprocessing import Pool
from shapely.geometry import Point
import shapely.geometry.polygon as shapely_poly
from abc import ABC, abstractmethod

class TopoParams:
	def __init__(self, noma, ffr, ffr_threshold, full_reuse, beamforming):
		self._noma = noma
		self._ffr = ffr
		self._ffr_threshold = ffr_threshold
		self._full_reuse = full_reuse
		self._many_areas = self._noma or self._ffr
		self._beamforming = beamforming

class AbstractTopo(ABC):
	def __init__(self, discrete):
		self._colormap = ['#00CDCB', '#FAEE00', '#FF0053', '#F76B00']
		self._cells = []
		self._adjacency = [[]]
		self._conflicts = [[]]
		self._discrete = discrete
		self._n_colors = 0

	def getSummary(self):
		summary = "== TOPO SUMMARY ==\n"
		for cell in self._cells:
			summary_cell = cell.getSummary()
			for line in summary_cell:
				summary += "\t" + line + "\n"

		return summary
		
	def computeConflicts(self, params):
		"""Compute the conflicts between the areas in the cells on an adjacency basis
		"""
		if params._many_areas:
			# 2N x 2N matrix if noma, since there is N cells and 2 areas per cell
			self._conflicts = [[0 for _ in range(2 * len(self._cells))] for _ in range(2 * len(self._cells))]
			for idxs,cell_adjacencies in enumerate(self._adjacency):
				idxs_inner_area = 2 * idxs
				idxs_outter_area = 2 * idxs + 1

				# The outter area of a cell is in conflict with the inner area of the same cell (not the other way around!)
				if params._noma:
					self._conflicts[idxs_outter_area][idxs_inner_area] = 1
				for idxt,adj_value in enumerate(cell_adjacencies):
					# Removing the adjacency rule
					if idxt != idxs:# adj_value != 0:
						idxt_inner_area = 2 * idxt
						idxt_outter_area = 2 * idxt + 1

						# The inner area is in conflict with the inner and outter area of the other cell
						self._conflicts[idxs_inner_area][idxt_inner_area] = 1
						if params._noma:
							self._conflicts[idxs_inner_area][idxt_outter_area] = 1
						# The outter area is in conflict with the inner area and the outter if it has the same color
						if params._noma:
							self._conflicts[idxs_outter_area][idxt_inner_area] = 1
						self._conflicts[idxs_outter_area][idxt_outter_area] = 1 if self._cells[idxs]._outter._color == self._cells[idxt]._outter._color else 0
		else:
			# N x N matrix if not noma, since there is N cells
			self._conflicts = [[0 for _ in range(len(self._cells))] for _ in range(len(self._cells))]
			for idxs_outter_area,cell_adjacencies in enumerate(self._adjacency):
				for idxt_outter_area,adj_value in enumerate(cell_adjacencies):
					# Removing the adjacency rule
					if idxs_outter_area != idxt_outter_area:# adj_value != 0:
						if params._full_reuse:
							self._conflicts[idxs_outter_area][idxt_outter_area] = 1
						else:
							# The outter area is in conflict with the inner area and the outter if it has the same color
							self._conflicts[idxs_outter_area][idxt_outter_area] = 1 if self._cells[idxs_outter_area]._outter._color == self._cells[idxt_outter_area]._outter._color else 0

	def showAdjacency(self, sidx, ax, color="black"):
		"""Draw the adjacencies of a cell on a given Matplotlib axis

		Args:
				sidx (int): the idx of the cell
				ax (axis): the Matplotlib axis to draw on
				color (str, optional): the color of the shown adjacencies. Defaults to "black".
		"""
		adjacencies = self._adjacency[sidx]
		for tidx, adj in enumerate(adjacencies):
			if adj == 1:
				xx = [self._cells[sidx]._x, self._cells[tidx]._x]
				yy = [self._cells[sidx]._y, self._cells[tidx]._y]
				ax.plot(xx, yy, c=color)

	def computeColoring(self):
		"""Compute the coloring of the tiling, such that no couple of adjacent cells have the same color
		"""
		graph = nx.from_numpy_matrix(np.array(self._adjacency))
		colors = nx.greedy_color(graph, interchange=True)
		
		# print(colors)
		unique_values = np.unique(list(colors.values()))
		self._n_colors = len(unique_values)
		for k,v in colors.items():
			self._cells[k].setColor(self._colormap[v], len(unique_values))

	def draw(self, ax, usersOnly=False):
		"""Draw the grid on a given axis

		Args:
				ax (axis): the Matplotlib axis to draw on
		"""
		for cell in self._cells:
			cell.draw(ax, usersOnly)

	def cellIndexFromAreaIndex(self, aidx, params):
		"""Given an area index, compute the index of the cell it belongs to

		Args:
				aidx (int): the area index

		Returns:
				int: the cell index the area belongs to
		"""
		if params._many_areas:
			return int(np.ceil(aidx / 2) - aidx % 2)
		else:
			return aidx

	def startTransmissions(self):
		"""Simulate transmissions and compute relevant metrics
		"""
		for idxarea_source,conflicts in enumerate(self._conflicts):
			idxcell_source = self.cellIndexFromAreaIndex(idxarea_source, self._params)
			area_source = self._cells[idxcell_source]._inner if self._params._many_areas and idxarea_source % 2 == 0 else self._cells[idxcell_source]._outter
			emitters = []

			for idxarea_target,conflict in enumerate(conflicts):
				# Conflict between two areas, must be taken into account for the SINR
				if conflict != 0:
					idxcell_target = self.cellIndexFromAreaIndex(idxarea_target, self._params)
					area_target = self._cells[idxcell_target]._inner if self._params._many_areas and idxarea_target % 2 == 0 else self._cells[idxcell_target]._outter
					emitters.append((area_target._base_station._tx_dBm, area_target._base_station._location, area_target.getBandwidth()))

			for user in area_source._users:
				user.computeReceptionMetrics(logPathLossSINR((area_source._base_station._tx_dBm, area_source._base_station._location, area_source.getBandwidth()), user._location, idxarea_source % 2 == 0, emitters, beamforming=self._params._beamforming))

	def getEndUsersSINRs(self):
		n = len(self._conflicts)
		sinrs = []
		for idxarea_source in range(n):
			area_sinrs = []
			idxcell_source = self.cellIndexFromAreaIndex(idxarea_source, self._params)
			area_source = self._cells[idxcell_source]._inner if self._params._many_areas and idxarea_source % 2 == 0 else self._cells[idxcell_source]._outter

			for user in area_source._users:
				area_sinrs.append([user._location[0], user._location[1], user._sinr])
			
			sinrs.append(area_sinrs)
		
		return sinrs

	def getBandwidths(self):
		bws = []

		for idxarea_source in range(len(self._conflicts)):
			idxcell_source = self.cellIndexFromAreaIndex(idxarea_source, self._params)
			inner = self._params._many_areas and idxarea_source % 2 == 0
			area_source = self._cells[idxcell_source]._inner if inner else self._cells[idxcell_source]._outter
			
			bws.append(area_source.getBandwidth())

		return np.array(bws)

	def getCardinalities(self):
		areas = []

		for idxarea_source in range(len(self._conflicts)):
			idxcell_source = self.cellIndexFromAreaIndex(idxarea_source, self._params)
			inner = idxarea_source % 2 == 0 and self._params._many_areas
			area_source = self._cells[idxcell_source]._inner if inner else self._cells[idxcell_source]._outter
			
			areas.append(area_source.getCardinality(self._discrete))

		return np.array(areas)

	def formatConfiguration(self, config):
		# 1-d dimensional into 2-d dimensional
		counter = 0
		bounds = self.getConfigurationBounds()
		formatted = []
		for agent in bounds:
			agent_conf = []
			for _ in agent:
				agent_conf.append(config[counter])
				counter += 1
			formatted.append(agent_conf)

		return formatted

	def setGlobalConfiguration(self, config):
		# Scale the configuration to the maximal value
		np_config = np.array(config)
		pmax = np.max(np_config[:, 0])
		scaling = 0 # MAX_DBM - pmax
		np_config[:, 0] += scaling

		for idx_cell in range(len(self._adjacency)):
			self._cells[idx_cell].setConfiguration(*np_config[idx_cell])

		self.postConfigurationUpdate()

	def getGlobalConfiguration(self):
		conf = []
		for c in self._cells:
			conf.append(c.getConfiguration())

		return conf

	def getConfigurationBounds(self):
		bounds = []
		for c in self._cells:
			bounds.append(c.getBounds())

		return bounds

	@abstractmethod
	def adjacencyMatrix(self):
		...

	@abstractmethod
	def postConfigurationUpdate(self):
		...

class VoronoiTopo(AbstractTopo):
	def __init__(self, width, height, n_bs, discrete, sampler="DPP", cells=None, adjacencies=None):
		super().__init__(discrete)

		self._w = width
		self._h = height
		self._n = n_bs
		self._bounding_box = np.array([0, self._w, 0, self._h])
		self._cells = []

		if cells is None:
			if sampler == "PPP":
				bs_x = np.random.uniform(0, width, self._n)
				bs_y = np.random.uniform(0, height, self._n)
				bs_locations = np.array([[x, y] for x,y in zip(bs_x, bs_y)])
			elif sampler == "DPP":
				bs_locations = dpp_sampler(self._n, np.array([[0, 0], [self._w, 0], [self._w, self._h], [0, self._h]]))

			self._vor = voronoi(bs_locations, self._bounding_box)
			for idx_point,p in enumerate(self._vor.filtered_points):
				shape_list = []
				for idx in self._vor.filtered_regions[self._vor.filtered_point_region[idx_point]]:
					shape_list.append(list(self._vor.vertices[idx]))

				self._cells.append(PolygonCell(p[0], p[1], shape_list))
		else:
			self._cells = []
			for cell in cells:
				self._cells.append(PolygonCell(cell["x"], cell["y"], cell["shape"], users=cell['users']))
		
		if adjacencies is None:
			self.adjacencyMatrix()
		else:
			self._adjacency = adjacencies
		self.computeColoring()
		self.computeConflicts()

	def adjacencyMatrix(self):
		self._adjacency = self._vor.filtered_adjacencies

	def postConfigurationUpdate(self):
		pass

	@staticmethod
	def fromFile(path):
		file = open(path)
		topo = json.load(file)

		return VoronoiTopo(topo['width'], topo['height'], topo['n_bs'], cells=topo['cells'], adjacencies=topo['adjacencies'])

	def saveToFile(self, path):
		file = open(path, 'w')
		topo_json = {
			"cells": [],
			"adjacencies": self._adjacency,
			"width": self._w,
			"height": self._h,
			"n_bs": self._n
		}

		for cell in self._cells:
			dict_cell = {
				"x": cell._x,
				"y": cell._y,
				"shape": cell._shape,
				"users": []
			}
			for user in cell._users:
				dict_user = {
					"x": user._location[0],
					"y": user._location[1]
				}

				dict_cell["users"].append(dict_user)

			topo_json["cells"].append(dict_cell)

		json.dump(topo_json, file)
		file.close()
			

	def draw(self, ax, usersOnly=False):
		for cell in self._cells:
			cell.draw(ax, usersOnly=usersOnly)

class DynamicTopology(AbstractTopo):
	def __init__(self, width, height, n_bs, alpha, discrete, sampler="DPP", cells=None, users=None, dxy=30, assoc_criterion="sinr_ext", denormalize=False, params=None, users_dist="UNIFORM"):
		super().__init__(discrete)
		self._w = width
		self._h = height
		self._n = n_bs
		self._dxy = dxy
		self._denormalize = denormalize
		self._params = params
		self._alpha = alpha
		self._bounding_box = np.array([0, self._w, 0, self._h])
		bs_locations = []
		self._cells = []
		self._candidates_for_cells = []
		self._assoc_criterion = assoc_criterion
		
		if cells is None:
			if sampler == "PPP":
				bs_x = np.random.uniform(0, width, self._n)
				bs_y = np.random.uniform(0, height, self._n)
				bs_locations = np.array([[x, y] for x,y in zip(bs_x, bs_y)])
			elif sampler == "DPP":
				bs_locations = dpp_sampler(self._n, np.array([[0, 0], [self._w, 0], [self._w, self._h], [0, self._h]]))
		else:
			for cell in cells:
				bs_locations.append([cell['x'], cell['y']])
			bs_locations = np.array(bs_locations)

		self._vor = voronoi(bs_locations, self._bounding_box)
		self._initial_shapes = []
		for idx_point,p in enumerate(self._vor.filtered_points):
			shape_list = []
			for idx in self._vor.filtered_regions[self._vor.filtered_point_region[idx_point]]:
				shape_list.append(list(self._vor.vertices[idx]))
			self._initial_shapes.append(shapely_poly.Polygon(shape_list))
			self._cells.append(DynamicCell(p[0], p[1], self._dxy, shape_list, idx_point + 1, params=self._params))

		self.adjacencyMatrix()
		self.computeColoring()
		self.computeConflicts(self._params)

		self._area_bandwidths = []
		for c in self._cells:
			if self._params._many_areas:
				self._area_bandwidths.append(c._inner.getBandwidth())
			self._area_bandwidths.append(c._outter.getBandwidth())

		if users is not None:
			self._users = [CellularObject(x, y, 0) for x,y in users]
		elif users_dist == "UNIFORM":
			self._users = [CellularObject(x, y, 0) for x in range(1, self._w, self._dxy) for y in range(1, self._h, self._dxy)]

		self._n_users = len(self._users)
		print(self._n_users / (self._w / 1e3 * self._h / 1e3))

		self.updateCellUsers()

	def BSPositions(self):
		data = []
		for cell in self._cells:
			data.append([cell._x, cell._y])
		return data
		
	def adjacencyMatrix(self):
		self._adjacency = self._vor.filtered_adjacencies

	def postConfigurationUpdate(self):
		self.updateCellUsers()

	@staticmethod
	def chooseCell(user, candidates_for_cells, initial_shapes, emitters_for_areas, cells, assoc_criterion, params):
		point = Point(user._location[0], user._location[1])
		for idx_cell,shape in enumerate(initial_shapes):
			if shape.intersects(point):
				# Candidates are all the areas in the interference list
				candidates = candidates_for_cells[idx_cell]
				# print("\tMy candidates:")

				# Find the best candidate for this user
				best_candidate = None
				best_criterion = 0
				best_cell_ext_sinr = None
				best_cell_in_sinr = None
				best_cell_in_snr = None
				for candidate in candidates:
					# The SINR for the user can be computed
					if params._many_areas:
						sinr = None
						# Inner
						sinr = logPathLossSINR((cells[candidate]._inner._base_station._tx_dBm, cells[candidate]._inner._base_station._location, cells[candidate]._inner.getBandwidth()),  user._location, True, emitters_for_areas[2 * candidate], beamforming=params._beamforming)

						user.computeReceptionMetrics(sinr)
						in_sinr = user._sinr

						# Inner SNR
						in_snr = logPathLossSNR((cells[candidate]._inner._base_station._tx_dBm, cells[candidate]._inner._base_station._location, cells[candidate]._inner.getBandwidth()),  user._location, beamforming=params._beamforming)

						# Outter
						sinr = logPathLossSINR((cells[candidate]._outter._base_station._tx_dBm, cells[candidate]._outter._base_station._location, cells[candidate]._outter.getBandwidth()), user._location, False, emitters_for_areas[2 * candidate + 1], beamforming=params._beamforming)
						user.computeReceptionMetrics(sinr)
						ext_sinr = user._sinr
					else:
						user.computeReceptionMetrics(logPathLossSINR((cells[candidate]._outter._base_station._tx_dBm, cells[candidate]._outter._base_station._location, cells[candidate]._outter.getBandwidth()), user._location, False, emitters_for_areas[candidate], beamforming=params._beamforming))
						ext_sinr = user._sinr

					# Compute the criterion
					criterion = None
					if assoc_criterion == "sinr_ext":
						criterion = ext_sinr
					elif assoc_criterion == "sinr_in":
						criterion = in_sinr
					elif assoc_criterion == "sinr_max":
						criterion = max(ext_sinr, in_sinr) if params._many_areas else ext_sinr
					# print(f"\t\tSINR: {user._sinr}")
					if best_candidate is None or best_criterion < criterion:
						best_candidate = candidate
						best_criterion = criterion
						# Update the SINRs
						best_cell_ext_sinr = ext_sinr
						if params._many_areas:
							best_cell_in_sinr = in_sinr
							best_cell_in_snr = in_snr

				return best_candidate, best_cell_ext_sinr, best_cell_in_sinr, best_cell_in_snr

		print("Error, we should not go there:", point)
		return None, None, None


	def updateCellUsers(self):
		cell_users = [[] for _ in range(self._n)]
		cell_ext_sinrs = [[] for _ in range(self._n)]
		cell_in_sinrs = [[] for _ in range(self._n)]
		cell_in_snrs = [[] for _ in range(self._n)]

		# All the candidates when the point belongs to a Voronoi cell
		if self._candidates_for_cells == []:
			for idx_cell in range(self._n):
				# Candidates are all the cells in the interference list
				candidates = []
				for idx_adj,adj in enumerate(self._adjacency[idx_cell]):
					if True: # adj > 0:
						candidates.append(idx_adj)
				self._candidates_for_cells.append(candidates)

		# All the conflict emitters are the exterior emitters of all adjacent cells
		emitters_for_areas = []
		for idx_area in range(2 * self._n if self._params._many_areas else self._n):
			emitters = []
			for idx_area_conflict,conflict in enumerate(self._conflicts[idx_area]):
				if conflict > 0:
					idx_cell_conflict = self.cellIndexFromAreaIndex(idx_area_conflict, self._params)
					area_conflict = self._cells[idx_cell_conflict]._inner if idx_area_conflict % 2 == 0 and self._params._many_areas else self._cells[idx_cell_conflict]._outter
					emitters.append((area_conflict._base_station._tx_dBm, area_conflict._base_station._location, area_conflict.getBandwidth()))
			emitters_for_areas.append(emitters)

		args = []
		for user in self._users:
			args.append((user, self._candidates_for_cells, self._initial_shapes, emitters_for_areas, self._cells, self._assoc_criterion, self._params))
			
		with Pool() as pool:
			associations = pool.starmap(DynamicTopology.chooseCell, args)

		for i,(best_candidate,ext_sinr,in_sinr,in_snr) in enumerate(associations):
			if best_candidate is not None:
				cell_users[best_candidate].append(self._users[i])
				cell_ext_sinrs[best_candidate].append(ext_sinr)
				cell_in_sinrs[best_candidate].append(in_sinr)
				cell_in_snrs[best_candidate].append(in_snr)

		for cell_idx in range(self._n):
			if self._params._many_areas:
				self._cells[cell_idx].setUsersManyAreas(cell_users[cell_idx], cell_ext_sinrs[cell_idx], cell_in_sinrs[cell_idx], cell_in_snrs[cell_idx], self._cells[cell_idx]._inner.getBandwidth(), self._cells[cell_idx]._outter.getBandwidth(), self._alpha, self._discrete, self._denormalize)
			else:
				self._cells[cell_idx].setUsersWithoutNOMA(cell_users[cell_idx])

	def saveToFile(self, path):
		file = open(path, 'w')
		topo_json = {
			"cells": [],
			"width": self._w,
			"height": self._h,
			"n_bs": self._n
		}

		for cell in self._cells:
			dict_cell = {
				"x": cell._x,
				"y": cell._y,
			}
			topo_json["cells"].append(dict_cell)

		json.dump(topo_json, file)
		file.close()

	@staticmethod
	def fromFile(path, alpha, discrete, assoc_criterion="sinr_ext", denormalize=False, params=None):
		file = open(path)
		topo = json.load(file)

		users = None
		if 'users' in topo.keys():
			users = topo['users']

		return DynamicTopology(topo['width'], topo['height'], topo['n_bs'], alpha, discrete, cells=topo['cells'], users=users, assoc_criterion=assoc_criterion, denormalize=denormalize, params=params)