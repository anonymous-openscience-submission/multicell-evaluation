from random import random
from areas import *
from tools import *
from objects import *
from scipy.spatial import ConvexHull
from abc import ABC, abstractmethod
from operator import itemgetter

MIN_DBM = 10.0
MAX_DBM = 45.0

class AbstractCell(ABC):
	def __init__(self, bs_x, bs_y, dxy, shape, color=None, cId=None):
		self._x = bs_x
		self._y = bs_y
		self._shape = shape
		self._dxy = dxy
		self._color = color
		self._n_colors = None
		self._cell_id = cId

		self._inner = None
		self._outter = None

	def setConfiguration(self, tx_in, in_rad, tx_out):
		tx_in, tx_out = l1_bounded(tx_in, tx_out, MIN_DBM, MAX_DBM)
		self._inner.setConfiguration(tx_in, in_rad)
		self._outter.setConfiguration(tx_out)

	def randomConfiguration(self):
		bounds = self.getBounds()
		betas = []
		for bound in bounds:
			beta = np.random.uniform(bound[0], bound[1])
			betas.append(beta)

		return betas

	def draw(self, ax, usersOnly=False):
		self._outter.draw(ax, usersOnly=usersOnly)
		self._inner.draw(ax, usersOnly=usersOnly)

	def setColor(self, c, n_colors):
		self._outter.setColor(c, n_colors)

	@abstractmethod
	def initializeUsers(self, users=None):
		...

	@abstractmethod
	def initializeAreas(self):
		...

	@abstractmethod
	def getBounds(self):
		...

class PolygonCell(AbstractCell):
	def __init__(self, bs_x, bs_y, shape, color=None, users=None):
		super().__init__(bs_x, bs_y, 20, shape, color=color)

		self._n_users = None if users is None else len(users)
		
		self._rad_max = self.findMaximalRadius()
		self._rad_min = self._dxy
		self.initializeUsers(users=users)
		self.initializeAreas()

	def initializeUsers(self, users=None):
		if users is None:
			poly = sg.Polygon(np.copy(np.array(self._shape)))
			self._users = []
			for x in range(int(np.ceil(poly.bounds[0])), int(np.floor(poly.bounds[2])), self._dxy):
				for y in range(int(np.ceil(poly.bounds[1])), int(np.floor(poly.bounds[3])), self._dxy):
					pnt = sg.point.Point(x, y)
					if pnt.within(poly):
						self._users.append(CellularObject(float(x), float(y), 0))
			self._n_users = len(self._users)
		else:
			self._users = [CellularObject(u["x"], u["y"], 0) for u in users]

	def initializeAreas(self):
		tx_in, tx_out = self.drawNormalizedPowers()
		self._inner = InnerArea(self._x, self._y, np.random.uniform(0.0, 1.0) * (self._rad_max - self._rad_min) + self._rad_min, self._users, self._rad_min, self._rad_max, tx_in, MIN_DBM, MAX_DBM) if self._params._many_areas else None
		self._outter = PolygonOutterArea(self._x, self._y, self._shape, self._users, self._inner, tx_out, MIN_DBM, MAX_DBM, self._color, self._params)
		# An inner area cannot have a surface greater than the polygon
		if self._params._many_areas:
			self._inner.setOutterArea(self._outter)

	def findMaximalRadius(self):
		max_dist = 0.0
		for p1,p2 in zip(self._shape, self._shape[1:] + [self._shape[0]]):
			dist = pnt2line((self._x, self._y), p1, p2)[0]

			if dist > max_dist:
				max_dist = dist
		
		return max_dist - self._dxy

	def getSummary(self):
		lines = []
		lines.append(f"== Cell ({self._x, self._y}) ==")
		lines.append("\t== Inner area ==")
		lines.append(f"\t\tInner power: {self._inner._base_station._tx_dBm} dBm")
		lines.append(f"\t\tInner radius: {self._inner._radius} meters")
		lines.append(f"\t\tUsers:")
		for usr in self._inner._users:
			lines.append(f"\t\t\t{usr._location}, SINR: {usr._sinr}, Distance to BS: {np.linalg.norm(np.array([self._x, self._y]) - np.array([*usr._location]))} meters")
		lines.append("\t== Outter area ==")
		lines.append(f"\t\tOutter power: {self._outter._base_station._tx_dBm} dBm")
		for usr in self._outter._users:
			lines.append(f"\t\t\t{usr._location}, SINR: {usr._sinr}, Distance to BS: {np.linalg.norm(np.array([self._x, self._y]) - np.array([*usr._location]))} meters")

		return lines

class DynamicCell(AbstractCell):
	def __init__(self, bs_x, bs_y, dxy, default_shape, cId, params):
		super().__init__(bs_x, bs_y, dxy, default_shape, None, cId=cId)

		self._shape = default_shape
		self._users = []
		self._n_users = None
		self._params = params

		self.initializeUsers()
		self.initializeAreas()

	def initializeUsers(self, users=None):
		pass

	def initializeAreas(self):
		if self._params._many_areas:
			self._inner = PolygonInnerArea(self._x, self._y, 0, MIN_DBM, MAX_DBM, params=self._params)
		self._outter = PolygonOutterArea(self._x, self._y, self._shape, [], self._inner, 0, MIN_DBM, MAX_DBM, params=self._params)
		random_config = self.randomConfiguration()
		self.setConfiguration(*random_config)

	def getShape(self):
		return self._outter.polygonApproximation(holes=False)

	def setConfiguration(self, total_tx, p_out=1.0):
		tx_mW = dBmtomW(total_tx)
		tx_mW_in, tx_mW_out = (1 - p_out) * tx_mW, p_out * tx_mW
		tx_in, tx_out = mWtodBm(tx_mW_in), mWtodBm(tx_mW_out)
		if self._params._many_areas:
			self._inner.setConfiguration(tx_in)
		self._outter.setConfiguration(tx_out)

	def getConfiguration(self):
		tx_mW_out = dBmtomW(self._outter._base_station._tx_dBm)
		if self._params._many_areas:
			tx_mW_in = dBmtomW(self._inner._base_station._tx_dBm)
			tx_mW_out = dBmtomW(self._outter._base_station._tx_dBm)
			tx_mW = tx_mW_in + tx_mW_out

			beta_1 = mWtodBm(tx_mW)
			beta_2 = tx_mW_out / tx_mW

			return [beta_1, beta_2]
		else:
			tx_mW_out = dBmtomW(self._outter._base_station._tx_dBm)
			beta_1 = mWtodBm(tx_mW_out)

			return [beta_1]

	def setUsersWithoutNOMA(self, users):
		self._users = users
		self._outter.updateUsers(users)
		self._n_users = len(users)

	def setUsersManyAreas(self, users, ext_sinrs, in_sinrs, in_snrs, in_bandwidth, ext_bandwidth, alpha, discrete, denormalize=False):
		# Sort the users, best SNR! to worst
		users = [u for _, u in sorted(zip(in_snrs, users), reverse=True, key=itemgetter(0))]
		ext_sinrs = np.array([x for _, x in sorted(zip(in_snrs, ext_sinrs), reverse=True, key=itemgetter(0))])
		in_sinrs = np.array([x for _, x in sorted(zip(in_snrs, in_sinrs), reverse=True, key=itemgetter(0))])
		self._users = users
		self._n_users = len(users)

		if self._n_users < 3:
			self._inner.setShape([])
			self._outter.setShape([])

			if self._n_users == 2:
				self._inner.setUsers(users[:1])
				self._outter.updateUsers(users[1:])
			else:
				self._inner.setUsers(users)
				self._outter.updateUsers([])
				
			return

		users_location = np.append(np.array([[u._location[0], u._location[1]] for u in users]), [[self._x, self._y]], axis=0)

		hull = ConvexHull(users_location)
		self._shape = users_location[hull.vertices]
		self._outter.setShape(self._shape)

		# Identify the best inner area according to the criterion
		rewards = [] # [alpha_fairness(ext_capacities, alpha)]
		inner_shapes = [] # [[]]
		inner_n_users = [] # [0]
		for i in range(1 if discrete else 2, self._n_users - 1):
			inner_users_location = np.append(users_location[:i], [users_location[-1]], axis=0)
			if not discrete:
				inner_hull = ConvexHull(inner_users_location)
				inner_shapes.append(inner_users_location[inner_hull.vertices])
			else:
				inner_shapes.append([] if i < 2 else inner_users_location[ConvexHull(inner_users_location).vertices])
			inner_n_users.append(i)
			self._inner.setShape(inner_shapes[-1])
			
			# Capacities
			area_in = i if discrete else self._inner.polygonApproximation().area
			inner_value = 0
			if area_in > 0:
				inner_capacities = in_bandwidth * np.log2(1.0 + in_sinrs[:i])
				inner_value = scheduled_alpha_fairness(inner_capacities, alpha)
				if denormalize:
					inner_value *= area_in

			area_out = (self._n_users - i) if discrete else self._outter.polygonApproximation().area
			outter_value = 0
			if area_out > 0:
				outter_capacities = ext_bandwidth * np.log2(1.0 + ext_sinrs[i:])
				outter_value = scheduled_alpha_fairness(outter_capacities, alpha)
				if denormalize:
					outter_value *= area_out
			rewards.append(inner_value + outter_value)

		amax = np.argmax(np.array(rewards))

		self._inner.setShape(inner_shapes[amax])
		self._inner.setUsers(self._users[:inner_n_users[amax]])
		self._outter.updateUsers(self._users[inner_n_users[amax]:])

	def getBounds(self):
		if self._params._many_areas:
			return [(MIN_DBM, MAX_DBM), (0.5, 0.99)]
		return [(MIN_DBM, MAX_DBM)]

	def getSummary(self):
		lines = []
		lines.append(f"== Cell ({self._x, self._y}) ==")
		if self._params._many_areas:
			lines.append("\t== Inner area ==")
			lines.append(f"\t\tInner power: {self._inner._base_station._tx_dBm} dBm")
			lines.append(f"\t\tUsers:")
			for usr in self._inner._users:
				lines.append(f"\t\t\t{usr._location}, SINR: {usr._sinr}, Distance to BS: {np.linalg.norm(np.array([self._x, self._y]) - np.array([*usr._location]))} meters")
			lines.append("\t== Outter area ==")

		lines.append(f"\t\tOutter power: {self._outter._base_station._tx_dBm} dBm")
		for usr in self._outter._users:
			lines.append(f"\t\t\t{usr._location}, SINR: {usr._sinr}, Distance to BS: {np.linalg.norm(np.array([self._x, self._y]) - np.array([*usr._location]))} meters")

		return lines

	def draw(self, ax, usersOnly=False):
		self._outter.draw(ax, usersOnly=usersOnly)
		if self._params._many_areas:
			self._inner.draw(ax, usersOnly=usersOnly, cId=self._cell_id)

	def setColor(self, c, n_colors):
		self._color = c
		self._n_colors = n_colors
		self._outter.setColor(c, n_colors)