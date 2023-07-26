from objects import *
import shapely.geometry.polygon as shapely_poly
import shapely.geometry as sg
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from matplotlib.patches import Polygon, RegularPolygon
from matplotlib import cm
from abc import ABC, abstractmethod
from tools import polygon_intersect_x, plot_polygon

BANDWIDTH = 20

class AbstractArea(ABC):
	def __init__(self, x, y, tx, tx_min, tx_max, params):
		self._x = x
		self._y = y
		self._tx_min = tx_min
		self._tx_max = tx_max
		self._base_station = CellularObject(self._x, self._y, tx)
		self._users = []
		self._params = params

	def getCardinality(self, discrete):
		return self.getNUsers() if discrete else self.getArea()

	@abstractmethod
	def getArea(self):
		...

	def getNUsers(self):
		return len(self._users)

class PolygonInnerArea(AbstractArea):
	def __init__(self, x, y, tx, tx_min, tx_max, params):
		super().__init__(x, y, tx, tx_min, tx_max, params)

	def setConfiguration(self, tx):
		self._base_station._tx_dBm = tx

	def getConfiguration(self):
		return [self._base_station._tx_dBm]

	def getBounds(self):
		return [(self._tx_min, self._tx_max)]

	def setShape(self, shape):
		self._shape = [] if len(shape) < 3 else shape

	def getArea(self):
		return 0 if self._shape == [] else shapely_poly.Polygon(self._shape).area

	def polygonApproximation(self):
		return shapely_poly.Polygon(self._shape)

	def setUsers(self, users):
		self._users = users

	def getBandwidth(self):
		bdw = BANDWIDTH
		if self._params._ffr:
			bdw = self._params._ffr_threshold * BANDWIDTH
		return bdw

	def draw(self, ax, usersOnly=False, cId=None):
		if not usersOnly:
			if self._shape != [] and not usersOnly:
				tx_alpha = 0.6 # (self._base_station._tx_dBm - self._tx_min) / (self._tx_max - self._tx_min)
				p = Polygon(np.array(self._shape), edgecolor='black', facecolor="#4D4D4D", alpha=tx_alpha)
				ax.add_patch(p)

			ax.scatter([self._x], [self._y], c='red', linewidth=0.25, edgecolor='black')
			# if cId is not None:
			# 	ax.annotate(cId, (self._x, self._y))

class AbstractCircularInnerArea(AbstractArea):
	def __init__(self, x_center, y_center, inner_radius, rad_min, rad_max, tx, tx_min, tx_max, n_edges_approximation=30):
		super().__init__(x_center, y_center, tx, tx_min, tx_max)
		self._n_edges_approximation = n_edges_approximation
		self._radius = inner_radius
		self._rad_min = rad_min
		self._rad_max = rad_max
		self._outter_area = None
		self._poly_approx = None
		self._updated = False

		# Create a set of ill-placed users
		self.updateUsers()

	def compatibleDblquad(self):
		return True

	def getMinArea(self):
		return np.pi * (self._rad_min ** 2)

	def getBandwidth(self):
		return BANDWIDTH

	def getYIntersections(self, x, poly=None):
		poly = self.polygonApproximation()
		return polygon_intersect_x(poly, x)

	def polygonApproximation(self):
		if not self._updated:
			return self._poly_approx

		angle = 2.0 * np.pi / self._n_edges_approximation
		outter_without_holes = self._outter_area.polygonApproximation(holes=False)
		inner = shapely_poly.Polygon(np.array([[self._x + self._radius * np.cos(i * angle), self._y + self._radius * np.sin(i * angle)] for i in range(self._n_edges_approximation)]))
		max_inner = inner.intersection(outter_without_holes)

		self._poly_approx = max_inner
		self._updated = False

		return self._poly_approx

	def setRadius(self, r):
		"""Set the radius of the area

		Args:
				r (double): the new radius of the area
		"""
		self._updated = True
		self._radius = r
		self.updateUsers()

	def draw(self, ax, usersOnly=False):
		"""Draw the area on a given Matplotlib axis

		Args:
				ax (axis): the Matplotlib axis to draw on
		"""
		if not usersOnly:
			tx_alpha = 0.6 # (self._base_station._tx_dBm - self._tx_min) / (self._tx_max - self._tx_min)
			circle = plt.Circle((self._x, self._y), self._radius, facecolor='#4D4D4D', edgecolor='black', alpha=tx_alpha)
			ax.add_patch(circle)

			# Base station
			ax.scatter([self._x], [self._y], c='red', linewidth=0.25, edgecolor='black')

	def setOutterArea(self, oa):
		self._updated = True
		self._outter_area = oa

	def getArea(self):
		return self.polygonApproximation().area

	def setConfiguration(self, power, radius):
		self._base_station._tx_dBm = power
		self.setRadius(radius)

	def getConfiguration(self):
		return [self._base_station._tx_dBm, self._radius]

	def getBounds(self, rad_normalized=False):
		return [(self._tx_min, self._tx_max), (self._rad_min, self._rad_max)]

	def getXBoundaries(self):
		p = self.polygonApproximation()
		return p.bounds[0], p.bounds[2]

	@abstractmethod
	def updateUsers(self):
		...

class InnerArea(AbstractCircularInnerArea):
	def __init__(self, x_center, y_center, inner_radius, users, rad_min, rad_max, tx, tx_min, tx_max):
		self._cell_users = users
		super().__init__(x_center, y_center, inner_radius, rad_min, rad_max, tx, tx_min, tx_max)

	def updateUsers(self, users=None):
		if users is not None:
			self._cell_users = users

		self._users = []
		for u in self._cell_users:
			du = np.linalg.norm(np.array(u._location) - np.array([self._x, self._y]))
			if du <= self._radius:
				self._users.append(u)

class AbstractOutterArea(AbstractArea):
	def __init__(self, x_bs, y_bs, inner_area, tx, tx_min, tx_max, color=None, n_colors=1, params=None):
		super().__init__(x_bs, y_bs, tx, tx_min, tx_max, params)
		self._inner_area = inner_area
		self._color = color
		self._n_colors = n_colors

		self.updateUsers()

	def compatibleDblquad(self):
		return False

	def setConfiguration(self, power):
		self._base_station._tx_dBm = power
		self.updateUsers()

	def setColor(self, color, n_colors=None):
		self._color = color

		if n_colors is not None:
			self._n_colors = n_colors

	def getBandwidth(self):
		bdw = BANDWIDTH / self._n_colors
		if self._params._ffr:
			bdw *= (1 - self._params._ffr_threshold)
		elif self._params._full_reuse:
			bdw = BANDWIDTH
		return bdw

	def getConfiguration(self):
		return [self._base_station._tx_dBm]

	def getBounds(self):
		return [(self._tx_min, self._tx_max)]

	def drawUsers(self, ax):
		pass

	def getYIntersections(self, x, poly=None):
		if poly is None:
			poly = self.polygonApproximation()

		return polygon_intersect_x(poly, x)

	@abstractmethod
	def getXBoundaries(self):
		...

	@abstractmethod
	def draw(self, ax, usersOnly=False):
		...

	@abstractmethod
	def updateUsers(self, users=None):
		...

	@abstractmethod
	def polygonApproximation(self):
		...
	

class PolygonOutterArea(AbstractOutterArea):
	def __init__(self, x_bs, y_bs, shape, users, inner_area, tx, tx_min, tx_max, color=None, params=None):
		self._shape = shape
		self._cell_users = users
		super().__init__(x_bs, y_bs, inner_area, tx, tx_min, tx_max, color=color, params=params)

	def draw(self, ax, usersOnly=False):
		if self._shape == []:
			return

		color = 'gray'
		if not usersOnly:
			if self._color is not None:
				color = self._color

			tx_alpha = 0.6
			p = Polygon(np.array(self._shape), edgecolor='black', facecolor=color, alpha=tx_alpha)
			ax.add_patch(p)

		self.drawUsers(ax)
		
	def updateUsers(self, users=None):
		if users is not None:
			self._cell_users = users
			
		self._users = []
		for u in self._cell_users:
			if not self._params._many_areas or u not in self._inner_area._users:
				self._users.append(u)

	def getArea(self, inner_area=None):
		if inner_area is None:
			inner_area = self._inner_area.getArea() if self._params._many_areas else 0

		x = np.array(self._shape)[:, 0]
		y = np.array(self._shape)[:, 1]
		return 0.5 * np.abs(np.dot(x,np.roll(y,1)) - np.dot(y,np.roll(x,1))) - inner_area # Shoelace formula

	def setShape(self, shape):
		self._shape = shape

	def polygonApproximation(self, holes=True):
		outter = np.array(self._shape)
		inner_approx = self._inner_area.polygonApproximation() if self._params._many_areas else []

		if not self._params._many_areas and (not holes or inner_approx == []):
			return shapely_poly.Polygon(outter)

		inner = np.array(list(zip(*inner_approx.exterior.coords.xy)))[:-1].reshape(1, -1, 2)
		poly = shapely_poly.Polygon(outter, holes=inner)

		return poly

	def getXBoundaries(self):
		xx = np.array([v[0] for v in self._shape])
		return np.min(xx), np.max(xx)
