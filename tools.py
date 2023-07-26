import numpy as np
import math
from time import sleep
from scipy.integrate import quad, dblquad
import shapely.geometry as sg
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
import sys

def flatten(l):
		return [item for sublist in l for item in sublist]

def l1_bounded(v1, v2, vmin, radius):
	s = v1 + v2
	if s <= radius:
		return v1, v2

	ratio_v1 = (v1 - vmin) / (s - 2 * vmin)
	ratio_v2 = 1 - ratio_v1
	scale = radius - 2 * vmin
	return ratio_v1 * scale + vmin, ratio_v2 * scale + vmin

def scheduled_alpha_fairness(values, alpha):
	n = len(values)
	if n == 0:
		return 0
	
	scheduling = scheduler(values, alpha)
	values = np.multiply(values, scheduling)

	if alpha == 1:
		return np.sum(np.log(values))

	return np.sum(np.power(values, [1.0 - alpha] * n)) / (1.0 - alpha)


# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
	path = Path.make_compound_path(
			Path(np.asarray(poly.exterior.coords)[:, :2]),
			*[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

	patch = PathPatch(path, **kwargs)
	collection = PatchCollection([patch], **kwargs)
	
	ax.add_collection(collection, autolim=True)
	ax.autoscale_view()
	return collection

def mWtodBm(mW):
	return 10 * np.log10(mW)

def dBmtomW(dBm):
	return 10 ** (dBm / 10)

def logPathLoss(txdBm, start, end, gain=0, ref_loss=-128.1, exp=3.76, min_dist=50):
	pl = txdBm
	distance = max(np.linalg.norm(end - start), min_dist)
	if distance >= 1:
		pl = txdBm + gain + (ref_loss - 10 * exp * np.log10(distance / 1e3))

	return pl

def scheduler(capacities, alpha):
	n = len(capacities)
	if n == 0:
		return []
	times = [0 for _ in range(n)]

	if alpha == 0:
		times[np.argmax(capacities)] = 1.0
	elif alpha == 1:
		for i in range(n):
			times[i] = 1.0 / n
	else:
		altered_sum = 0
		for i in range(n):
			altered_sum += capacities[i] ** ((1.0 - alpha) / alpha)
		for i in range(n):
			times[i] = capacities[i] ** ((1.0 - alpha) / alpha) / altered_sum
	return times

def getEnergy(config):
	e = 0.0
	for elem in config:
		e += dBmtomW(elem[0]) / 1e3

	return e

def logPathLossSINR(emitter, receiver, in_tx, others, gain=0, ref_loss=-128.1, exp=3.76, background_noise=-100, beamforming=0):
	received_mW = dBmtomW(logPathLoss(emitter[0], emitter[1], receiver, gain=gain, ref_loss=ref_loss, exp=exp)) / emitter[2]
	noise_mW = dBmtomW(background_noise)
	interferences_mW = 0
	for tx,loc,bdw in others:
		to_add = dBmtomW(logPathLoss(tx, loc, receiver, gain=gain, ref_loss=ref_loss, exp=exp)) / bdw
		if in_tx and bdw < emitter[2]:
			to_add *= bdw / emitter[2]
		interferences_mW += to_add

	return min((10 ** (beamforming / 10)) * received_mW / (noise_mW + interferences_mW), 500) # 

def logPathLossSNR(emitter, receiver, gain=0, ref_loss=-128.1, exp=3.76, background_noise=-150, beamforming=0):
	received_mW = dBmtomW(logPathLoss(emitter[0], emitter[1], receiver, gain=gain, ref_loss=ref_loss, exp=exp) + beamforming) / emitter[2]
	noise_mW = dBmtomW(background_noise)

	return received_mW / noise_mW

def in_box(stations, bounding_box):
	return np.logical_and(
		np.logical_and(
			bounding_box[0] <= stations[:, 0],
			stations[:, 0] <= bounding_box[1]),
		np.logical_and(
			bounding_box[2] <= stations[:, 1],
			stations[:, 1] <= bounding_box[3])
	)

def polygon_intersect_x(poly, x_val):
	"""
	Find the intersection points of a vertical line at
	x=`x_val` with the Polygon `poly`.
	"""
	# poly = poly.boundary
	vert_line = sg.LineString([[x_val, poly.bounds[1]],
															[x_val, poly.bounds[3]]])
	

	intersect = poly.intersection(vert_line)
	pts = []
	if type(intersect) != sg.point.Point:
		for line in intersect if type(intersect) != sg.LineString else [intersect]:
			if type(line) == sg.point.Point:
				continue
			pts += list(line.xy[1])

	return pts if pts != [] else [0.0, 0.0]

# The Impact of User Spatial Heterogeneity in Heterogeneous Cellular Networks
def logGaussianCoxProcess(w, h, dxy, mu, sigma):
	points = []
	for xi in range(0, w - dxy, dxy):
		for yi in range(0, h - dxy, dxy):
			# Sample a Gaussian 
			g = np.random.normal(mu, sigma)
			# PPP of intensity exp(g)
			intensity = np.exp(g)
			n_points = np.random.poisson(intensity)
			for _ in range(n_points):
				x = np.random.uniform(float(xi), float(xi+dxy))
				y = np.random.uniform(float(yi), float(yi+dxy))
				points.append([x, y])
	return np.array(points)

def voronoi(stations, bounding_box, eps=10*sys.float_info.epsilon):
	eps = min(0.01 * (bounding_box[1] - bounding_box[0]), 0.01 * (bounding_box[3] - bounding_box[2]))
	# Select stations inside the bounding box
	i = in_box(stations, bounding_box)
	# Mirror points
	points_center = stations[i, :]
	points_left = np.copy(points_center)
	points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
	points_right = np.copy(points_center)
	points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
	points_down = np.copy(points_center)
	points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
	points_up = np.copy(points_center)
	points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
	points = np.append(
		points_center,
		np.append(
			np.append(
				points_left,
				points_right,
				axis=0),
			np.append(
				points_down,
				points_up,
				axis=0),
		axis=0),
	axis=0)
	# Compute Voronoi
	vor = Voronoi(points)
	# Filter regions
	regions = []
	regions_idx = []
	for idx,region in enumerate(vor.regions):
		flag = True
		for index in region:
			if index == -1:
					flag = False
					break
			else:
					x = vor.vertices[index, 0]
					y = vor.vertices[index, 1]
					if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
									bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
							flag = False
							break
		if region != [] and flag:
			regions.append(region)
			regions_idx.append(idx)

	# Extract the region in the bounding box
	n_points = len(stations)
	vor.filtered_point_region = []
	# For point in the bounding box
	for idx_point in range(n_points):
		# Get the initial region index
		initial_region_idx = vor.point_region[idx_point]
		# Find the region index in the filtered region set
		new_region_idx = regions_idx.index(initial_region_idx)
		# Add the correspondance
		vor.filtered_point_region.append(new_region_idx)

	# Adjacency for each filtered_point
	vor.filtered_adjacencies = [[0 for _ in range(n_points)] for _ in range(n_points)]
	for p1,p2 in vor.ridge_points:
		if p1 < n_points and p2 < n_points:
			vor.filtered_adjacencies[p1][p2] = 1
			vor.filtered_adjacencies[p2][p1] = 1

	vor.filtered_points = points_center
	vor.filtered_regions = regions

	return vor

def plot_filtered_voronoi_2d(vor, ax):
	# Plot ridges
	for region in vor.filtered_regions:
			vertices = vor.vertices[region + [region[0]], :]
			ax.plot(vertices[:, 0], vertices[:, 1], 'k-')

def gaussian_kernel(x, y, L=500):
	# print(x, y, np.exp(-(np.linalg.norm(x - y) ** 2) / (2 * (L ** 2))))
	return np.exp(-(np.linalg.norm(x - y) ** 2) / (2 * (L ** 2)))

def triangle_area(triangle):
	a = np.linalg.norm(triangle[0] - triangle[1])
	b = np.linalg.norm(triangle[0] - triangle[2])
	c = np.linalg.norm(triangle[1] - triangle[2])

	s = (a + b + c) / 2
	return np.sqrt(s*(s-a)*(s-b)*(s-c))

def triangle_uniform_sampler(triangle):
	# One coordinate of the triangle in the origin
	dx = triangle[0, 0]
	dy = triangle[0, 1]

	centered_triangle = np.copy(triangle)
	centered_triangle[:, 0] -= dx
	centered_triangle[:, 1] -= dy

	u, v = centered_triangle[1], centered_triangle[2]

	s, t = np.random.uniform(0, 1, 2)
	in_centered_triangle = s + t <= 1

	p = s * u + t * v if in_centered_triangle else (1 - s) * u + (1 - t) * v
	p[0] += dx
	p[1] += dy

	return p

def polygon_uniform_sampler(n_points, triangles):
	points = []
	areas = []
	n = len(triangles)
	for triangle in triangles:
		areas.append(triangle_area(triangle))

	areas = np.array(areas)
	areas = areas / np.linalg.norm(areas, 1)

	for i in range(n_points):
		t_idx = np.random.choice(range(n), 1, p=areas)[0]
		p = triangle_uniform_sampler(triangles[t_idx])
		points.append(list(p))

	return np.array(points)


def dpp_sampler(n_points, bounding_shape, kernel=gaussian_kernel, T=500):
	# Sample n uniform points from the box
	triangles = Delaunay(bounding_shape)
	triangles = triangles.points[triangles.simplices]
	points = polygon_uniform_sampler(n_points, triangles)

	gram = []
	for p1 in points:
		g_line = []
		for p2 in points:
			g_line.append(kernel(p1, p2))
		gram.append(g_line)

	gram = np.array(gram)
	det = np.linalg.det(gram)

	for _ in range(T):
		# Find a swap
		to_remove = np.random.random_integers(0, n_points-1)
		point_to_add = polygon_uniform_sampler(1, triangles)[0]

		# Replace the point
		new_points = np.copy(points)
		new_points[to_remove] = point_to_add

		# Compute the Gram matrix with the new point
		new_line = np.array([kernel(point_to_add, p) for p in new_points])
		new_column = np.array([kernel(p, point_to_add) for p in new_points])
		new_gram = np.copy(gram)
		new_gram[to_remove, :] = new_line
		new_gram[:, to_remove] = new_column

		# Compute the new determinant
		new_det = np.linalg.det(new_gram)
		r = min(new_det / det, 1.0)

		# print(f"{new_det} / {det} gives prob = {r}")
		if np.random.uniform(0, 1) < r:
			# print("Swap accepted")
			# The swap is accepted
			gram = np.copy(new_gram)
			points = np.copy(new_points)
			det = new_det
		
	return points

def dot(v,w):
	x,y = v
	X,Y = w
	return x*X + y*Y

def length(v):
	x,y = v
	return math.sqrt(x*x + y*y)

def vector(b,e):
	x,y = b
	X,Y = e
	return (X-x, Y-y)

def unit(v):
	x,y = v
	mag = length(v)
	return (x/mag, y/mag)

def distance(p0,p1):
	return length(vector(p0,p1))

def scale(v,sc):
	x,y = v
	return (x * sc, y * sc)

def add(v,w):
	x,y = v
	X,Y = w
	return (x+X, y+Y)


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest 
# distance from pnt to the line and the coordinates of the 
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line. 
# Malcolm Kesson 16 Dec 2012
def pnt2line(pnt, start, end):
	line_vec = vector(start, end)
	pnt_vec = vector(start, pnt)
	line_len = length(line_vec)
	line_unitvec = unit(line_vec)
	pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
	t = dot(line_unitvec, pnt_vec_scaled)    
	if t < 0.0:
			t = 0.0
	elif t > 1.0:
			t = 1.0
	nearest = scale(line_vec, t)
	dist = distance(nearest, pnt_vec)
	nearest = add(nearest, start)
	return (dist, nearest)