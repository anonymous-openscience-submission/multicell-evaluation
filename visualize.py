from pickle import NONE
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import viridis
from inspire import areaCapacities
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import seaborn as sb
import itertools as it
import numpy as np


markers = ['x', 'D', 'o', '*', '^', 'v', '>', 's']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def txt_tick(i, dynamic=False):
	idx_cell = int(np.floor(i / (2.0 if dynamic else 3.0))) + 1
	label_id = i % (2 if dynamic else 3)
	labels = ['intx', 'outtx'] if dynamic else ['intx', 'inrad', 'outtx']

	return f"{idx_cell}/{labels[label_id]}"

def flatten(l):
	return list(it.chain(*l))

def ema(series, alpha):
	ema_series = [series[0]]
	n = len(series)

	for i in range(1, n):
		ema_series.append((1 - alpha) * ema_series[-1] + alpha * series[i])

	return ema_series

def inner_tx(config):
	txs = []
	for i,v in enumerate(config):
		if i % 2 == 0:
			txs.append(v[0])
	return txs

def inner_rad(config):
	rads = []
	for i,v in enumerate(config):
		if i % 2 == 0:
			rads.append(v[1])
	return rads

def outter_tx(config):
	txs = []
	for i,v in enumerate(config):
		if i % 2 == 1:
			txs.append(v[0])
	return txs

def normalize(config, bounds):
	normalized = []
	for i,vs in enumerate(config):
		n_vs = []
		for j,v in enumerate(vs):
			m,M = bounds[i][j]
			n_vs.append((v - m) / (M - m))
		normalized.append(n_vs)
	return normalized

def flat_normalize(configs, bounds):
	flatten_configs = [flatten(c) for c in configs]
	flatten_bounds = flatten(bounds)
	configs = np.array(flatten_configs)
	configs = np.transpose(configs)

	for i,(m,M) in enumerate(flatten_bounds):
		configs[i] = (configs[i] - m) / (M - m)

	return configs

def read_file(path):
	f = open(path)
	data = []
	lines = f.readlines()

	for line in lines:
		data.append(eval(line))

	return data

def rewards(global_rew, indiv_rews, alpha, path=None):
	ema_global = ema(global_rew, alpha)
	ema_indivs = []

	for indiv_rew in indiv_rews:
		ema_indivs.append(ema(indiv_rew, alpha))

	xx = range(len(global_rew))

	_,axes = plt.subplots(ncols=2)
	axes[0].plot(xx, ema_global, label="Global Reward")
	axes[0].bar(xx, global_rew, alpha=0.2)
	color = 'orange'
	for i,ema_indiv in enumerate(ema_indivs):
		axes[1].plot(xx, ema_indiv, c=color, alpha=0.2)

	if path is None:
		plt.show()
	else:
		plt.savefig(path, bbox_inches='tight')

def many_alpha_fairnesses(many_reward_files, alpha, bmf, techs, ax=None, c=20, filepath=None):
	plt.rcParams.update({'font.size': 18})
	ax_is_none = ax is None
	idx_to_plot = np.arange(0, 110, 10)
	xx = np.array(range(1, 111))[idx_to_plot]
	if ax is None:
		_, ax = plt.subplots()

	for i,reward_files in enumerate(many_reward_files):
		data = []
		for f in reward_files:
			d = read_file(f)
			data.append([(c ** (1 - alpha)) * max(d[:i]) for i in range(1, len(d)+1)])

		data_ema = np.array(data)[:, idx_to_plot]
		data_mean = np.mean(data_ema, axis=0)
		data_errs = np.std(data_ema, axis=0)

		ax.plot(xx, data_mean, marker=markers[i], markersize=10, label=f"{techs[i]}", color=colors[i])
		ax.fill_between(xx, (data_mean - data_errs), (data_mean + data_errs), alpha=0.2, color=colors[i])
	
	leg = ax.legend(ncol=1, labelspacing=0.25, handletextpad=0.5, columnspacing=0.5, loc=7)
	leg.set_draggable(True)
	ax.set_xlabel("Optimization step")
	ax.set_ylabel("α-Fairness")
	ax.set_title(f"α={alpha}, Bmf={bmf} dB")

	if ax_is_none:
		if filepath is None:
			plt.show()
		else:
			plt.savefig(filepath, bbox_inches="tight")
			plt.clf()
	
def many_alpha_global_rewards(many_reward_files, alphas, techs, ax=None, c=20):
	plt.rcParams.update({'font.size': 18})
	ax_is_none = ax is None
	idx_to_plot = np.arange(0, 110, 10)
	xx = np.array(range(1, 111))[idx_to_plot]
	if ax is None:
		_, ax = plt.subplots()

	n_alphas = len(alphas)
	n_techs = len(techs)

	for i,reward_files in enumerate(many_reward_files):
		data = []
		idx_tech = i % n_techs
		idx_alpha = int(np.floor(i / n_techs))
		for f in reward_files:
			print(f)
			d = read_file(f)
			data.append([(c ** (1 - alphas[idx_alpha])) * max(d[:i]) for i in range(1, len(d)+1)])

		data_ema = np.array(data)[:, idx_to_plot]
		data_mean = np.mean(data_ema, axis=0)
		data_errs = np.std(data_ema, axis=0)

		ax.plot(xx, data_mean, marker=markers[idx_tech], markersize=10, label=f"{techs[idx_tech]}", color=colors[idx_alpha])
		ax.fill_between(xx, (data_mean - data_errs), (data_mean + data_errs), alpha=0.2, color=colors[idx_alpha])
	
	leg = ax.legend(ncol=3, labelspacing=0.25, handletextpad=0.5, columnspacing=0.5, loc=7)
	leg.set_draggable(True)
	ax.set_xlabel("Optimization step")
	ax.set_ylabel("Global reward")

	if ax_is_none:
		plt.show()

def many_global_rewards(reward_files, names, ax=None, enhance=None, distribution=False, scale=None):
	plt.rcParams.update({'font.size': 18})
	ax_is_none = ax is None
	if ax is None:
		_, ax = plt.subplots()

	is_enhanced = enhance is not None

	not_enhanced_style = "--" if is_enhanced else "-"
	data_ema = []
	for f in reward_files:
		d = read_file(f)
		data_ema.append([max(d[:i]) for i in range(1, len(d)+1)])

	lens = [len(d) for d in data_ema]
	min_len = min(lens)
	xx = np.array(range(1, min_len+1))
	data_ema = [d[:min_len] for d in data_ema]

	data_ema = np.array(data_ema)
	data_mean = np.mean(data_ema, axis=0)
	std_err = np.std(data_ema, axis=0) / np.sqrt(len(data_ema))
	idx_to_plot = np.arange(0, 110, 10)
	print(idx_to_plot)

	if not distribution:
		ax.plot(xx[idx_to_plot], data_mean[idx_to_plot], marker='d', markersize=10)
		ax.fill_between(xx[idx_to_plot], (data_mean - std_err)[idx_to_plot], (data_mean + std_err)[idx_to_plot], alpha=0.2)
	else:
		data_not_enhanced = np.array([d for i,d in enumerate(data_ema) if i != enhance])
		q = np.quantile(data_not_enhanced, [0.0, 0.25, 0.5, 0.75, 1.0], axis=0)
		p = ax.plot(xx, q[2], label="Med(Rewards)")
		ax.fill_between(xx, q[1], q[3], color=p[0].get_color(), alpha=0.2)
		ax.plot(xx, q[0], linestyle="--", color=p[0].get_color(), alpha=0.2, label="Min(Rewards)")
		ax.plot(xx, q[4], linestyle="--", color=p[0].get_color(), alpha=0.2, label="Max(Rewards)")
		if is_enhanced:
			ax.plot(xx, data_ema[enhance], linewidth=3, label="Enhanced Reward")

	if is_enhanced or distribution:
		ax.legend()
	ax.set_title("Max Global Reward w.r.t Optimization Step")
	ax.set_xlabel("Optimization step")
	ax.set_ylabel("Global reward")
	if scale is not None:
		ax.set_yscale(scale)

	if ax_is_none:
		plt.show()

def mean_configurations(configs, bounds, path=None, ax=None, title="Mean of configurations", dynamic=False):
	ax_none = ax is None
	if ax is None:
		_,ax = plt.subplots()

	configs = flat_normalize(configs, bounds)

	p = sb.heatmap(np.mean(configs, axis=1).reshape((2 if dynamic else 3, -1), order='F'), cmap='viridis', ax=ax, vmin=0.0, vmax=1.0, xticklabels=range(1, int(configs.shape[0] / (2 if dynamic else 3)) + 1))

	ticks = np.arange(0, 2 if dynamic else 3)
	ticklabels = ['intx', 'outtx'] if dynamic else ['intx', 'inrad', 'outtx']
	p.set_yticks(ticks + 0.5)
	p.set_yticklabels(ticklabels, rotation=0)

	ax.set_title(title)

	if ax_none:
		if path is None:
			plt.show()
		else:
			plt.savefig(path, bbox_inches='tight')

def std_dev_configurations(configs, bounds, path=None, ax=None, title="Standard deviations of configurations", dynamic=False):
	ax_none = ax is None
	if ax is None:
		_,ax = plt.subplots()

	configs = flat_normalize(configs, bounds)

	p = sb.heatmap(np.std(configs, axis=1).reshape((2 if dynamic else 3, -1), order='F'), cmap='bwr', ax=ax, center=0, xticklabels=range(1, int(configs.shape[0] / (2 if dynamic else 3)) + 1))

	ticks = np.arange(0, 2 if dynamic else 3)
	ticklabels = ['intx', 'outtx'] if dynamic else ['intx', 'inrad', 'outtx']
	p.set_yticks(ticks + 0.5)
	p.set_yticklabels(ticklabels, rotation=0)

	ax.set_title(title)

	if ax_none:
		if path is None:
			plt.show()
		else:
			plt.savefig(path, bbox_inches='tight')

def covariance_matrix_configurations(configs, bounds, path=None, ax=None, xtick_period=2, ytick_period=3, title="Covariance matrix of configurations"):
	ax_none = ax is None
	if ax is None:
		_,ax = plt.subplots()

	flatten_configs = [flatten(c) for c in configs]
	configs = flat_normalize(configs, bounds)

	p = sb.heatmap(np.cov(configs), cmap='bwr', ax=ax, center=0, vmin=-1.0, vmax=1.0)

	xticks = np.arange(0, len(flatten_configs[0]), xtick_period)
	xticklabels = [txt_tick(i) for i in xticks]
	p.set_xticks(xticks + 0.5)
	p.set_xticklabels(xticklabels)

	yticks = np.arange(0, len(flatten_configs[0]), ytick_period)
	yticklabels = [txt_tick(i) for i in yticks]
	p.set_yticks(yticks + 0.5)
	p.set_yticklabels(yticklabels)

	ax.set_title(title)

	if ax_none:
		if path is None:
			plt.show()
		else:
			plt.savefig(path, bbox_inches='tight')

def topology_with_beamforming(simulators, configs, beamforming, criteria, label):
	height = len(beamforming)
	width = len(criteria)

	fig, axes = plt.subplots(nrows=height, ncols=width)
	if height == 1:
		axes = [axes]
	if width == 1:
		for i in range(len(axes)):
			axes[i] = [axes[i]]

	for i,sim in enumerate(simulators):
		x = i % width
		y = int(np.floor(i / width))
		sim.setGlobalConfiguration(configs[i])
		sim.draw(ax=axes[y][x], title="" if y != 0 else f"$\\alpha = {criteria[x]}$", ylabel="" if x != 0 else f"Bmf Gain: {beamforming[y]} dB")
	
	fig.suptitle(label)
	plt.show()

def SINR_distributions_alphafair(simulators, criteria, technologies, configs):
	height = len(criteria)
	width = len(technologies)
	data = [[None for j in range(height)] for i in range(width)] # Criterion / Techs / SINR
	max_logsinr = 0
	min_logsinr = 0

	for i,(sim,config) in enumerate(zip(simulators, configs)):
		y = i % height
		x = int(np.floor(i / height))

		sim.setGlobalConfiguration(config)
		sim._topo.startTransmissions()
		sinrs = sim._topo.getEndUsersSINRs()

		flattened_sinrs = []
		for s_area in sinrs:
			flattened_sinrs += [s[2] for s in s_area]

		challenger_max = np.max(np.log10(flattened_sinrs))
		challenger_min = np.min(np.log10(flattened_sinrs))
		if challenger_max > max_logsinr:
			max_logsinr = challenger_max
		if challenger_min < min_logsinr:
			min_logsinr = challenger_min

		data[x][y] = flattened_sinrs


	logbins = np.logspace(min_logsinr, max_logsinr,15)
	_, axes = plt.subplots(nrows=height, ncols=width)
	for i in range(len(simulators)):
		y = i % height
		x = int(np.floor(i / height))
		axes[y][x].hist(data[x][y], bins=logbins)
		axes[y][x].set_xscale('log')

		if x == 0:
			axes[y][x].set_ylabel(f"$\\alpha = {criteria[y]}$")
		if y == 0:
			axes[y][x].set_title(f"{technologies[x]}")

	plt.show()

def compare_throughput_cdfs(configured_simulators, title, fills, ax=None, xlabel=None, ylabel=None, legend=False, imgpath=None, zone=None):
	ax_is_none = ax is None
	if ax_is_none:
		_, ax = plt.subpots()
	marker_period = 0.1
	networks_flattened = []
	min_v, max_v = 10000000, 0.000000001
	for k,sim in enumerate(configured_simulators):
		metrics = sim.basicUserMetric(scheduled=True, zone=zone)

		i = 0
		network_flattened = []
		for cell_metric in metrics:
			data_cell = []
			for _,area_metric in cell_metric:
				data_cell.append([metric for metric in area_metric])
				network_flattened += data_cell[-1]
				i += 1
			
		min_v = min(min(network_flattened), min_v)
		max_v = max(max(network_flattened), max_v)
		networks_flattened.append(network_flattened)

	scatters = []
	for k,network_flattened in enumerate(networks_flattened):
		n = list(np.arange(0,len(network_flattened)+1) / np.float(len(network_flattened)))
		n.append(1.0)

		network_flattened += [min_v, max_v]
		network_flattened_s = np.sort(network_flattened)
		lines = ax.step(network_flattened_s, n, linewidth=2, marker=markers[k], markevery=int(len(n) * marker_period), markersize=9)
		if fills[k]:
			ax.fill_between(network_flattened_s, np.zeros(len(n)), n, color=lines[0].get_color(), alpha=0.15)
		scatters.append(lines)

	ax.set_xscale('log')
	if xlabel is not None:
		ax.set_xlabel(xlabel)
	if ylabel is not None:
		ax.set_ylabel(ylabel)
	if title is not None:
		ax.set_title(title)
	if legend:
		ax.legend(labelspacing=0.25, handletextpad=0.5, columnspacing=0.5)
	if imgpath is None and ax_is_none:
		plt.show()
	elif imgpath is not None:
		plt.savefig(f"{imgpath}", bbox_inches='tight')
		plt.clf()
	
	return scatters

def throughput_distributions_alphafair(simulators, criteria, configs, label):
	plt.rcParams.update({'font.size': 13})
	data = [] # Criterion / Cell / Area
	ratios = []
	max_throughputs = []
	min_throughputs = []
	max_densities = []
	for (sim, config) in zip(simulators, configs):
		data_crit = []
		ratio_crit = []
		sim.setGlobalConfiguration(config)
		metrics = sim.basicUserMetric()
		areas = sim._topo.getCardinalities()

		ratio_available = len(areas) > len(metrics)

		i = 0
		for j,cell_metric in enumerate(metrics):
			data_cell = []
			flattened = []
			if ratio_available:
				ratio_crit.append(areas[i] / areas[i + 1])
			for _,area_metric in cell_metric:
				w = areas[i]
				data_cell.append([metric / w for metric in area_metric])
				flattened += data_cell[-1]
				i += 1
			
			h,b = np.histogram(flattened)
			if len(max_throughputs) < j+1:
				max_densities.append(np.max(h))
				max_throughputs.append(b[-1])
				min_throughputs.append(b[0])
			else:
				max_densities[j] = max(max_densities[j], np.max(h))
				max_throughputs[j] = max(max_throughputs[j], b[-1])
				min_throughputs[j] = min(min_throughputs[j], b[0])
			data_crit.append(data_cell)
		
		ratios.append(ratio_crit)
		data.append(data_crit)

	height = len(criteria)
	width = len(data[0])
	fig, axes = plt.subplots(nrows=height, ncols=width)
	for i,crit in enumerate(criteria):
		for j,cell in enumerate(data[i]):
			axes[i][j].hist(cell, histtype='barstacked')
			axes[i][j].set_xlim(min_throughputs[j], max_throughputs[j])
			axes[i][j].set_ylim(0, max_densities[j])
			if ratio_available:
				axes[i][j].text((max_throughputs[j] - min_throughputs[j]) * 0.025 + min_throughputs[j], 0.9 * max_densities[j], f"Ratio #I/#O: {str(round(ratios[i][j], 2))}")
			if i == 0:
				axes[i][j].set_title(f"Cell {j+1}")
			if i == len(criteria)-1:
				axes[i][j].set_xlabel("Capacity (Mbps)")
			if j == 0:
				axes[i][j].set_ylabel(f"$\\alpha = {crit}$")

	fig.suptitle(label)
	plt.show()

def showManyEnergy(sets_of_points, labels, criteria, bmf, ylabel=False, ax=None):
	ax_is_none = ax is None
	if ax_is_none:
		_, ax = plt.subplots()

	X_axis = np.arange(len(criteria))
	
	for k,(lab,points) in enumerate(zip(labels, sets_of_points)):
		scatter = ax.plot(X_axis, points, linewidth=2, marker=markers[k], markersize=9, label=lab)
	ax.set_xticklabels([0.25] + criteria)

	ax.set_xlabel("α")
	ax.set_title(f"Bmf Gain = {bmf} dB")

	if ylabel:
		ax.set_ylabel("Power (W)")
	
	if ax_is_none:
		plt.legend()
		plt.show()

	return scatter

def throughput_distributions(s, config, label):
	s.setGlobalConfiguration(config)
	metrics = s.basicUserMetric()
	areas = s._topo.getCardinalities()

	xx = []
	i = 0
	n = len(metrics)
	for cell_metric in metrics:
		cell_xx = []
		for _,area_metric in cell_metric:
			w = areas[i]
			cell_xx.append([metric / w for metric in area_metric])
			i += 1
		xx.append(cell_xx)

	height = int(round(np.sqrt(9 * n / 16)))
	width = int(np.ceil(n / height))
	_, axes = plt.subplots(nrows=height, ncols=width)
	for i in range(n):
		x = i % width
		y = int(np.floor(i / width))
		print([len(d) for d in xx[i]])
		axes[y][x].hist(xx[i], histtype='barstacked')
		axes[y][x].set_xlabel("End User Throughput")
		axes[y][x].set_title(f"Cell {i+1}")
	
	plt.title(label)
	plt.show()

def correlation_matrix_configurations(configs, bounds, path=None, ax=None, xtick_period=2, ytick_period=3, title="Correlation matrix of configurations", dynamic=False):
	ax_none = ax is None
	if ax is None:
		_,ax = plt.subplots()

	flatten_configs = [flatten(c) for c in configs]
	configs = flat_normalize(configs, bounds)

	p = sb.heatmap(np.corrcoef(configs), cmap='bwr', ax=ax, center=0, vmin=-1.0, vmax=1.0)
	xticks = np.arange(0, len(flatten_configs[0]), xtick_period)
	xticklabels = [txt_tick(i, dynamic=dynamic) for i in xticks]
	p.set_xticks(xticks + 0.5)
	p.set_xticklabels(xticklabels)

	yticks = np.arange(0, len(flatten_configs[0]), ytick_period)
	yticklabels = [txt_tick(i) for i in yticks]
	p.set_yticks(yticks + 0.5)
	p.set_yticklabels(yticklabels)

	ax.set_title(title)

	if ax_none:
		if path is None:
			plt.show()
		else:
			plt.savefig(path, bbox_inches='tight')

def compareBestConfigurations(reward_files, config_files, bounds, alpha, path=None, dynamic=False, scale=None):
	best_configs = []
	best_rews = []
	configs = []
	rewards = []
	for c,r in zip(config_files, reward_files):
		configs.append(read_file(c))
		rewards.append(read_file(r))

	min_len = min(len(r) for r in rewards)
	rewards = [r[:min_len] for r in rewards]
	configs = [c[:min_len] for c in configs]

	for i,(c,r) in enumerate(zip(configs, rewards)):
		config, rew = extractBestConfiguration(c, r)
		print(reward_files[i], rew)
		best_configs.append(config)
		best_rews.append(rew)

	# Values
	f = plt.figure()
	f.set_figheight(8.5)
	f.set_figwidth(17)
	ax1 = plt.subplot2grid((2, 3), (0, 0))
	ax2 = plt.subplot2grid((2, 3), (0, 1))
	ax3 = plt.subplot2grid((2, 3), (0, 2))
	ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)

	many_global_rewards(reward_files, alpha, ax=ax1, scale=scale)
	ax2.bar(range(1, len(reward_files)+1), best_rews)
	ax2.set_title("Best rewards obtained from many simulations")
	ax2.set_xlabel("Simulation ID")
	if scale is not None:
		ax2.set_yscale(scale)
	ax2.set_ylabel("Best reward")
	
	np_rewards = np.array(rewards)
	print(np_rewards.shape)
	indices_of_max = np.argmax(np_rewards, axis=1)
	ax3.hist(indices_of_max, range(1, np_rewards.shape[1]+1))
	ax3.set_xlabel("Optimization step")
	ax3.set_title("Distribution of indices of best configurations")

	ax4.boxplot(np_rewards)
	ax4.plot(range(1, np_rewards.shape[1]+1), np.mean(np_rewards, axis=0), marker='x', c='red')
	ax4.set_xlabel("Optimization step")
	ax4.set_ylabel("Global reward")
	if scale is not None:
		ax4.set_yscale(scale)
	ax4.set_title("Distribution of rewards at each optimization step for all the simulations")


	plt.tight_layout()
	if path is None:
		plt.show()
	else:
		plt.savefig(path, bbox_inches="tight")

	# Marginal distributions
	f = plt.figure()
	f.set_figheight(8.5)
	f.set_figwidth(17)
	distributions_along_dimensions(best_configs, bounds, dynamic=dynamic)
	
	# Variations
	nrows = 6
	ncols = 4
	f = plt.figure()
	f.set_figheight(8.5)
	f.set_figwidth(17)

	ax1 = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=nrows-1)
	ax2 = plt.subplot2grid((nrows, ncols), (0, 1), rowspan=nrows-1, colspan=ncols-1)
	middle = int(ncols / 2)
	ax3 = plt.subplot2grid((nrows, ncols), (nrows-1, 0), colspan=middle)
	ax4 = plt.subplot2grid((nrows, ncols), (nrows-1, middle), colspan=middle)

	colormap_configurations(best_configs, bounds, ax=ax1, ytick_period=3, title="Best Configurations", dynamic=dynamic)
	correlation_matrix_configurations(best_configs, bounds, ax=ax2, ytick_period=3, xtick_period=5, title="Correlation Matrix of Best Configurations", dynamic=dynamic)
	mean_configurations(best_configs, bounds, ax=ax3, title="Mean of Best Configurations", dynamic=dynamic)
	std_dev_configurations(best_configs, bounds, ax=ax4, title="Standard Deviations of Best Configurations", dynamic=dynamic)

	plt.tight_layout()
	if path is None:
		plt.show()
	else:
		plt.savefig(path, bbox_inches='tight')

	# Visualization
	f = plt.figure()
	f.set_figheight(8.5)
	f.set_figwidth(17)
	config_kpca(configs, rewards, bounds, title="Best Configs", scale=scale) # configs_for_pca=best_configs

def config_kpca(all_configs, all_rewards, bounds, path_ax=None, box_ax=None, title="Configurations", configs_for_pca=None, scale=None):
	ax_is_none = path_ax is None
	if ax_is_none:
		path_ax = plt.subplot2grid((1, 1), (0, 0), projection='3d')

	if configs_for_pca is None:
		configs_for_pca = []
		for sim_configs in all_configs:
			configs_for_pca += sim_configs

	configs_for_pca = np.transpose(flat_normalize(configs_for_pca, bounds))
	all_flattened_configs = []
	all_selected_rewards = []
	n = len(all_configs)
	values = list(range(n))
	choices = []
	for _ in range(n):
		choices.append(np.random.choice(values))
		idx = values.index(choices[-1])
		values = values[:idx] + values[idx+1:]

	for idx_sim in choices:
		all_flattened_configs.append(np.transpose(flat_normalize(all_configs[idx_sim], bounds)))
		all_selected_rewards.append(all_rewards[idx_sim])

	ss = StandardScaler()
	X_scaled = ss.fit_transform(configs_for_pca)

	# PCA of configs
	pca = KernelPCA(kernel='rbf')
	pca.fit_transform(X_scaled)

	# PCA of trajectories
	# Scale the reward between 0 and 1
	rewards = np.array(all_selected_rewards)
	for idx_sim,sim_flattened in enumerate(all_flattened_configs):
		pca_sim_flattened = pca.transform(ss.transform(sim_flattened))
		sim_rewards = rewards[idx_sim]
		colors = viridis(np.array(range(len(all_configs[0]))) / (len(all_configs[0]) - 1))
		path_ax.scatter(pca_sim_flattened[:, 0], pca_sim_flattened[:, 1], sim_rewards, c=colors, alpha=0.6, edgecolors='gray')
	
	if scale is not None:
		path_ax.set_zscale(scale)
	path_ax.set_title("Simulation trajectories in features space w.r.t. rewards")
	path_ax.set_xlabel("kPC 1")
	path_ax.set_ylabel("kPC 2")
	path_ax.set_zlabel("Reward")

	if ax_is_none:
		plt.tight_layout()
		plt.show()

def colormap_configurations(configs, bounds, path=None, ax=None, ytick_period=3, title="Configurations", dynamic=False):
	ax_none = ax is None
	if ax is None:
		_,ax = plt.subplots()

	flatten_configs = [flatten(c) for c in configs]
	configs = flat_normalize(configs, bounds)

	p = sb.heatmap(configs, cmap='viridis', ax=ax)
	yticks = np.arange(0, len(flatten_configs[0]), ytick_period)
	yticklabels = [txt_tick(i, dynamic=dynamic) for i in yticks]
	p.set_yticks(yticks + 0.5)
	p.set_yticklabels(yticklabels)

	ax.set_title(title)

	if ax_none:
		if path is None:
			plt.show()
		else:
			plt.savefig(path, bbox_inches='tight')

def distributions_along_dimensions(configs, bounds, dynamic=False):
	configs = np.transpose(flat_normalize(configs, bounds))

	nrows = 4 if dynamic else 5
	ncols = int(len(configs[0]) / nrows)
	for i in range(nrows):
		for j in range(ncols):
			ax = plt.subplot2grid((nrows, ncols), (i, j))
			idx = i * ncols + j
			title = f"{int(np.floor(idx / 2.0)+1)}/{['intx', 'outtx'][idx % 2]}" if dynamic else f"{int(np.floor(idx / 3.0)+1)}/{['intx', 'inrad', 'outtx'][idx % 3]}"
			data = configs[:, idx]
			ax.hist(data, range=(0, 1))
			ax.set_title(title)

	plt.tight_layout()
	plt.show()

def colormap_configurations_diff(configs, bounds, alpha, ax_hmap=None, ax_variability=None, ax_dimension=None, path=None):
	ax_is_none = False
	if ax_hmap is None and ax_variability is None and ax_dimension is None:
		ax_is_none = True
		axes = plt.subplots(nrows=2, ncols=2)
		ax_hmap = axes[0][0]
		ax_dimension = axes[0][1]
		ax_variability = axes[1][0]

	configs = np.array(configs)
	configs = np.transpose(configs)

	for i,(m,M) in enumerate(bounds):
		configs[i] = (configs[i] - m) / (M - m)

	configs = np.diff(configs)
	abs_configs_diff = np.abs(configs)
	grad_l1 = np.sum(abs_configs_diff, axis=0)
	dim_all_var = np.sum(abs_configs_diff, axis=1)

	if ax_hmap is not None:
		sb.heatmap(configs, cmap='bwr', vmin=-2, vmax=2, ax=ax_hmap)

	if ax_variability is not None:
		ax_variability.plot(range(configs.shape[1]), ema(grad_l1, alpha))
		ax_variability.bar(range(configs.shape[1]), grad_l1, alpha=0.2)

	if ax_dimension is not None:
		ax_dimension.barh(range(configs.shape[0]), dim_all_var)
		ax_dimension.invert_yaxis()

	if ax_is_none:
		if path is None:
			plt.show()
		else:
			plt.savefig(path, bbox_inches='tight')

def showIntegralQualityEvolution(simulator, configs, rewards, idx_cell, bounds):
	best_config = []
	for b_list in bounds:
		conf_list = []
		for b in b_list:
			conf_list.append(b[0])
		best_config.append(conf_list)
	
	# best_config = configs[int(np.argmax(rewards))]

	radii = np.arange(0.05, 1, 0.05)
	tx_inner = 30.0
	tx_outter = 30.0

	config_idx_cell = 2*idx_cell
	for radius in radii:
		print(f"== RADIUS {radius} ==")
		best_config[config_idx_cell][0] = tx_inner
		best_config[config_idx_cell][1] = radius * bounds[config_idx_cell][1][1]
		best_config[config_idx_cell+1][0] = tx_outter
		simulator.setGlobalConfiguration(best_config)
		quality, bandwidths, areas = simulator.simulate(mode="integration", specific_cells=[idx_cell])
		print(quality)
		print(bandwidths)
		print(areas)
		metric = areaCapacities(quality, bandwidths, areas)
		print(metric)

		print("STOP\n")

	# for 

def extractBestConfiguration(configs, rewards):
	idx_max = int(np.argmax(rewards))
	return configs[idx_max], rewards[idx_max]

def showManyBestConfigurations(simulators, best_configs, types, names):
	vmin, vmax = None, None
	for t,best_config in zip(types, best_configs):
		simulators[t].setGlobalConfiguration(best_config)
		m = simulators[t].basicUserMetric()
		min_chal = min([min([min(u[1]) if len(u[1]) > 0 else 1000000000000 for u in v]) for v in m])
		max_chal = max([max([max(u[1]) if len(u[1]) > 0 else -1000000000000 for u in v]) for v in m])
		if vmin is None or min_chal < vmin:
			vmin = min_chal
		if vmax is None or max_chal > vmax:
			vmax = max_chal

	# Dimensions of the plot
	n = len(best_configs)
	fig_w = 14.2
	fig_h = 8.0
	ratio = fig_w / fig_h
	nrows = int(round(np.sqrt(n / ratio)))
	ncols = int(np.ceil(n / nrows))

	print(nrows, ncols)

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
	fig.set_size_inches(fig_w, fig_h)
	for i,(t,best_config) in enumerate(zip(types, best_configs)):
		x = i % ncols
		y = int(np.floor(i / ncols))
		ax = axes[y][x] if nrows > 1 else axes[x]
		simulators[t].setGlobalConfiguration(best_config)
		simulators[t].capacityHeatmap(ax=ax, fig=fig, card=True, vmin=vmin, vmax=vmax)
		ax.set_title(f"α = {names[i]}")
		print(f"Figure {i} at {x, y} drawn")

	plt.tight_layout()
	plt.show()

def showBestConfiguration(simulator, configs, rewards, criterion=None):
	best_config = extractBestConfiguration(configs, rewards)[0]

	simulator.setGlobalConfiguration(best_config)
	simulator.simulate()
	simulator.draw()

def extractObjectives(configured_simulators, alpha, acceptance_zone, fair_crit="jain"):
	xx = []
	yy = []
	for i,s in enumerate(configured_simulators):
		metrics = s.basicUserMetric(scheduled=True, zone=acceptance_zone)

		# Cumsum
		cumsum = 0
		cumsum_square = 0
		true_rew = 0
		min_metric = 1000000000
		n = 0
		i = 0
		for cell_metric in metrics:
			for _,area_metric in cell_metric:
				if area_metric != []:
					min_challenger = min(area_metric)
					min_metric = min(min_challenger, min_metric)
					for metric in area_metric:
						cumsum += metric
						cumsum_square += (metric * metric)
						n += 1
				i += 1

		mean_capacities = cumsum / len(metrics)
								
		# Fairness
		fair_criterion = 0
		if fair_crit == "jain":
			fair_criterion = cumsum * cumsum / (n * cumsum_square)
		elif fair_crit == "min":
			fair_criterion = min_metric
		# print(f"{types[i]} Config: {config} // Cumsum: {cumsum} // Fairness: {fair_criterion} // Sample: {sum(s.sample())}")
		xx.append(fair_criterion)
		yy.append(mean_capacities)

	return xx, yy

def showFairCumsumSpace(simulators, configs, types=None, legend="", fair_crit="jain"):
	xx, yy = extractObjectives(simulators, configs, fair_crit=fair_crit)

	if types is not None:
		xx = np.array(xx)
		yy = np.array(yy)
		types = np.array(types)
		markertypes = ['o', '^', 's', 'p', 'P', 'h', 'X', 'D']
		for i,t in enumerate(np.unique(types)):
			indices = np.where(types == t)
			plt.scatter(xx[indices], yy[indices], marker=markertypes[i], label=legend + str(t))
			plt.legend()
	else:
		plt.scatter(xx, yy)

	plt.xlabel(f"Fairness ({fair_crit})")
	plt.ylabel("Cumulative Sum")
	plt.show()

def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def showManyParetoFronts(sets_of_points, labels, criteria, bmf, ylabel=False, legend=False, ax=None, filepath=None):
	# For each set of points, extract the pareto front
	ax_is_none = ax is None
	if ax_is_none:
		_, ax = plt.subplots()
	for j,points_for_alpha in enumerate(sets_of_points):
		to_plot = []
		for i,(xx, yy) in enumerate(points_for_alpha):
			data = np.concatenate((np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1))), axis=1)
			pareto_front_idx = is_pareto_efficient_dumb(data)
			pareto_front = data[pareto_front_idx]
			sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
			to_plot.append((sorted_front, criteria[i]))
		
		xx, yy = [], []
		for k,(front_to_plot, alpha) in enumerate(to_plot):
			for xy in front_to_plot:
				xx.append(xy[0])
				yy.append(xy[1])
				ax.annotate(str(alpha), (xy[0], xy[1]))

		ax.scatter(xx, yy, marker=markers[j], s=150, label=labels[j])

	ax.set_xlabel("Fairness (Jain's Index)")
	if ylabel:
		ax.set_ylabel("Average Sum Rate per Cell (Mbps)")
	ax.set_title(f"Bmf Gain = {bmf} dB")
	if ax_is_none or legend:
		ax.legend(ncol=2, labelspacing=0.25, handletextpad=0.5, columnspacing=0.5)
	if filepath is None and ax_is_none:
		plt.show()
	elif filepath is not None:
		plt.savefig(filepath, bbox_inches="tight")
		plt.clf()

def animateOptimization(simulator, configs, rewards, alpha, bounds, path, dynamic=False, scale=None):
	fig = plt.figure(figsize=(16, 10))
	ax_sim = plt.subplot2grid((2, 3), (0, 0))
	ax_hm = plt.subplot2grid((2, 3), (0, 1))
	ax_rew = plt.subplot2grid((2, 3), (0, 2))
	if scale is not None:
		ax_rew.set_yscale(scale)
	ax_dist = plt.subplot2grid((2, 3), (1, 0), colspan=2)
	ax_evol = plt.subplot2grid((2, 3), (1, 2))
	normalized_configs = [normalize(config, bounds) for config in configs]
	
	half_areas = int(len(configs[0]) / 2)
	tx_inner = [inner_tx(conf) for conf in configs]
	tx_outter = [outter_tx(conf) for conf in configs]
	normalized_tx_inner = [inner_tx(conf) for conf in normalized_configs]
	normalized_tx_outter = [outter_tx(conf) for conf in normalized_configs]
	if not dynamic:
		normalized_rad_inner = [inner_rad(conf) for conf in normalized_configs]
	stx_inner = np.sum(normalized_tx_inner, axis=1) / half_areas
	stx_outter = np.sum(normalized_tx_outter, axis=1) / half_areas
	if not dynamic:
		srad_inner = np.sum(normalized_rad_inner, axis=1) / half_areas

	xx = range(len(rewards))

	p1, = ax_rew.plot(xx, ema(rewards, alpha))
	barcollection = ax_rew.bar(xx, rewards, alpha=0.2)
	for b in barcollection:
		b.set_height(0)
	ax_rew.set_xlabel("Optimization step")
	ax_rew.set_ylabel("Global reward")

	ax_dist.set_ylim([0, half_areas])
	ax_dist.set_xlim([5, 50])
	ax_dist.hist(tx_inner[-1], bins=range(9, 47), label="Inner areas")
	ax_dist.hist(tx_outter[-1], bins=range(9, 47), label="Outter areas")

	p2, = ax_evol.plot(xx, ema(stx_inner, alpha), label="Normalized average inner TX")
	p3, = ax_evol.plot(xx, ema(stx_outter, alpha), label="Normalized average outter TX")
	if not dynamic:
		p4, = ax_evol.plot(xx, ema(srad_inner, alpha), label="Normalized average inner radius")
	ax_evol.set_xlabel("Optimization step")
	ax_evol.legend()

	simulator.draw(ax=ax_sim, ax_heatmap=ax_hm, fig=fig)

	def animate(i):
		print(i)
		simulator.setGlobalConfiguration(configs[i])
		# simulator.simulate()
		simulator.draw(ax=ax_sim, ax_heatmap=ax_hm, fig=None)

		p1.set_data(xx[:i+1], ema(rewards[:i+1], alpha))
		barcollection[i].set_height(rewards[i])

		ax_dist.clear()
		ax_dist.set_ylim([0, int(len(configs[0]) / 2)])
		ax_dist.set_xlim([5, 50])
		ax_dist.hist(tx_inner[i], bins=range(9, 47), label="Inner areas")
		ax_dist.hist(tx_outter[i], bins=range(9, 47), label="Outter areas")
		ax_dist.set_xlabel("dBm")
		ax_dist.set_ylabel("Number of Areas")
		ax_dist.legend()

		p2.set_data(xx[:i+1], ema(stx_inner[:i+1], alpha))
		p3.set_data(xx[:i+1], ema(stx_outter[:i+1], alpha))
		if not dynamic:
			p4.set_data(xx[:i+1], ema(srad_inner[:i+1], alpha))

		return p1, barcollection[i]

	anim = animation.FuncAnimation(fig, animate, frames=min(300, len(rewards)))
	anim.save(path, fps=4, codec='libx264')