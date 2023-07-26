import visualize as v
from simulator import *
from tools import getEnergy

if __name__ == "__main__":
	# Topology name
	topo_str = "Presquile_Lyon"
	# Association based on SINR
	assoc_criterion = "sinr_max"
	# Mechanisms considered
	mechanisms = ["FULL_REUSE", "COLORING", "NOMA", "FFR"]
	mechanisms_name = ["FULL REUSE", "COLORING", "NOMA+FFR", "FFR"]
	model_names = ["INSPIRE"]
	beamforming = [0.0, 3.0, 6.0]
	criteria = [0.25, 0.5, 1, 2]
	acceptance_zone = [[250, 4300], [250, 4750]]
	simulators = []
	scatters = []
	
	plt.rcParams.update({'font.size': 14})
	fig_cdf, axes_cdf = plt.subplots(nrows=len(beamforming), ncols=len(criteria), figsize=(18, 11))
	fig_topos, axes_topos = plt.subplots(nrows=1, ncols=len(criteria), figsize=(18, 5.25))
	fig_pareto, axes_pareto = plt.subplots(ncols=len(beamforming), figsize=(15, 5.25))
	max_x_pareto, min_x_pareto, max_y_pareto, min_y_pareto = None, None, None, None
	fig_energy, axes_energy = plt.subplots(ncols=len(beamforming), figsize=(15, 5.25))
	for bmf_index,bmf in enumerate(beamforming):
		sets_of_points = [[] for _ in range(len(mechanisms))]
		energies = [[] for _ in range(len(mechanisms))]
		for crit_index,criterion in enumerate(criteria):
			sim_for_cdf = []
			names_for_cdf = []
			for tech_index,tech in enumerate(mechanisms):
				print(mechanisms_name[tech_index], criterion, bmf)
				params = TopoParams(noma=tech == "NOMA", ffr=tech == "FFR", ffr_threshold=0.42, full_reuse=tech == "FULL_REUSE", beamforming=bmf)
				topo = DynamicTopology.fromFile(f"topos/{topo_str}.json", criterion, True, assoc_criterion=assoc_criterion, denormalize=False, params=params)
				noises = np.array([0.75] * topo._n)

				file_label = f"{criterion}_INSPIRE_{tech}_{bmf}"
				index = 1
				if tech == "FULL_REUSE" and (bmf == 6.0 and criterion == 2) or\
				   tech == "COLORING" and ((bmf == 0.0 and criterion == 2) or (bmf == 3.0 and criterion == 1)) or\
				   tech == "FFR" and (bmf == 0.0 and criterion == 2):
					index = 2
				print(index)
				reward_file = f"data/rewards_{topo_str}_discrete_{assoc_criterion}_{file_label}_{index}.txt"
				config_file = f"data/configs_{topo_str}_discrete_{assoc_criterion}_{file_label}_{index}.txt"

				d = v.extractBestConfiguration(v.read_file(config_file)[:110], v.read_file(reward_file)[:110])
				simulator = Simulator(topo, criterion, noises)
				simulator.setGlobalConfiguration(d[0])

				if bmf == 3.0:
					simulator.draw(ax=axes_topos[crit_index], title=f"α = {criterion}", ylabel="" if crit_index != 0 else f"Bmf Gain = {bmf} dB", rectangle=[[1600, 3100], [1400, 3000]]) # rectangle=[[1600, 3100], [1400, 3000]]
				energies[tech_index].append(getEnergy(d[0]))

				sim_for_cdf.append(simulator)
				sets_of_points[tech_index].append(v.extractObjectives([simulator], criterion, acceptance_zone))
				# Axes limits
				min_x_pareto = min(sets_of_points[tech_index][-1][0]) if min_x_pareto is None else min(min_x_pareto, min(sets_of_points[tech_index][-1][0]))
				min_y_pareto = min(sets_of_points[tech_index][-1][1]) if min_y_pareto is None else min(min_y_pareto, min(sets_of_points[tech_index][-1][1]))
				max_x_pareto = max(sets_of_points[tech_index][-1][0]) if max_x_pareto is None else max(max_x_pareto, max(sets_of_points[tech_index][-1][0]))
				max_y_pareto = max(sets_of_points[tech_index][-1][1]) if max_y_pareto is None else max(max_y_pareto, max(sets_of_points[tech_index][-1][1]))
			
			scatters = v.compare_throughput_cdfs(sim_for_cdf, f"α = {criterion}" if bmf_index == 0 else None, [mechanisms_name[tech_index] == "NOMA+FFR" for tech_index in range(len(mechanisms))], axes_cdf[bmf_index][crit_index], xlabel=f"Capacities (Mbps)" if bmf_index == len(beamforming)-1 else None, ylabel=f"Bmf Gain = {bmf} dB" if crit_index == 0 else None, zone=acceptance_zone)#, imgpath=f"img/{topo_str}_CdfCapacities_INSPIRE_{criterion}_{bmf}.png")
		# Extract Pareto front
		scatters_pareto = v.showManyParetoFronts(sets_of_points, mechanisms_name, criteria, bmf, ylabel=bmf_index==0, ax=axes_pareto[bmf_index])# f"img/{topo_str}_ParetoFronts_INSPIRE_{beam}.png")
		scatters_energy = v.showManyEnergy(energies, mechanisms_name, criteria, bmf, ylabel=bmf_index==0, ax=axes_energy[bmf_index])# f"img/{topo_str}_ParetoFronts_INSPIRE_{beam}.png")

	# Fix Pareto axes
	delta = 0.05
	delta_x = delta * (max_x_pareto - min_x_pareto)
	delta_y = delta * (max_y_pareto - min_y_pareto)
	for bmf_index in range(len(beamforming)):
		axes_pareto[bmf_index].set_xlim(min_x_pareto - delta_x, max_x_pareto + delta_x)
		axes_pareto[bmf_index].set_ylim(min_y_pareto - delta_y, max_y_pareto + delta_y)

	fig_cdf.legend(scatters, labels=mechanisms_name, ncol=len(mechanisms_name), labelspacing=0.25, handletextpad=0.5, columnspacing=0.5, loc="lower center", bbox_to_anchor=(0.5, 0.01))
	fig_cdf.savefig(f"img/{topo_str}_INSPIRE_cdfs_filtered.png", bbox_inches='tight')

	fig_pareto.legend(scatters_pareto, labels=mechanisms_name, ncol=len(mechanisms_name), labelspacing=0.25, handletextpad=0.5, columnspacing=0.5, loc="lower center", bbox_to_anchor=(0.5, -0.085))
	fig_pareto.savefig(f"img/{topo_str}_INSPIRE_paretos_filtered.png", bbox_inches='tight')
	
	fig_energy.legend(scatters_energy, labels=mechanisms_name, ncol=len(mechanisms_name), labelspacing=0.25, handletextpad=0.5, columnspacing=0.5, loc="lower center", bbox_to_anchor=(0.5, -0.085))
	fig_energy.savefig(f"img/{topo_str}_INSPIRE_energy.png", bbox_inches='tight')

	fig_topos.savefig(f"img/{topo_str}_INSPIRE_topos.png", bbox_inches='tight')