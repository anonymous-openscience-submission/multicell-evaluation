from simulator import Simulator
from topos import *
from inspire import *
import numpy as np
import warnings

warnings.simplefilter('ignore', Warning)

n = 110
nu = 1.5
discrete = True
denormalize = False
topo_name = "toy_example_dynamic"
assoc_criterion = "sinr_max"
model_names = ["INSPIRE"]
mechanisms = ["FULL_REUSE", "FFR", "NOMA", "COLORING"]
beamforming = [0.0, 3.0, 6.0]
# Number of replications
for nSimulation in range(10):
	# Values for alpha fairness
	for criterion in [0, 0.25, 0.5, 0.75, 1, 2]:
		# Optimizer
		for model_name in model_names:
			# Mechanisms
			for tech in mechanisms:
				# Beamforming gains
				for beam in beamforming:
					params = TopoParams(noma=tech == "NOMA", ffr=tech == "FFR", ffr_threshold=0.42, full_reuse=tech == "FULL_REUSE", beamforming=beam)
					file_label = f"{criterion}_{model_name}_{tech}_{beam}"
					file_path = f"data/rewards_{topo_name}_{'discrete' if discrete else 'continuous'}_{assoc_criterion}_{file_label}_{nSimulation + 1}.txt"
					print(file_label)

					# Get the cellular network topology
					topology = DynamicTopology.fromFile(f"topos/{topo_name}.json", criterion, discrete, assoc_criterion=assoc_criterion, denormalize=denormalize, params=params)
					# Create the simulator with the topology
					s = Simulator(topology, criterion, np.array([0.75] * topology._n))
					# Draw the cellular network
					s.draw()

					toTest = s.getConfiguration()
					selfish_rewards = s.sample()
					dimensions = s.getIntervals()

					# Instantiate INSPIRE
					model = Inspire(dimensions, s.getAdjacencies(), nu=nu)

					global_rewards = [np.sum(selfish_rewards)]
					configs = [toTest]
					exp_avg = []
					alpha = 0.125

					# Optimization loop
					for i in range(n-1):
						print(f"\n== ITERATION {i} ==")

						# Get the prediction of the model
						toTest = model.predict()
						s.setGlobalConfiguration(toTest)

						# Print data
						print("Configuration:", toTest)
						print("EUs distribution:", s._topo.getCardinalities())

						# Save the config, collect the reward
						configs.append(toTest)
						selfish_rewards = s.sample()
						print("Selfish rewards:", selfish_rewards)
						# Learn the new model
						model.tell(np.array(flatten(toTest)) if model_name != "INSPIRE" else toTest, selfish_rewards)

						# Compute and plot the global reward
						global_rew = np.sum(selfish_rewards)
						global_rewards.append(global_rew)
						if exp_avg == []:
							exp_avg = [global_rew]
						else:
							exp_avg.append(exp_avg[-1] * (1 - alpha) + alpha * global_rew)

						print("Global reward:", global_rew, "// EMA:", exp_avg[-1], "// Sum:", sum(global_rewards), "// Max:", max(global_rewards), "// Idx Max:", global_rewards.index(max(global_rewards)))

					# Save the files (reward and config) after the optimization
					print("== SAUVEGARDE DES FICHIERS ==")
					config_file = open(f"data/configs_{topo_name}_{'discrete' if discrete else 'continuous'}_{assoc_criterion}_{file_label}_{nSimulation + 1}.txt", 'w')
					for conf in configs:
						config_file.write(str(conf) + "\n")
					config_file.close()

					reward_file = open(file_path, 'w')
					for rew in global_rewards:
						reward_file.write(str(rew) + "\n")
					reward_file.close()