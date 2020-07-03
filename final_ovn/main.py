from network import Network
from analysis import MonteCarloAnalysis
import analysis
from montecarlo import MonteCarloSim

network_small = Network('nodes_2.json', nch=10, fiber_type='LEAF', upgrade_line='FE')
network_big = Network('nodes_big.json', nch=10)
network_mesh = Network('nodes_mesh.json', nch=10)

# specify network
network = network_small

# draw network
network.draw()

# # simulate network
mc = MonteCarloSim(num_realizations=5, network=network)
mc.simulate(300, 7)
# at 11-14 the network_mesh gets congested
#
#

# analyze the small network
mc_analysis = MonteCarloAnalysis(mc)

mc_analysis.plot_bit_rate_matrix()
mc_analysis.plot_lines_occupation()
mc_analysis.draw_network_occupation()

# print results
mc_analysis.print_stats()
print('line to be upgraded: ' + mc_analysis.find_line_to_upgrade())

