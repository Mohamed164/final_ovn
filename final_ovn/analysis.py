import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import montecarlo


def create_traffic_matrix(nodes, rate, multiplier=1):
    """
    :param nodes: nodes of the network
    :param rate: rate at which requests are made (more rate --> more congestion)
    :param multiplier: is an integer number multiplying the values of the traffic matrix in order to
    increase the number of lightpaths that need to be allocated for each connection request between a node pair
    :return: a matrix in which every cell indicates amount of traffic to be conveyed between two nodes
    """

    # increasing the multiplier value will make the network more congested and the lines more occupied
    s = pd.Series(data=[0.0] * len(nodes), index=nodes)
    df = pd.DataFrame(float(multiplier * rate), index=s.index, columns=s.index, dtype=s.dtype)
    np.fill_diagonal(df.values, 0.0)

    return df


def plot_3d_bars(t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data = np.meshgrid(np.arange(t.shape[1]), np.arange(t.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = t.flatten()
    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)
    plt.show()


class MonteCarloAnalysis(object):
    def __init__(self, mc):
        self._mc = mc

    def plot_lines_occupation(self):
        plt.bar(range(len(self._mc.lines_congestion())), height=self._mc.lines_congestion().values)
        plt.xticks(range(len(self._mc.lines_congestion())), self._mc.lines_congestion().keys())
        plt.title('lines occupation')
        plt.show()

    def find_line_to_upgrade(self):
        return self._mc.lines_congestion().sort_values(ascending=False).keys()[0]

    def print_stats(self):
        print('Total Capacity: {:.2f} Tbps '.format(self._mc.total_capacity()))
        print('Avg Capacity: {:.2f} Gbps '.format(self._mc.avg_rbl()))
        print('Avg SNR: {:.2f} dB'.format(self._mc.avg_snr()))

    def bit_rate_matrix(self):
        node_labels = list(self._mc.network.nodes.keys())
        s = pd.Series(data=[0.0] * len(node_labels), index=node_labels)
        df = pd.DataFrame(0.0, index=s.index, columns=s.index, dtype=s.dtype)

        connections = self._mc.connections[0]

        # averaging bitrates for connections
        bitrates = [[connection.bitrate for connection in connection_list] for connection_list in self._mc.connections]
        bitrates = np.mean(bitrates, axis=0)

        for i in range(len(connections)):
            df.loc[connections[i].input_node, connections[i].output_node] = bitrates[i]

        return df

    def plot_bit_rate_matrix(self):
        matrix = self.bit_rate_matrix()
        plot_3d_bars(matrix.values)

    def draw_network_occupation(self):
        nodes = self._mc.network.nodes
        fig, (ax1, ax2) = plt.subplots(1, 2)
        drawn = []
        fig.suptitle('Network')

        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            ax1.plot(x0, y0, 'go', markersize=10)
            ax1.text(x0 + 20, y0 + 20, node_label)
            ax2.plot(x0, y0, 'go', markersize=10)
            ax2.text(x0 + 20, y0 + 20, node_label)

            for connected_node_label in n0.connected_nodes:
                line_label = node_label + connected_node_label
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                start = [x0, x1]
                end = [y0, y1]

                if line_label in drawn:
                    ax = ax2
                else:
                    ax = ax1

                if self._mc.lines_congestion()[str(node_label + connected_node_label)] == 1.0:
                    ax.plot(start, end, 'r')
                elif self._mc.lines_congestion()[str(node_label + connected_node_label)] > 0.9:
                    ax.plot(start, end, 'y')
                else:
                    ax.plot(start, end, 'b')

                ax.text(np.mean(start), np.mean(end), line_label)
                drawn.append(line_label[::-1])

        plt.show()

    def blocking_ratio(self):
        ratios = []
        for connections_list in self._mc.connections:
            ratios.append(len([c for c in connections_list if c.bitrate == 0.0])/len(connections_list))
        return np.mean(ratios)
