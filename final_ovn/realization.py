import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import analysis


class Realization:
    def __init__(self, network, streamed_connections):
        self._network = network
        self._streamed_connections = streamed_connections

        rbl = []
        snrs = []
        rbc = []

        for connection in streamed_connections:
            # snr
            snrs.extend(connection.snr)
            # rbl
            for lightpath in connection.lightpaths:
                rbl.append(lightpath.bitrate)
            # rbc
            rbc.append(connection.calculate_capacity())

        self._rbl = rbl
        self._snrs = snrs
        self._rbc = rbc

    @property
    def rbl(self):
        return self._rbl

    @property
    def rbc(self):
        return self._rbc

    @property
    def snrs(self):
        return self._snrs

    @property
    def network(self):
        return self._network

    @property
    def lines(self):
        return self.network.lines

    @property
    def lines_states(self):
        return self.network

    @property
    def streamed_connections(self):
        return self._streamed_connections

    def plot_SNR_dist(self):
        plt.hist(self.snrs, bins=10)
        plt.title('SNR Distribution')
        plt.show()

    def plot_bitrate_dist(self):
        plt.hist(self.rbc, bins=10)
        plt.title('Bitrate Distribution [Gbps]')
        plt.show()

    def plot_lightpath_capacity_dist(self):
        plt.hist(self.rbl, bins=10)
        plt.title('Lightpaths Capacity Distribution [Gbps]')
        plt.show()

    def plot_connection_capacity_dist(self):
        plt.hist(self.rbc, bins=10)
        plt.title('Connection Capacity Distribution [Gbps]')
        plt.show()

    def print_stats(self):
        # print('Total Capacity Connections: {:.2f} Tbps'.format(np.sum(self.rbc) * 1e-3))
        # print('Total Capacity Lightpaths: {:.2f} Tbps'.format(np.sum(self.rbl) * 1e-3))
        print('Total Capacity: {:.2f} Tbps '.format(np.sum(self.rbc) * 1e-3))
        print('Avg Capacity: {:.2f} Gbps '.format(np.mean(self.rbc)))
        print('Avg SNR: {:.2f} dB'.format(np.mean(list(filter(lambda x: x != 0, self.snrs)))))

    def bit_rate_matrix(self):
        node_labels = list(self.network.nodes.keys())
        s = pd.Series(data=[0.0] * len(node_labels), index=node_labels)
        df = pd.DataFrame(0.0, index=s.index, columns=s.index, dtype=s.dtype)

        for connection in self.streamed_connections:
            df.loc[connection.input_node, connection.output_node] = connection.bitrate

        return df

    def plot_bit_rate_matrix(self):
        matrix = self.bit_rate_matrix()
        analysis.plot_3d_bars(matrix.values)

    def blocking_ratio(self):
        return len([c for c in self.streamed_connections if c.bitrate == 0.0])/len(self.streamed_connections)



