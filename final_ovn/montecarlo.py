import numpy as np
import pandas as pd
from random import shuffle
from connection import Connection
import analysis
from realization import Realization


class MonteCarloSim(object):
    def __init__(self, num_realizations, network):
        self._rbcs = []
        self._rbls = []
        self._snrs = []
        self._connections = []
        self._lines_states = []
        self._lines = []
        self._num_realizations = num_realizations
        self._realizations = []
        self._network = network

    def simulate(self, rate, multiplier):
        """
        :param rate: rate at which requests are made (uniform for all nodes)
        :param multiplier: for data rate
        :return: None
        """
        realizations = []
        for i in range(self.num_realizations):

            # create a matching network
            network = self.network.make_new_network()

            # change order of connections by shuffling pairs
            pairs = network.node_pairs().copy()
            shuffle(pairs)

            # create a connection for every random pair
            traffic_matrix = analysis.create_traffic_matrix(list(network.nodes.keys()), rate, multiplier=multiplier)
            connections = []
            for node_pair in pairs:
                connection = Connection(node_pair[0], node_pair[1],
                                        rate_request=float(traffic_matrix.loc[node_pair[0], node_pair[1]]))
                connections.append(connection)

            # stream created connections
            streamed_connections = network.stream(connections, best='snr')

            # create realization
            realization = Realization(network, streamed_connections)
            realizations.append(realization)

        self._realizations = realizations

    @property
    def network(self):
        return self._network

    @property
    def num_realizations(self):
        return self._num_realizations

    @property
    def realizations(self):
        return self._realizations

    @property
    def rbls(self):
        return [realization.rbl for realization in self.realizations]

    @property
    def rbcs(self):
        """
        :return: a list of lists each containing the rbcs of every connection of a single MonteCarlo realization
        """
        return [realization.rbc for realization in self.realizations]

    @property
    def snrs(self):
        return [realization.snrs for realization in self.realizations]

    @property
    def connections(self):
        return [realization.streamed_connections for realization in self.realizations]

    def avg_snr(self):
        return np.mean([np.mean(snr) for snr in self.snrs])

    def avg_rbl(self):
        """
        :return: avg lightpath bitrate in Gbps
        """
        return np.mean([np.mean(rbl) for rbl in self.rbls])

    def avg_rbc(self):
        return np.mean(self.rbcs)

    def total_capacity(self):
        """
        :return: total capacity of the network in Tbps
        """
        return np.mean([np.sum(rbl) for rbl in self.rbls]) * 1e-3

    def lines(self):
        return [realization.lines for realization in self.realizations]

    def lines_congestion(self):
        df = pd.DataFrame(self.lines())
        df = df.applymap(lambda x: x.state.count('occupied')/len(x.state))

        return df.mean()

