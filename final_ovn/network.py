import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from node import Node
from line import Line
from lightpath import Lightpath
from scipy.special import erfcinv
from path import Path


def calculate_bitrate(lightpath, bert=1e-3, bn=12.5e9):
    """
    calculate bitrate along a lightpath depending on the used transceiver
    saves the calculated bitrate to lightpath
    :param lightpath:
    :param bert:
    :param bn:
    :return:
    """
    snr = lightpath.snr
    rs = lightpath.rs
    rb = None

    if lightpath.transceiver.lower() == 'fixed-rate':
        # fixed-rate transceiver --> PM-QPSK modulation
        snrt = 2 * erfcinv(2 * bert) * (rs / bn)
        rb = np.piecewise(snr, [snr < snrt, snr >= snrt], [0, 100])

    elif lightpath.transceiver.lower() == 'flex-rate':
        snrt1 = 2 * erfcinv(2 * bert) ** 2 * (rs / bn)
        snrt2 = (14 / 3) * erfcinv(3 / 2 * bert) ** 2 * (rs / bn)
        snrt3 = (10) * erfcinv(8 / 3 * bert) ** 2 * (rs / bn)

        cond1 = (snr < snrt1)
        cond2 = (snrt1 <= snr < snrt2)
        cond3 = (snrt2 <= snr < snrt3)
        cond4 = (snr >= snrt3)

        rb = np.piecewise(snr, [cond1, cond2, cond3, cond4], [0, 100, 200, 400])

    elif lightpath.transceiver.lower() == 'shannon':
        rb = 2 * rs * np.log2(1 + snr * (rs / bn)) * 1e-9

    lightpath.bitrate = float(rb)
    return float(rb)


class Network(object):
    def __init__(self, json_path, nch=10, upgrade_line='', fiber_type='SMF'):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._connected = False
        self._weighted_paths = None
        self._route_space = None
        self._nch = nch
        self._all_paths = None
        self._json_path = json_path
        self._upgrade_line = upgrade_line
        self._fiber_type = fiber_type

        # loop through all nodes
        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))
                line_dict['Nch'] = self.nch

                self._lines[line_label] = Line(line_dict, fiber_type= fiber_type)

        # upgrade a line by decreasing by 3 dB the noise figure of the amplifiers along it
        if not upgrade_line == '':
            self.lines[upgrade_line].noise_figure = self.lines[upgrade_line].noise_figure - 3


    @property
    def nodes(self):
        return self._nodes

    @property
    def nch(self):
        return self._nch

    @property
    def lines(self):
        return self._lines

    def draw(self):
        if not self.connected:
            self.connect()

        nodes = self.nodes

        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0,'go', markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)

            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')

        plt.title('Network')
        plt.show()

    def find_paths_wrong(self, label1, label2, available_only=False):

        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {'0': label1}

        # generate all possible combinations of paths
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node for cross_node in cross_nodes if (
                        (inner_path[-1] + cross_node in cross_lines) & (cross_node not in inner_paths))]

        # filtered based on existing/available
        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)

        return paths

    def find_paths(self, node1_label, node2_label):

        path = Path(node1_label, node2_label)
        paths = []

        self.find_paths_from(node1_label, path, paths)

        return paths

    def find_paths_from(self, current_node_label, path, paths):
        """
        :param current_node_label: node to start from
        :param path: current path
        :param paths: all paths found so far
        :return: updated paths list
        """
        current_node = self.nodes[current_node_label]

        for connected_node in current_node.connected_nodes:
            # avoid loops
            if connected_node == path.start_node or connected_node in path.path_string:
                continue

            line = current_node_label + connected_node
            if line in self.lines:

                if connected_node != path.end_node:
                    # continue along the path
                    npath = Path(path.start_node, path.end_node)
                    npath.path_string = path.path_string + connected_node
                    self.find_paths_from(connected_node, npath, paths)
                else:
                    # add path to list
                    paths.append(path.path_string + connected_node)

        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        self._connected = True

        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]

    def propagate(self, lightpath, occupation=False):
        start_node = self.nodes[lightpath.path[0]]

        propagated_lightpath = start_node.propagate(lightpath, occupation)

        return propagated_lightpath

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space

    @property
    def connected(self):
        return self._connected

    def set_weighted_paths(self):
        """

        :return:
        """
        if not self.connected:
            self.connect()

        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []

        for pair in self.node_pairs():
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                # Propagation
                lightpath = Lightpath(path=path, channel=0)
                self.optimization(lightpath)
                self.propagate(lightpath, occupation=False)

                latencies.append(lightpath.latency)
                noises.append(lightpath.noise_power)
                snrs.append(10 * np.log10(lightpath.snr))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs

        self._weighted_paths = df

        route_space = pd.DataFrame()
        route_space['path'] = paths

        # wavelength availability matrix
        for i in range(self.nch):
            route_space[str(i)] = ['free'] * len(paths)
        self._route_space = route_space

    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        return best_path

    def find_best_latency(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)

        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0]
        else:
            best_path = None

        return best_path

    def stream(self, connections, best='latency', transceiver='shannon'):
        streamed_connections = []

        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node

            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('ERROR: best input not recognized.Value:', best)
                continue

            if path:
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                channel = [i for i in range(len(path_occupancy)) if path_occupancy[i] == 'free'][0]
                path = path.replace('->', '')

                connection.full_path = path

                # calculate GSNR
                in_lightpath = Lightpath(path, channel, transceiver=transceiver)
                in_lightpath = self.optimization(in_lightpath)
                out_lightpath = self.propagate(in_lightpath, True)

                # bitrate depending on transceiver technology
                calculate_bitrate(out_lightpath)

                if out_lightpath.bitrate == 0.0:
                    # [self.update_route_space(path, channel, 'free') for lp in connection.lightpaths]
                    self.update_route_space(path, channel, 'free')
                    connection.block_connection()

                else:
                    connection.set_connection(out_lightpath)
                    self.update_route_space(path, channel, 'occupied')

                    if connection.residual_rate_request > 0:
                        self.stream([connection], best, transceiver)
            else:
                # [self.update_route_space(path, channel, 'free') for lp in connection.lightpaths]
                connection.block_connection()

            streamed_connections.append(connection)

        return streamed_connections

    def available_paths(self, input_node, output_node):
        if self.weighted_paths is None:
            self.set_weighted_paths()

        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == input_node) and (path[-1] == output_node))]
        available_paths = []

        for path in all_paths:
            path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
            if 'free' in path_occupancy:
                available_paths.append(path)

        return available_paths

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    @property
    def all_paths(self):
        if self._all_paths is None:
            self._all_paths = [self.path_to_line_set(p) for p in self.route_space.path.values]
        return self._all_paths

    def update_route_space(self, path, channel, state):
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)

        for i in range(len(self.all_paths)):
            line_set = self.all_paths[i]
            if lines.intersection(line_set):
                states[i] = state

        self.route_space[str(channel)] = states

    def optimization(self, lightpath):
        """
        OLS controller
        find optimal channel power (for ---> ASE + NLI)

        :param lightpath: lightpath to calculate the GSNR of
        :return: optimized lightpath
        """
        path = lightpath.path

        start_node = self.nodes[path[0]]
        optimized_lightpath = start_node.optimize(lightpath)

        # path changes with recursion, redefine it
        optimized_lightpath.path = path

        return optimized_lightpath

    def node_pairs(self):
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)

        return pairs

    def make_new_network(self):
        return Network(self._json_path, self.nch, upgrade_line=self._upgrade_line, fiber_type=self._fiber_type)
