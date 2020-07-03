class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath.signal_power = lightpath.optimized_powers[line_label]
            lightpath = line.propagate(lightpath, occupation)

        return lightpath

    def optimize(self, lightpath):
        """
        a recursive func used to calculate GSNR along a lightpath
        performs the role of the OLS controller
        :param lightpath: ligthpath along which GSNR is being calculated
        :return: optimized ligthpath
        """
        path = lightpath.path

        # not the last node
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            # calculate line noise
            ase = line.ase_generation()
            # eta = P_NLI/P_CH^3
            eta = line.nli_generation(1, lightpath.rs, lightpath.df)
            # calculate optimum power
            p_opt = (ase / (2 * eta)) ** (1 / 3)
            # optimum power update for line
            lightpath.optimized_powers.update({line_label: p_opt})
            lightpath.next()
            node = line.successive[lightpath.path[0]]
            # recursively move to next nodes
            lightpath = node.optimize(lightpath)

        return lightpath

