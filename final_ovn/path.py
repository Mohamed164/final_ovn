class Path(object):
    def __init__(self, start_node, end_node):
        self._start_node = start_node
        self._end_node = end_node
        self._path_string = start_node


    @property
    def start_node(self):
        return self._start_node

    @property
    def end_node(self):
        return self._end_node

    @property
    def path_string(self):
        return self._path_string

    @path_string.setter
    def path_string(self, path):
        self._path_string = path


