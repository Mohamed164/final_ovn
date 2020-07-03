class Lightpath(object):
    def __init__(self, path, channel, power=None, Rs=32e9, df=50e9, transceiver= 'shanon'):
        self._signal_power = power
        self._path = path
        self._channel = channel
        self._Rs = Rs
        self._df = df
        self._noise_power = 0
        self._latency = 0
        self._snr = None
        self._optimized_powers = {}
        self._transceiver = transceiver
        self._bitrate = None

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, power):
        self._signal_power = power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def channel(self):
        return self._channel

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @property
    def rs(self):
        return self._Rs

    @property
    def df(self):
        return self._df

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):
        self.noise_power += noise

    def add_latency(self, latency):
        self.latency += latency

    def next(self):
        self.path = self.path[1:]

    @property
    def optimized_powers(self):
        return self._optimized_powers

    @optimized_powers.setter
    def optimized_powers(self, optimized_powers):
        self._optimized_powers = optimized_powers

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    def update_snr(self, snr):
        if self.snr is None:
            self.snr = snr
        else:
            self.snr = 1 / (1 / self.snr + 1 / snr)

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

