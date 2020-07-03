from scipy.constants import c, Planck
import numpy as np


class Line(object):
    def __init__(self, line_dict, fiber_type='SMF'):
        self._label = line_dict['label']
        self._length = line_dict['length']*1e3
        self._Nch = line_dict['Nch']
        self._amplifiers = int(np.ceil(self._length / 80e3))
        self._span_length = self._length / self._amplifiers
        self._state = ['free'] * 10
        self._successive = {}
        self._gain = 20
        self._noise_figure = 5

        # Physical parameters of the fiber
        self._alpha = 4.6e-5
        self._gamma = 1.27e-3
        # beta --> dispersion: the spreading out of a light pulse in time as it propagates down the fiber
        if fiber_type == 'LEAF':
            self._beta = 6.58e-27
        else:
            self._beta = 21.27e-27

        # Set Gain to transparency
        self._gain = self.transparency()

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        state = [s.lower().strip() for s in state]

        if set(state).issubset({'free', 'occupied'}):
            self._state = state
        else:
            print('ERROR: line state  not recognized.Value: ', set(state) - {'free', 'occupied'})

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    @property
    def amplifiers(self):
        return self._amplifiers

    @property
    def span_length(self):
        return self._span_length

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def noise_figure(self):
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self, noise_figure):
        self._noise_figure = noise_figure

    @property
    def Nch(self):
        return self._Nch

    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency

    def noise_generation(self, lightpath):
        noise = self.ase_generation() + self.nli_generation(lightpath.signal_power, lightpath.rs, lightpath.df)
        return noise

    def propagate(self, lightpath, occupation=False):

        # Update latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)

        # Update noise
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)

        # Update SNR
        snr = lightpath.signal_power / noise
        lightpath.update_snr(snr)

        # Update line state
        if occupation:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = 'occupied'
            self.state = new_state

        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)

        return lightpath

    def ase_generation(self):
        """
        Amplified spontaneous emission --> added after every amplifier: ILA(inline amplifier), Booster, Preamp
        amplified spontaneous emission calculation for a line
        :return: ase noise resulting from amplifying
        """
        gain_lin = 10 ** (self._gain / 10)
        noise_figure_lin = 10 ** (self._noise_figure / 10)
        n_amplifiers = self._amplifiers
        f = 193.4e12
        h = Planck
        Bn = 12.5e9
        ase_noise = n_amplifiers * h * f * Bn * noise_figure_lin * (gain_lin - 1)

        return ase_noise

    def nli_generation(self, signal_power, Rs, df):
        """
        non linear interference calculation: a Gaussian noise
        generated along the fiber span
        Kerr effect
        :param signal_power:
        :param Rs:
        :param df:
        :return:
        """

        Pch = signal_power
        Bn = 12.5e9
        loss = np.exp(- self.alpha * self.span_length)
        # num of spans = num of amps
        N_spans = self.amplifiers
        eta = 16 / (27 * np.pi) * np.log(
            np.pi ** 2 * self.beta * Rs ** 2 * self.Nch ** (2 * Rs / df) / (2 * self.alpha)) * self.gamma ** 2 / (
                          4 * self.alpha * self.beta * Rs ** 3)

        nli_noise = N_spans * (Pch ** 3 * loss * self.gain * eta * Bn)
        return nli_noise

    def transparency(self):
        gain = 10 * np.log10(np.exp(self.alpha * self.span_length))
        return gain
