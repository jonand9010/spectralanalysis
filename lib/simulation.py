import numpy as np

class SimulateSpectra:
    def __init__(self, x):
        self.x = x
        self.peak_list = []
        
    def add_line(self, A, x0, std, linetype = 'gaussian'):
        if linetype == 'gaussian':
            self.peak_list.append(A * (np.exp(-(self.x-x0)**2 /(2*std**2) )))
        else:
            print('Select correct line type. No line added to spectrum.')
            
    def generate_spectrum(self):
        self.spectrum = np.zeros(self.peak_list[0].shape)
        for peak in self.peak_list:
            self.spectrum += peak
            
    def define_shift_matrix(self, time_shifts, decay_weights):
        bins = self.spectrum.shape[0]
        self.shift_matrix = np.eye(bins)

        bin_shifts = []
        decays = []
        for (time_shift, weight) in zip(time_shifts, decay_weights):
            bin_shifts.append(self.get_bin_shift(time_shift))
            decays.append(weight)

        for (shift, decay) in zip(bin_shifts, decays):
            for bin in range(bins):
                iter_bin = bin
                while iter_bin + shift < bins:
                    self.shift_matrix[bin, iter_bin + shift] += decay
                    iter_bin += shift
        self.shift_matrix += self.shift_matrix.T
        np.fill_diagonal(self.shift_matrix, 1)
            
    def simulate_measurements(self, samples = 10, noise = 0.5):
        X = []
        bins = self.spectrum.shape[0]
        for sample in range(samples):
            sample_spectrum = self.shift_spectrum()
            sample_spectrum += np.abs(np.random.normal(0, noise, bins))
            X.append(sample_spectrum)
        X = np.array(X)
        return X
    
    def shift_spectrum(self):
        shifted_spectrum = np.dot(self.shift_matrix, self.spectrum.T)
        return shifted_spectrum

    def get_bin_shift(self, time_shift):
        return np.where(self.x >= time_shift)[0][0]