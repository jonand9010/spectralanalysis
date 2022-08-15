import numpy as np

class Inverter:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def define_shift_matrix(self, time_shifts, decay_weights):
        bins = self.y.shape[0]
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
            
    def get_bin_shift(self, time_shift):
        return np.where(self.x >= time_shift)[0][0]
    
    def clean_spectrum(self):
        M_inv = np.linalg.inv(self.shift_matrix)
        cleaned_spectrum = np.dot(M_inv, self.y)
        return cleaned_spectrum
        