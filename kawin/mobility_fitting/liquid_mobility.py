import numpy as np

class SpeciesData:
    def __init__(self, name, atomic_number, diameter, Tm, melting_phase = None):
        self.name = name
        self.atomic_number = atomic_number
        self.diameter = diameter
        self.Tm = Tm
        self.melting_phase = melting_phase
        self.K0 = self._getK0(melting_phase)

    def _getK0(self, melting_phase):
        if melting_phase is None:
            return 0
        
        phases = {'BCC': 1, 'HCP': 2, 'FCC': 3}
        for p in phases:
            if p in melting_phase:
                return phases[p]
        return 0
    
def compute_liquid_mobility(species = list[SpeciesData]):
    liquid_mobility_data = []
    n = len(species)
    for i in range(n):
        for j in range(n):
            Q = 0.17 * 8.314 * species[i].Tm * (16 + species[i].K0)
            D0 = (8.95 - 0.0734 * species[i].atomic_number) * 1e-8

            d_ratio = species[i].diameter / species[j].diameter
            D0 *= d_ratio
            liquid_mobility_data.append((species[i].name, species[j].name, Q, D0))
            
    return liquid_mobility_data

