import numpy as np

from kawin.Constants import AVOGADROS_NUMBER

class VolumeParameter:
    MOLAR_VOLUME = 0
    ATOMIC_VOLUME = 1
    LATTICE_PARAMETER = 2

    def __init__(self, value=None, volumeType=None, atomsPerCell=None):
        if value is None or volumeType is None or atomsPerCell is None:
            self.a = None
            self.Va = None
            self.Vm = None
            self.atomsPerCell = None
        else:
            self.setVolume(value, volumeType, atomsPerCell)
        self._updateCallbacks = []

    def setVolume(self, value, volumeType, atomsPerCell):
        '''
        Function to set lattice parameter, atomic volume and molar volume

        Parameters
        ----------
        value : float
            Value for volume parameters (lattice parameter, atomic (unit cell) volume or molar volume)
        valueType : VolumeParameter or str
            States what volume term that value is
        atomsPerCell : int
            Number of atoms in the unit cell
        '''
        self.atomsPerCell = atomsPerCell
        if volumeType == self.MOLAR_VOLUME or volumeType == 'VM':
            self.Vm = value
            self.Va = atomsPerCell * self.Vm / AVOGADROS_NUMBER
            self.a = np.cbrt(self.Va)
        elif volumeType == self.ATOMIC_VOLUME or volumeType == 'VA':
            self.Va = value
            self.Vm = self.Va * AVOGADROS_NUMBER / atomsPerCell
            self.a = np.cbrt(self.Va)
        elif volumeType == self.LATTICE_PARAMETER or volumeType == 'a':
            self.a = value
            self.Va = self.a**3
            self.Vm = self.Va * AVOGADROS_NUMBER / atomsPerCell
        else:
            valid_values = "['VM', 'VA', 'a', VolumeParameter.MOLAR_VOLUME, VolumeParameter.ATOMIC_VOLUME, VolumeParameter.LATTICE_PARAMETER]"
            raise ValueError(f'Unknown volume type {volumeType}. Values must be: {valid_values}')
        
        for callback in self._updateCallbacks:
            callback()