from enum import Enum

'''
Defines class to handle a single stopping conditions

Per iteration, these will take in a model, and check with internal members to see if stopping condition has been satisfied

If it has, then it will be set to True and the time will be recorded
'''

class Inequality (Enum):
    GREATER_THAN = 0
    LESSER_THAN = 1

class PrecipitationStoppingCondition:
    def __init__(self, condition, value, phase = None, element = None):
        self._condition = condition
        self._value = value
        self._isSatisfied = False
        self._satisfiedTime = -1
        self._phase = phase
        self._element = element
        self._modelVar = None

    def _poll(self, model, n):
        p = model.phaseIndex(self._phase)
        return getattr(model, self._modelVar)[n,p]
    
    def _testCondition(self, model):
        if self._condition == Inequality.GREATER_THAN:
            return self._poll(model, model.n) > self._value
        else:
            return self._poll(model, model.n) < self._value
    
    def testCondition(self, model):
        if not self._isSatisfied:
            self._isSatisfied = self._testCondition(model)

            if self._isSatisfied:
                if model.n > 0:
                    currVal, currTime = self._poll(model, model.n), model.time[model.n]
                    prevVal, prevTime = self._poll(model, model.n-1), model.time[model.n-1]
                    self._satisfiedTime = (currTime - prevTime) * (self._value - prevVal) / (currVal - prevVal) + prevTime
                else:
                    self._satisfiedTime = model.time[model.n]

    def isSatisfied(self):
        return self._isSatisfied
    
    def satisfiedTime(self):
        return self._satisfiedTime

class VolumeFractionCondition (PrecipitationStoppingCondition):
    def __init__(self, condition, value, phase = None):
        super().__init__(condition, value, phase = phase)
        self._modelVar = 'betaFrac'

class AverageRadiusCondition (PrecipitationStoppingCondition):
    def __init__(self, condition, value, phase = None):
        super().__init__(condition, value, phase = phase)
        self._modelVar = 'avgR'
        
class DrivingForceCondition (PrecipitationStoppingCondition):
    def __init__(self, condition, value, phase = None):
        super().__init__(condition, value, phase = phase)
        self._modelVar = 'dGs'

class NucleationRateCondition (PrecipitationStoppingCondition):
    def __init__(self, condition, value, phase = None):
        super().__init__(condition, value, phase = phase)
        self._modelVar = 'nucRate'

class PrecipitateDensityCondition (PrecipitationStoppingCondition):
    def __init__(self, condition, value, phase = None):
        super().__init__(condition, value, phase = phase)
        self._modelVar = 'precipitateDensity'

class CompositionCondition (PrecipitationStoppingCondition):
    def __init__(self, condition, value, element = None):
        super().__init__(condition, value, element = element)
        self._modelVar = 'xComp'

    def _poll(self, model, n):
        e = 0 if self._element is None else model.elements.index(self._element)
        return getattr(model, self._modelVar)[n,e]
