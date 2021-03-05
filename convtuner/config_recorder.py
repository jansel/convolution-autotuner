from opentuner import ConfigurationManipulator, PermutationParameter, IntegerParameter
from opentuner.search.manipulator import BooleanParameter
from opentuner.search.manipulator import FloatParameter
from opentuner.search.manipulator import PowerOfTwoParameter
from opentuner.search.manipulator import EnumParameter

from .utils import Once


class ConfigRecorder(object):
    def __init__(self, manipulator: ConfigurationManipulator):
        super(ConfigRecorder, self).__init__()
        self.manipulator = manipulator
        self.first = Once()

    def permutation(self, name, items):
        if self.first(name):
            self.manipulator.add_parameter(PermutationParameter(name, items))
        return items

    def power_of_two(self, name, min_value, max_value):
        if self.first(name):
            self.manipulator.add_parameter(PowerOfTwoParameter(name, min_value, max_value))
        return min_value

    def integer(self, name, min_value, max_value):
        if self.first(name):
            self.manipulator.add_parameter(IntegerParameter(name, min_value, max_value))
        return min_value

    def float(self, name, min_value, max_value):
        if self.first(name):
            self.manipulator.add_parameter(FloatParameter(name, min_value, max_value))
        return min_value

    def boolean(self, name):
        if self.first(name):
            self.manipulator.add_parameter(BooleanParameter(name))
        return False

    def enum(self, name, items):
        if self.first(name):
            self.manipulator.add_parameter(EnumParameter(name, items))
        return items[0]


class ConfigProxy(object):
    def __init__(self, config: dict):
        super(ConfigProxy, self).__init__()
        self.config = config

    def get(self, name, *args, **kwargs):
        return self.config[name]

    permutation = get
    power_of_two = get
    integer = get
    boolean = get
    float = get
    enum = get


class DummyConfig(object):
    def permutation(self, name, items):
        return items

    def power_of_two(self, name, min_value, max_value):
        return min_value

    def integer(self, name, min_value, max_value):
        return min_value

    def boolean(self, name):
        return False

    def float(self, name, min_value, max_value):
        return min_value

    def enum(self, name, items):
        return items[0]
