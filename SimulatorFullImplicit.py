from fullimplicit import Simulator1DIMPLICIT
from conversion import daysToSeconds,secondsToDays
import numpy as np
import matplotlib.pyplot as plt

simulator = Simulator1DIMPLICIT(10,200.0)
simulator.simulateTo(daysToSeconds(100))

print(simulator.pressure)
