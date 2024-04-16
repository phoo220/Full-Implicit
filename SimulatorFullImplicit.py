from fullimplicit import Simulator1DIMPLICIT
from conversion import daysToSeconds,secondsToDays
import numpy as np
from corey import coreyWater, coreyOil
import matplotlib.pyplot as plt

simulator = Simulator1DIMPLICIT(10,200.0)
simulator.simulateTo(daysToSeconds(1))

""" simulator_refined = Simulator1DIMPLICIT(50,200.0)
simulator_refined.deltat = daysToSeconds(30) """

#print(simulator.pressure)
#print(simulator.saturation)
#print(simulator.distance)
#print(simulator.time)
#J = simulator.matrixJ
#print('matrixJ----', simulator.matrixJ, 'end of matrixJ')

times = [daysToSeconds(100), daysToSeconds(200), daysToSeconds(300), daysToSeconds(400),daysToSeconds(500)]
pressures = []
saturation = []
for time in times:
    simulator.simulateTo(time)
    pressures.append(simulator.pressure)  # Make a copy to avoid modifying original pressure array
    saturation.append(simulator.saturation)  # Make a copy to avoid modifying original saturation array
cell_lengths = np.linspace(0, simulator.length, simulator.Ncells)

cmap = plt.get_cmap('plasma')
for i, (time, sat) in enumerate(zip(times, saturation)):
    plt.plot(cell_lengths,sat, color= cmap(1-i/len(times)))
plt.title('Saturation Profiles at Different Times')
plt.xlabel('Distance [m]')
plt.ylabel('Saturation')
plt.show()

# Plotting pressure
for i, (time, pressure) in enumerate(zip(times, pressures)):
    plt.plot(cell_lengths, pressure, color= cmap(1-i/len(times)))
plt.title('Pressure Profiles at Different Times')
plt.xlabel('Distance[m]')
plt.ylabel('Pressure')
plt.show()