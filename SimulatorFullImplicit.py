from fullimplicit import Simulator1DIMPLICIT
from conversion import daysToSeconds,secondsToDays
import numpy as np
import matplotlib.pyplot as plt

simulator = Simulator1DIMPLICIT(30,200.0)
simulator.simulateTo(daysToSeconds(30))

simulator_refined = Simulator1DIMPLICIT(50,200.0)
simulator_refined.deltat = daysToSeconds(30)

print(simulator.pressure)
print(simulator.saturation)
print(simulator.time)

times = [daysToSeconds(100), daysToSeconds(200), daysToSeconds(300), daysToSeconds(400),daysToSeconds(500)]
pressures = []
saturation = []
pressures_refined = []
saturation_refined = []
for time in times:
    simulator.simulateTo(time)
    pressures.append(simulator.pressure.copy())  # Make a copy to avoid modifying original pressure array
    saturation.append(simulator.saturation.copy())  # Make a copy to avoid modifying original saturation array
    simulator_refined.simulateTo(time)
    pressures_refined.append(simulator_refined.pressure.copy())
    saturation_refined.append(simulator_refined.saturation.copy())
cell_lengths = np.linspace(0, simulator.length, simulator.Ncells)
cell_lengths_r = np.linspace(0, simulator_refined.length, simulator_refined.Ncells)

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