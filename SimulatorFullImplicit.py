from fullimplicit import Simulator1DIMPLICIT
from impes1d import Simulator1DIMPES
from conversion import daysToSeconds,secondsToDays
import numpy as np
import matplotlib.pyplot as plt

simulator = Simulator1DIMPLICIT(50,200.0)
simulator_IMPES = Simulator1DIMPES(50,200.0)

times = [daysToSeconds(100), daysToSeconds(200), daysToSeconds(300), daysToSeconds(400), daysToSeconds(500)]
pressures = []
saturation = []
pressures_IMPES = []
saturation_IMPES = []
for time in times:
    simulator.simulateTo(time)
    simulator_IMPES.simulateTo(time)
    saturation.append(simulator.saturation)
    pressures.append(simulator.pressure)
    saturation_IMPES.append(simulator.saturation)
    pressures_IMPES.append(simulator.pressure)
cell_lengths = np.linspace(0, simulator.length, simulator.Ncells)
cell_lengths_IMPES = np.linspace(0, simulator_IMPES.length, simulator_IMPES.Ncells)

cmap = plt.get_cmap('plasma')
for i, (time, sat, sat_IMPES) in enumerate(zip(times, saturation, saturation_IMPES)):
    plt.plot(cell_lengths,sat,label=r'Full Implicit', color= cmap(1-i/len(times)))
    plt.plot(cell_lengths_IMPES,sat_IMPES,label=r'IMPES', linestyle='dashed', color= cmap(1-i/len(times)))
plt.title('Saturation Profiles at Different Times')
plt.xlabel('Distance [m]')
plt.ylabel('Saturation')
plt.legend()
plt.show()

# Plotting pressure
for i, (time, pressure, pressure_IMPES) in enumerate(zip(times, pressures, pressures_IMPES)):
    plt.plot(cell_lengths, pressure,label=r'Full Implicit', color= cmap(1-i/len(times)))
    plt.plot(cell_lengths_IMPES,pressure_IMPES,label=r'IMPES', linestyle='dashed', color= cmap(1-i/len(times)))
plt.title('Pressure Profiles at Different Times')
plt.xlabel('Distance[m]')
plt.ylabel('Pressure')
plt.legend()
plt.show()