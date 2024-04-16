import numpy as np
from corey import coreyWater, coreyOil
from conversion import daysToSeconds,secondsToDays

class Simulator1DIMPES:
    def __init__(self,Ncells,length):
        '''
        Args:
        Ncells: Number of cells
        length: Total length [m]
        '''
        self.Ncells = Ncells
        self.length = length
        self.deltaX = length/Ncells
        self.poro = 0.2*np.ones(Ncells)
        self._perm = self.setPermeabilities(1.0E-13*np.ones(Ncells))
        self.pressure = 1.0E7*np.ones(Ncells)
        self.saturation = 0.2*np.ones(Ncells)
        self.rightPressure = 1.0E7
        self.leftDarcyVelocity = 2.315E-6 * self.poro[0]
        self.mobilityWeighting = 1.0
        self.deltat = daysToSeconds(1)
        self.time = 0.0
        self.oilViscosity = 2.0E-3
        self.waterViscosity = 1.0E-3
        self.relpermWater = coreyWater(2.0,0.4,0.2,0.2)
        self.relpermOil = coreyOil(2.0,0.2,0.2)
    
    def setPermeabilities(self,permVector):
        '''
        Set permeabilities
        Args:
        permVector: A numpy array of length
        self.Ncells with perm values
        '''
        self._perm = permVector
        self._Tran = (2.0/(1.0/self._perm[:-1]+1.0/self._perm[1:]))/self.deltaX**2
        self._TranRight = self._perm[-1]/self.deltaX**2
    
    def doTimestep(self):
        '''
        Do one time step of length self.deltat
        '''
        mobOil = self.relpermOil(self.saturation)/self.oilViscosity
        mobWater = self.relpermWater(self.saturation)/self.waterViscosity
        upW = self.mobilityWeighting
        downW = 1.0-upW
        mobOilW = mobOil[:-1]*upW + mobOil[1:]*downW
        mobWaterW = mobWater[:-1]*upW + mobWater[1:]*downW
        oilTrans = self._Tran*mobOilW
        waterTrans = self._Tran*mobWaterW
        oilTransRight = self._TranRight*mobOil[-1]
        waterTransRight = self._TranRight*mobWater[-1]
        totalTrans = oilTrans + waterTrans
        totalTransRight = oilTransRight + waterTransRight

        # ----------------------------
        # Solve implicit for pressure:
        #
        # We solve a linear system matrixA pressure = vectorE
        #
        # Since the system is small and 1D we can buid a
        # dense matrix and use explicit inversion

        # --- Build matrixA:
        matrixA = np.zeros((self.Ncells,self.Ncells))
        # First row
        matrixA[0,0] = -totalTrans[0]
        matrixA[0,1] = totalTrans[0]
        # Middle rows
        for ii in np.arange(1,self.Ncells-1):
            matrixA[ii,ii-1] = totalTrans[ii-1]
            matrixA[ii,ii] = -totalTrans[ii-1]-totalTrans[ii]
            matrixA[ii,ii+1] = totalTrans[ii]
        # Last row
        matrixA[-1,-2] = totalTrans[-1]
        matrixA[-1,-1] = -2*totalTransRight - totalTrans[-1]
        
        # ------
        # --- Build vectorE:
        vectorE = np.zeros(self.Ncells)
        vectorE[0] = -self.leftDarcyVelocity/self.deltaX
        vectorE[-1] = -2.0*totalTransRight*self.rightPressure
        # ------
        # --- Solve linear system:
        matrixAInv = np.linalg.inv(matrixA)
        pressure = np.dot(matrixAInv,vectorE)
        # --------------------------------
        # Solve explicitly for saturation:
        dtOverPoro = self.deltat/self.poro
        self.saturation[1:-1] = self.saturation[1:-1] - dtOverPoro[1:-1]*(oilTrans[1:]*(pressure[2:]-pressure[1:-1]) +oilTrans[:-1]*(pressure[:-2]-pressure[1:-1]))
        self.saturation[0] = self.saturation[0] - dtOverPoro[0]*oilTrans[0]*(pressure[1]-pressure[0])
        self.saturation[-1] = self.saturation[-1] + dtOverPoro[-1]*(2*waterTransRight*(self.rightPressure-pressure[-1])-waterTrans[-1]*(pressure[-1]-pressure[-2]))
        maxsat = 1.0-self.relpermOil.Sorw
        minsat = self.relpermOil.Swirr
        self.saturation[ self.saturation>maxsat ] = maxsat
        self.saturation[ self.saturation<minsat ] = minsat
        # --------------------------------
        self.pressure = pressure
        self.time = self.time + self.deltat
    
    def simulateTo(self,time):
        '''
        Progress simulation to specific time with
        a constant timestep self.deltat
        Args:
        time: Time to advance to [s]
        '''
        baseDeltat = self.deltat
        while self.time < time:
            if self.time + baseDeltat >= time:
                self.deltat = time - self.time
                self.doTimestep()
                self.deltat = baseDeltat
                self.time = time
            else:
                self.doTimestep()
          