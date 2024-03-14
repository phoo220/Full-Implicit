import numpy as np
from corey import coreyWater, coreyOil
from conversion import daysToSeconds,secondsToDays

class Simulator1DIMPLICIT:
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
        #This next line will also define the transmissibilities
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


        # --- Build matrixJ:
        Porooverdt = self.poro/self.deltat
        matrixJ = np.zeros((2*self.Ncells,2*self.Ncells))
        # First row
        matrixJ[0,0] = -totalTrans[0]
        matrixJ[0,1] = (totalTrans[0]/self.oilViscosity)*self.relpermOil(self.saturation[0])*(self.pressure[1]-self.pressure[0])+Porooverdt[0]
        matrixJ[1,0] = -totalTrans[0]
        matrixJ[1,1] = (totalTrans[0]/self.waterViscosity)*self.relpermWater(self.saturation[0])*(self.pressure[1]-self.pressure[0])-Porooverdt[0]
        matrixJ[0,2] = totalTrans[0]
        matrixJ[0,3] = 0
        matrixJ[1,2] = totalTrans[0]
        matrixJ[1,3] = 0

        # Middle rows
        for ii in np.arange(2, (2*self.Ncells)-2,2):
            if ii+2 < len(totalTrans) and ii-4 >=0:
                matrixJ[ii,ii-2] = totalTrans[ii]
                matrixJ[ii,ii-1] = (totalTrans[ii-2]/self.oilViscosity)*self.relpermOil(self.saturation[ii-2])*(self.pressure[ii-1]-self.pressure[ii-2])
                matrixJ[ii+1,ii-2] = totalTrans[ii-2]
                matrixJ[ii+1,ii-1] = (totalTrans[ii-2]/self.waterViscosity)*self.relpermWater(self.saturation[ii-2])*(self.pressure[ii-1]-self.pressure[ii-2])

                matrixJ[ii,ii] = -totalTrans[ii]-totalTrans[ii-4]
                matrixJ[ii,ii+1] = (totalTrans[ii]/self.oilViscosity)*self.relpermOil(self.saturation[ii])*(self.pressure[ii+1]-self.pressure[ii])+Porooverdt[ii]
                matrixJ[ii+1,ii] =-totalTrans[ii]-totalTrans[ii-1]
                matrixJ[ii+1,ii+1] =(totalTrans[ii]/self.waterViscosity)*self.relpermWater(self.saturation[ii])*(self.pressure[ii+1]-self.pressure[ii])-Porooverdt[ii]
   
                matrixJ[ii,ii+2] = totalTrans[ii+2]
                matrixJ[ii,ii+3] = 0
                matrixJ[ii+1,ii+2] =totalTrans[ii+2]
                matrixJ[ii+1,ii+3] = 0

        # Last row
        matrixJ[-2,-4] = -totalTrans[-1]
        matrixJ[-2,-3] = (totalTrans[-1]/self.oilViscosity)*self.relpermOil(self.saturation[-1])*(self.pressure[-1]-self.pressure[-2])
        matrixJ[-1,-4] = -totalTrans[-1]
        matrixJ[-1,-3] = (totalTrans[-1]/self.waterViscosity)*self.relpermWater(self.saturation[-1])*(self.pressure[-1]-self.pressure[-2])

        matrixJ[-2,-2] = -2*totalTransRight-totalTrans[-1]
        matrixJ[-2,-1] = (2*totalTrans[-1]/self.oilViscosity)*self.relpermOil(self.saturation[-1])*(self.pressure[-1]-self.pressure[-2])+Porooverdt[-1]
        matrixJ[-1,-2] = -2*totalTransRight-totalTrans[-1]
        matrixJ[-1,-1] = (2*totalTrans[-1]/self.waterViscosity)*self.relpermWater(self.saturation[-1])*(self.pressure[-1]-self.pressure[-2])-Porooverdt[-1]
        
        # ------
        # --- Build vectorX:
        vectorX =  np.zeros(2*self.Ncells)
        vectorX[::2] =self.pressure
        vectorX[1::2] = self.saturation
        # ------
        # --- Solve linear system:
        #matrixJInv = np.linalg.inv(matrixJ)
        Residual = np.dot(matrixJ,vectorX)
        self.residual = Residual
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
        