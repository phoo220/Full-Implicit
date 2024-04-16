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
        self.poro = 0.25*np.ones(2*Ncells,dtype=np.float64)
        #This next line will also define the transmissibilities
        self._perm = self.setPermeabilities(1.0E-13*np.ones(2*Ncells))
        self.pressure = 1.0E7*np.ones(2*Ncells,dtype=np.float64)
        self.saturation = 0.2*np.ones(2*Ncells,dtype=np.float64)
        self.rightPressure = 1.0E7
        self.leftDarcyVelocity = 2.315E-6 * self.poro[0]
        self.mobilityWeighting = 1.0
        self.deltat = daysToSeconds(0.1)
        self.time = 0.0
        self.oilViscosity = 2.0E-3
        self.waterViscosity = 1.0E-3
        self.relpermWater = coreyWater(2.0,0.4,0.2,0.2)
        self.relpermOil = coreyOil(2.0,0.2,0.2)
    
    def setPermeabilities(self,permVector):
        self._perm = permVector
        self._Tran = (2.0/(1.0/self._perm[:-1]+1.0/self._perm[1:]))/self.deltaX**2
        self._TranRight = self._perm[-1]/self.deltaX**2
    
    def doTimestep(self, tolerance=1e-6, max_iterations=10):
        '''
        Do one time step of length self.deltat until residual is small enough.
        '''
        iteration = 0
        self.prevSat=np.copy(self.saturation)
        while True:
            print('Iteration-',iteration)
            # Calculate mobility for oil and water phases based on current saturation
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
            Porooverdt = self.poro/self.deltat
            
            # --- Build vectorR:
            vectorR =  np.zeros(2*self.Ncells, dtype=np.float64)
            for i in range(2,2*self.Ncells-2):
                if i % 2 == 0:
                    vectorR[i] = oilTrans[i]*(self.pressure[i+1]-self.pressure[i])-oilTrans[i-1]*(self.pressure[i]-self.pressure[i-1])+Porooverdt[i]*(self.saturation[i]-self.prevSat[i])
                else:
                    vectorR[i] = waterTrans[i]*(self.pressure[i+1]-self.pressure[i])-waterTrans[i-1]*(self.pressure[i]-self.pressure[i-1])-Porooverdt[i]*(self.saturation[i]-self.prevSat[i])
            vectorR[0] = oilTrans[0]*(self.pressure[1]-self.pressure[0])+Porooverdt[0]*(self.saturation[0]-self.prevSat[0])
            vectorR[1] = waterTrans[0]*(self.pressure[1]-self.pressure[0])+self.leftDarcyVelocity/self.deltaX-Porooverdt[0]*(self.saturation[0]-self.prevSat[0])
            vectorR[-2] = 2*oilTransRight*(self.rightPressure-self.pressure[-1])-oilTrans[-2]*(self.pressure[-1]-self.pressure[-2])+Porooverdt[-1]*(self.saturation[-1]-self.prevSat[-1])
            vectorR[-1] = 2*waterTransRight*(self.rightPressure-self.pressure[-1])-waterTrans[-2]*(self.pressure[-1]-self.pressure[-2])-Porooverdt[-1]*(self.saturation[-1]-self.prevSat[-1])

            # --- Build vectorX:
            vectorX =  np.zeros(2*self.Ncells,dtype=np.float64)
            vectorX[::2] = self.pressure[::2]
            vectorX[1::2] = self.saturation[1::2]
            vectorX[-2] = self.rightPressure

            # --- Build matrixJ
            matrixJ = np.zeros((2*self.Ncells,2*self.Ncells),dtype=np.float64)

            # First row
            matrixJ[0,0] = -oilTrans[0]
            matrixJ[0,1] = (self._Tran[0]/self.oilViscosity)*self.relpermOil(self.saturation[0])*(self.pressure[1]-self.pressure[0])+Porooverdt[0]
            matrixJ[1,0] = -waterTrans[0]
            matrixJ[1,1] = (self._Tran[0]/self.waterViscosity)*self.relpermWater(self.saturation[0])*(self.pressure[1]-self.pressure[0])-Porooverdt[0]

            matrixJ[0,2] = oilTrans[0]
            matrixJ[0,3] = 0
            matrixJ[1,2] = waterTrans[0]
            matrixJ[1,3] = 0

            # Middle rows
            for ii in np.arange(2,(2*self.Ncells)-2,2):
                #if ii+2 < len(oilTrans) and ii-4 >=0:
                matrixJ[ii,ii-2] = oilTrans[ii-1]
                matrixJ[ii,ii-1] = -(self._Tran[ii-1]/self.oilViscosity)*self.relpermOil(self.saturation[ii-1])*(self.pressure[ii]-self.pressure[ii-1])
                matrixJ[ii+1,ii-2] = waterTrans[ii-1]
                matrixJ[ii+1,ii-1] = -(self._Tran[ii-1]/self.waterViscosity)*self.relpermWater(self.saturation[ii-1])*(self.pressure[ii]-self.pressure[ii-1])

                matrixJ[ii,ii] = -oilTrans[ii]-oilTrans[ii-1]
                matrixJ[ii,ii+1] = (self._Tran[ii]/self.oilViscosity)*self.relpermOil(self.saturation[ii])*(self.pressure[ii+1]-self.pressure[ii])+Porooverdt[ii]
                matrixJ[ii+1,ii] =-waterTrans[ii]-waterTrans[ii-1]
                matrixJ[ii+1,ii+1] =(self._Tran[ii]/self.waterViscosity)*self.relpermWater(self.saturation[ii])*(self.pressure[ii+1]-self.pressure[ii])-Porooverdt[ii]

                matrixJ[ii,ii+2] = oilTrans[ii+1]
                matrixJ[ii,ii+3] = 0
                matrixJ[ii+1,ii+2] =waterTrans[ii+1]
                matrixJ[ii+1,ii+3] = 0

            # Last row
            matrixJ[-2,-4] = oilTrans[-2]
            matrixJ[-2,-3] = -(self._Tran[-2]/self.oilViscosity)*self.relpermOil(self.saturation[-2])*(self.pressure[-1]-self.pressure[-2])
            matrixJ[-1,-4] = waterTrans[-2]
            matrixJ[-1,-3] = -(self._Tran[-2]/self.waterViscosity)*self.relpermWater(self.saturation[-2])*(self.pressure[-1]-self.pressure[-2])

            matrixJ[-2,-2] = -2*oilTransRight-oilTrans[-2]
            matrixJ[-2,-1] = (2*self._TranRight/self.oilViscosity)*self.relpermOil(self.saturation[-1])*(self.rightPressure-self.pressure[-1])+Porooverdt[-1]
            matrixJ[-1,-2] = -2*waterTransRight-waterTrans[-2]
            matrixJ[-1,-1] = (2*self._TranRight/self.waterViscosity)*self.relpermWater(self.saturation[-1])*(self.rightPressure-self.pressure[-1])-Porooverdt[-1]


            # --- Solve linear system:
            matrixJInv = np.linalg.inv(matrixJ)
            deltaX = -np.dot(matrixJInv,vectorR)
            Xm = vectorX + deltaX

            # Update vectorX and saturation
            self.vectorX = Xm
            self.pressure[::2] = Xm[::2]
            self.saturation[1::2] = Xm[1::2]
            maxsat = 1.0-self.relpermOil.Sorw
            minsat = self.relpermOil.Swirr
            self.saturation[ self.saturation>maxsat ] = maxsat
            self.saturation[ self.saturation<minsat ] = minsat
            self.time = self.time + self.deltat
            self.residual = vectorR
            self.prevSat=np.copy(self.saturation)

            # Check convergence
            residual_norm = np.linalg.norm(vectorR)
            if residual_norm < tolerance or iteration >= max_iterations:
                break
            iteration += 1
            print('residual_norm-',residual_norm)
            print('pressure - ',self.pressure)
            print('saturation -',self.saturation)
            #print('residual -', self.residual)
        #self.residual = vectorR
        #self.pressure[::2] = Xm[::2]
        #self.saturation[1::2] = Xm[1::2]
        

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
                