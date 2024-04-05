import numpy as np

def normSw(fSw,fSwirr,fSnrw):
    return (fSw-fSwirr)/(1.0-fSnrw-fSwirr)

'''Water relative permeability'''
class coreyWater:
    def __init__(self,Nw,Krwo,Swirr,Sorw):
        '''
        Args:
        Nw: Exponent
        Krwo: Relperm at 1-S_{orw}
        Swirr: S_{wi}
        Sorw: S_{orw}
        '''
        self.Nw = Nw
        self.Krwo = Krwo
        self.Swirr = Swirr
        self.Sorw = Sorw
    def __call__(self,Sw):
        nSw = normSw(Sw,self.Swirr,self.Sorw)
        dkrWater_value = self.Krwo*self.Nw*nSw**(self.Nw-1.0)
        dkrWater = dkrWater_value
        dkrWater = np.maximum(1E-8,dkrWater)
        return dkrWater

class coreyOil:
    def __init__(self,No,Swirr,Sorw):
        '''
        Args:
        No: Exponent
        Swirr: S_{wi}
        Sorw: S_{orw}
        '''
        self.No = No
        self.Swirr = Swirr
        self.Sorw = Sorw
    def __call__(self,Sw):
        nSw = normSw(Sw,self.Swirr,self.Sorw)
        dkrOil_value = -self.No*(1.0-nSw)**(self.No-1.0)
        dkrOil = dkrOil_value
        dkrOil = np.maximum(1E-8,dkrOil)
        return dkrOil

    

