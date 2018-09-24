import numpy as np
import ismrmrd
from gadgetron import Gadget
import ismrmrd
import ismrmrd.xsd

class AccumulateAndRecon(Gadget):
    def __init__(self, next_gadget=None):
        Gadget.__init__(self,next_gadget)
        self.girf = None
        self.girf_header = None


    def process_config(self, conf):
        self.header = ismrmrd.xsd.CreateFromDocument(conf)
        self.enc = self.header.encoding[0]

    def process(self, acqheader : ismrmrd.AcquisitionHeader, data : np.ndarray):

        if acqheader.is_flag_set(30): #For now we set this flag to 30. It feels... maybe not perfect?
            self.girf = data
            self.girf_header = acqheader
        else:
            if

        return 0 #Everything OK