from gadgetron import Gadget
import numpy as np
from skimage.restoration import unwrap_phase

class PhaseGadget(Gadget):
    def process_config(self, cfg):
        print("Attempting to open window")
        print("Window running")
        #Configuration Ignored

    def process(self, h,image,meta):

        image = np.squeeze(image)
        print(image.shape)
        phase = np.angle(image)

        phase = unwrap_phase(phase)

        image = np.abs(image)*np.exp(1j*phase)
        image = image.astype("complex64")

        self.put_next(h,phase)
        return 0