import gadgetron as g
import ismrmrd
import numpy as np



def extract_data(waveforms):

    data = list(map(lambda w: w.data,waveforms))
    time_data = list(map(lambda w: w.time_stamp*2.5e-3+np.arange(0,w.number_of_samples*w.sample_time_us*1e-6,
                                                          w.sample_time_us*1e-6), waveforms))
    print(data)
    data = np.concatenate(data,axis=1)
    data = data.astype('float64')

    data[0:4] -= 2048

    time_data = np.concatenate(time_data)

    return data, time_data



def get_acquisition_clusters(acquisition_times):

    diffs = np.diff(acquisition_times)

    avg_timestep = np.median(diffs)

    group = 0
    labels = []
    for d in diffs:

        labels.append(group)
        if d > avg_timestep*50:
            group += 1
    labels.append(group)
    labels = np.array(labels)
    acq_times = [(np.min(acquisition_times[labels == label]),np.max(acquisition_times[labels == label])) for label in np.unique(labels)]

    return acq_times



def plot_waveforms(waveforms, acquisition_headers):
    print(type(acquisition_headers))

    acquisition_times = np.array([acq.acquisition_time_stamp for acq in acquisition_headers.ravel()])*2.5e-3


    time_between_acquisitions = np.median(np.diff(acquisition_times))

    acquisition_lengths = np.array([acq.number_of_samples*acq.sample_time_us for acq in acquisition_headers.ravel()])*1e-6

    ecg_data,time_data = extract_data(waveforms)


    ecg_trigger = ecg_data[4].astype("int")
    ecg_trigger = np.bitwise_and(ecg_trigger,2**14) > 0



    offset = np.min(time_data)
    time_data -= offset
    acquisition_times -= offset



    import matplotlib.pyplot as plt

    acquisition_clusters = get_acquisition_clusters(acquisition_times)
#
    for i in range(4):
        ax=plt.subplot(4,1,i+1)
        if i ==0: plt.title("ECG data")
        for start_time,end_time in acquisition_clusters:
            ax.axvspan(start_time,end_time,color="C2",alpha=0.8)
        ax.plot(time_data,ecg_data[i],color="C0")
        ax.plot(time_data[ecg_trigger],ecg_data[i,ecg_trigger],'o',color="C1")


    plt.show()


class PlotGadget(g.Gadget):


    def __init__(self,next_gadget=None):
        g.Gadget.__init__(self,next_gadget)
        self.waveform_data = []
        self.headers = []

    def process_config(self, conf):
        pass

    def process(self, *args):
        if isinstance(args[0], ismrmrd.WaveformHeader):
            self.waveform_data.append(args[1])
            self.headers.append(args[0])
        elif len(args[0]) > 0 and isinstance(args[0][0],g.IsmrmrdReconBit):
            waveforms = [ ismrmrd.Waveform(h,d) for h,d in zip(self.headers,self.waveform_data)]

            waveforms = list(filter(lambda wav: wav.waveform_id == 0,waveforms))
            if (len(waveforms) > 0):
                result = plot_waveforms(waveforms, args[0][0].data.headers)
            self.headers = []
            self.waveform_data = []
            # self.put_next(*result)
        else:

            print(args)
            self.put_next(*args)

        return 0









