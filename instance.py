import numpy
import scipy
import scikits.talkbox
from matplotlib import mlab as pylab

class Instance:
    @classmethod
    def make_instances(cls, frames):
        ffts = numpy.fft.fft(frames)
        instances = []
        for i in xrange(0,len(frames)):
            if numpy.sum(abs(frames[i])) > 0:
                instances.append(Instance(frames[i], abs(ffts[i])))

        return instances

    def __init__(self, frame, fft):
        self.lpc = scikits.talkbox.lpc(frame, 12)
        self.psd = pylab.psd(frame)[0]
        self.transforms = {
                            "energy":self.energy,
                            "zero.crossings":self.zero_crossings,
                            #"dominant.frequency":self.dominant_frequency,
                            #"full.energy":self.full_energy,
                            #"frame.mean":self.frame_mean,
                            #"frame.snr":self.snr,
                            #"2.4.khz.energy":self.two_four_khz_band_energy,
                            "3500.4300.peak":self.try_this_energy,
                            "5400.6800.peak":self.second_peak,
                            "1800,2700.peak":self.one_more_peak,
                            #"dom.0.300":self.dominant_frequency_in_0hz_300hz,
                            "dom.300.5000":self.dominant_frequency_in_300hz_5000hz,
                            #"dom.5000.24000":self.dominant_frequency_above_5000hz,
                            "low.freq.peak":self.another_peak,
                            "fft.kurtosis":self.fft_kurtosis,
                            "lpc.residual":self.lpc_residual,
                            "psd.spike":self.psd_peak,
                            "psd.other_spike":self.psd_hugepeak,
                            "psd.argmax.after.3":self.psd_argmax_after_3,
                            "psd.21.peak":self.psd_another_peak
                          }
        for i in range(0,12):
            self.transforms["lpc.2." + str(i)] = self.lpc2x(i)
            #self.transforms["lpc.0." + str(i)] = self.lpc0x(i)
        self.attributes = {}
        self.frame = frame
        self.fft = fft
        for transform in self.transforms:
            self.attributes[transform] = self.transforms[transform](frame, fft)

    def to_vector(self, keys = None):
        if keys is None:
            return self.attributes.values()
        else:
            return [self.attributes[key] for key in keys]

    def headings(self):
        return self.attributes.keys()

    def not_silent(self):
        return self.attributes["energy"] != 0

    def energy(self, frame, fft):
        return sum(x*x for x in frame)

    def psd_peak(self, frame, fft):
        s = 0
        for i in range(30,37):
            s += self.psd[i]

        return s

    def psd_hugepeak(self, frame, fft):
        return sum(self.psd[0:3])

    def psd_argmax_after_3(self, frame, fft):
        return numpy.argmax(self.psd[4:])

    def psd_another_peak(self, frame, fft):
        return self.psd[21]

    def psd_sfm(self, frame, fft):
        return numpy.mean(self.psd[4:])/scipy.stats.mstats.gmean(self.psd[4:])

    def fft_kurtosis(self, frame, fft):
        return scipy.stats.kurtosis(fft)


    def energy_filter(self, upper, lower, frame, fft):
        t = numpy.fft.fftfreq(int(48000*16/1000.0), 1.0/48000)
        s = 0
        assert len(t) == len(fft)
        for i in xrange(0,len(t)):
            if t[i] >= upper and t[i] <= lower:
                s += fft[i]

        return s*s

    def try_this_energy(self, frame, fft):
        return self.energy_filter(3500,4300,frame,fft)

    def second_peak(self, frame, fft):
        return self.energy_filter(5400,6800,frame,fft)

    def another_peak(self, frame, fft):
        return self.energy_filter(0,100,frame,fft)

    def one_more_peak(self, frame, fft):
        return self.energy_filter(1900, 2700, frame, fft)


    def zero_crossings(self, frame, fft):
        zero_crossings = numpy.where(numpy.diff(numpy.sign(frame)))[0]
        return len(zero_crossings)

    def dominant_frequency(self, frame, fft):
        return numpy.argmax(fft)

    def dominant_frequency_in_range(self,lower,upper,frame,fft):
        t = numpy.fft.fftfreq(int(48000*16/1000.0), 1.0/48000)
        i = 0

        m_i = 0
        m = 0

        for i in xrange(0,len(t)):
            if t[i] >= lower and t[i] < upper:
                if fft[i] > m:
                    m = fft[i]
                    m_i = i

        return t[m_i]

    def dominant_frequency_in_0hz_300hz(self, frame, fft):
        return self.dominant_frequency_in_range(0,300,frame,fft)

    def dominant_frequency_in_300hz_5000hz(self, frame, fft):
        return self.dominant_frequency_in_range(300,5000,frame,fft)

    def dominant_frequency_above_5000hz(self, frame, fft):
        return self.dominant_frequency_in_range(5000,24000,frame,fft)


    def full_energy(self, frame, fft):
        return 10*numpy.log10(0.1*numpy.correlate(frame, frame)[0])

    def frame_mean(self, frame, fft):
        return numpy.mean(frame)

    def snr(self, frame, fft):
        return numpy.mean(frame)/numpy.std(frame)


    def lpc2x(self, i):
        return self.lpcnx(i, 2)

    def lpc0x(self, i):
        return self.lpcnx(i, 0)

    def lpcnx(self, i, n):
        return lambda frame,fft: self.lpc[n][i]

    def lpc_residual(self, frame, fft):
        return sum([x*x for x in self.lpc[1]])
