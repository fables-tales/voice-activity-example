from __future__ import division
import struct
import instance
import scipy
import wave
import numpy
from instance import Instance

def decode_frame(wave_reader, frame):
    width = wave_reader.getsampwidth()
    assert width == 2
    return struct.unpack("<h", frame)[0]

def getframes(wave_reader, nframes):
    buf = wave_reader.readframes(nframes)
    frames = list(struct.unpack_from("h"*int(len(buf)/2), buf))
    return frames

def read_region(in_file_name, start_time, length):
    wave_reader = wave.open(in_file_name)
    wave_reader.setpos(int(start_time*wave_reader.getframerate()))
    frames = getframes(wave_reader, int(length*wave_reader.getframerate()))
    return frames

def cut_region(in_file_name, start_time, length, out_file_name):
    wave_reader = wave.open(in_file_name)
    tf = open(out_file_name, "w")
    wave_writer = wave.open(tf, "w")
    wave_writer.setnchannels(1)
    wave_writer.setsampwidth(2)
    wave_writer.setframerate(wave_reader.getframerate())

    wave_reader.setpos(int(start_time*wave_reader.getframerate()))

    frames = getframes(wave_reader, int(length*wave_reader.getframerate()))
    outframes = [struct.pack("h", frame)[0:2] for frame in frames]
    wave_writer.writeframes("".join(outframes))

    wave_writer.close()

def compute_end_samples_seconds(waveobj, end_seconds):
    if end_seconds != -1:
        end_samples = end_seconds * waveobj.getframerate()
    else:
        end_samples = waveobj.getnframes()
        end_seconds = end_samples / waveobj.getframerate()

    return end_samples,end_seconds

def get_samples(filename, start_seconds, end_seconds):
    return read_region(filename, start_seconds, end_seconds-start_seconds)

window_width_ms = 16

def samples_to_16ms_frames(samples, framerate=48000):
    frame_size = int(window_width_ms/1000.0 * framerate)

    while len(samples) % frame_size != 0:
        samples.append(0)


    mses = numpy.reshape(samples, (-1, frame_size))
    return mses


def get_16_ms_frames(filename, start_seconds, end_seconds):
    return samples_to_16ms_frames(get_samples(filename, start_seconds, end_seconds))

def n_zero_crossings(values):
    zero_crossings = numpy.where(numpy.diff(numpy.sign(values)))[0]

    return len(zero_crossings)

def load_vectors(filename, start, end):
    samples = get_samples(filename, start, end)
    frames = samples_to_16ms_frames(samples)
    return Instance.make_instances(frames)

def features_from_frames(frames):
    instances = instance.Instance.make_instances(frames)
    headings = sorted(instances[0].headings())
    feature_instances = [x.to_vector(headings) for x in instances]
    return feature_instances

def remove_zero_frames(frames):
    r = []
    for frame in frames:
        af = numpy.abs(frame)
        if sum(af) > 0:
            r.append(frame)
    return r
