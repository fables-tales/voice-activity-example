import vadutils
from sklearn.ensemble import RandomForestClassifier
import numpy
import cPickle as pickle
import wave

if __name__ == "__main__":
    s = vadutils.load_vectors("sample.wav",0,8)
    c = pickle.loads(open("final_classifier").read())
    c.verbose = 3

    headings = sorted(s[0].headings())
    vectors = numpy.array([x.to_vector(headings) for x in s])
    predictions = c.predict(vectors)



    f = wave.open("sample.wav")

    writer = wave.open("sample-filtered.wav", "w")

    writer.setnchannels(1)
    writer.setsampwidth(f.getsampwidth())
    writer.setframerate(f.getframerate())


    for pred in predictions:
        x = f.readframes(768)
        if pred == 1:
            writer.writeframes(x)
        else:
            writer.writeframes("".join(["\x00" for value in x]))

    writer.close()
