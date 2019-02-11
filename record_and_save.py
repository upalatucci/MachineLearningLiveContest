import sounddevice as sd
import soundfile as sf
import threading


class RecordandSaveThread(threading.Thread):
    def __init__(self, duration=1, frequency=44100, steps=10):
        super().__init__()
        self.duration = duration
        self.frequency = frequency
        self.steps = steps

    def run(self):
        count = 0
        while True:
            record = sd.rec(self.duration * self.frequency, self.frequency, channels=2)
            sd.wait()
            sf.write("test{}.wav".format(count), record, self.frequency)
            count = (count + 1) % self.steps
            if count == self.steps:
                count = 0