import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interp = Interpreter(model_path='yamnet.tflite')
wav = np.zeros(16000, dtype=np.float32)
interp.resize_tensor_input(0, [wav.shape[0]])
interp.allocate_tensors()
t0 = time.time()
interp.set_tensor(0, wav)
interp.invoke()
scores = interp.get_tensor(203)
t1 = time.time()
print('frames,classes:', scores.shape, 'time_ms:', (t1-t0)*1000)
idx = [69,70,72,73,74]
print('bark-ish scores (max over frames):', scores[:, idx].max(axis=0))
print('top class per frame:', scores.argmax(axis=1))
