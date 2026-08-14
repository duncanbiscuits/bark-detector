from ai_edge_litert.interpreter import Interpreter
interp = Interpreter(model_path='yamnet.tflite')
interp.allocate_tensors()
print('INPUT:', interp.get_input_details())
print('OUTPUT:', interp.get_output_details())
