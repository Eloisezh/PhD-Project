import numpy as np

from ewq import ewq
from our_method import statisticFilter
from INT8 import INT8

for ratio in np.arange(0.35, 1.00, 0.1):
	model = "yolo26n-seg.onnx"
	data = "coco.yaml"
	ewq_result = ewq(ratio=ratio).filter()
	our_result = statisticFilter(ratio=ratio).filter()

	quantizer = INT8.StaticQuantizer(model, data, ewq_result, f"ewq-{ratio}", print_original=False, print_quantized=True)
	q_model = str(quantizer.static_quantize())

	quantizer = INT8.StaticQuantizer(model, data, our_result, f"our-{ratio}", print_original=False, print_quantized=True)
	q_model = str(quantizer.static_quantize())
