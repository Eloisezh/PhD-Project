import numpy as np

from ewq import ewq
from our_method import statisticFilter
import INT8

for ratio in range(35, 100, 10):
	ratio /= 100
	model = "yolo26n-seg.onnx"
	data = "coco.yaml"
	ewq_result = ewq(target=ratio).filter()
	our_result = statisticFilter(target=ratio).filter()
	# print("ratio:", ratio)
	# print("ewq result:", ewq_result)
	# print("our result:", our_result)
	# print("num of ewq:", len(ewq_result))
	# print("num of our:", len(our_result))

	quantizer = INT8.StaticQuantizer(model, data, ewq_result, f"ewq-{ratio}", print_original=False, print_quantized=True)
	q_model = str(quantizer.static_quantize())

	quantizer = INT8.StaticQuantizer(model, data, our_result, f"our-{ratio}", print_original=False, print_quantized=True)
	q_model = str(quantizer.static_quantize())
