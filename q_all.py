#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ewq import ewq
import our_method
import our_method_mean_only
import INT8

for ratio in range(35, 100, 10):
	ratio /= 100
	model = "yolo26n-seg.onnx"
	data = "coco.yaml"
	ewq_result = ewq(target=ratio).filter()
	our_result = our_method.statisticFilter(target=ratio).filter()
	union = list(set(ewq_result).union(set(our_result)))
	print("ratio:", ratio)
	print("ewq result:", ewq_result)
	print("our result:", our_result)
	print("num of ewq:", len(ewq_result))
	print("num of our:", len(our_result))

	# quantizer = INT8.StaticQuantizer(model, data, ewq_result, f"ewq-{ratio}", print_original=False, print_quantized=True)
	# q_model = str(quantizer.static_quantize())

	quantizer = INT8.StaticQuantizer(model, data, union, f"union-{ratio}", print_original=False, print_quantized=True)
	q_model = str(quantizer.static_quantize())


	# our_mean_only_result = our_method_mean_only.statisticFilter(target=ratio).filter()
	# union_mean_only = list(set(ewq_result).union(set(our_mean_only_result)))
	# print("ratio:", ratio)
	# print("ewq result:", ewq_result)
	# print("our mean only result:", our_mean_only_result)
	# print("num of ewq:", len(ewq_result))
	# print("num of our mean only:", len(our_mean_only_result))

	# if our_mean_only_result is not None:
	# 	quantizer = INT8.StaticQuantizer(model, data, union_mean_only, f"union_mean_only-{ratio}", print_original=False, print_quantized=True)
	# 	q_model = str(quantizer.static_quantize())

	# quantizer = INT8.StaticQuantizer(model, data, our_mean_only_result, f"our-mean-only-{ratio}", print_original=False, print_quantized=True)
	# q_model = str(quantizer.static_quantize())


# In[ ]:




