#!/usr/bin/env python
# coding: utf-8

# ### Step 1. Prepare calibration data.

# In[22]:


import numpy as np
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat
import cv2
	
class ImageCalibrationDataReader(CalibrationDataReader):
	def __init__(self, image_paths):
		self.image_paths = image_paths
		self.idx = 0
		self.input_name = "images"

	def preprocess(self, frame):
		frame = cv2.imread(frame)
		X = cv2.resize(frame, (640, 640))
		image_data = np.array(X).astype(np.float32) / 255.0  # Normalize to [0, 1] range
		image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
		image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
		return image_data

	def get_next(self):
		if self.idx >= len(self.image_paths):
			return None

		batch_size = 1  # 你想要的 batch 大小
		batch_images = []

		for _ in range(batch_size):
			if self.idx >= len(self.image_paths):
				break
			image_path = self.image_paths[self.idx]
			input_data = self.preprocess(image_path)
			batch_images.append(input_data[0])  # remove batch dim to stack later
			self.idx += 1
		
		if not batch_images:
			return None

		batch = np.stack(batch_images, axis=0)  # shape: (B, 3, 640, 640)
		print(f"[INFO] Feeding batch, shape = {batch.shape}")
		return {self.input_name: batch}


# ### Step 2. Static quantize the model.

# In[ ]:


import os, onnx, datetime
from ultralytics import YOLO
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

class StaticQuantizer():
	def __init__(self, model_path, data, nodes_exclude, suffix, print_original=False):
		self.model_path = model_path
		self.data = data
		self.nodes_exclude = nodes_exclude
		self.print_original = print_original
		self.suffix = suffix
	
	def static_quantize(self):
		if self.print_original:
			# before quant val
			model = YOLO(self.model_path, task="segment")
			performance = model.val(data=self.data,workers=0,batch=4)
			### OUTPUT
			output = ">>> VAL BEFORE", performance.box.map50, performance.box.map, performance.seg.map50, performance.seg.map50, "\n MODEL_NAME", self.model_path, "<<<"
			print(output)
			### help func starts (ignore)
			current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			with open("data.txt", "a", encoding="utf-8") as f:
				f.write(f"[{current_time}] {output}\n")
				f.write("-" * 80 + "\n")
			### help func ends

		# static quantize (main)
		### help func starts (ignore)
		directory = os.path.dirname(self.model_path)
		filename = os.path.basename(self.model_path)
		name, ext = os.path.splitext(filename)
		new_filename = f"QINT8_{name}_{self.suffix}{ext}"	# whatever easy to read with quantized sign and idx
		### help func ends

		quantized_path = os.path.join("q_models", new_filename)
		calibration_image_paths = ['calibration_data/frame_0000.jpg','calibration_data/frame_0001.jpg','calibration_data/frame_0002.jpg',
						'calibration_data/frame_0003.jpg','calibration_data/frame_0004.jpg','calibration_data/frame_0005.jpg',
						'calibration_data/frame_0006.jpg','calibration_data/frame_0007.jpg','calibration_data/frame_0008.jpg',
						'calibration_data/frame_0009.jpg','calibration_data/frame_0010.jpg']
		calibration_data_reader = ImageCalibrationDataReader(calibration_image_paths)
		quantize_static(
			self.model_path,
			quantized_path,
			weight_type=QuantType.QInt8,
			activation_type=QuantType.QUInt8,
			quant_format=QuantFormat.QDQ,
			nodes_to_exclude=self.nodes_exclude,
			per_channel=True,
			calibration_data_reader=calibration_data_reader,
		) # adjust with experiment settings

		### help func starts (ignore)
		inter_model = onnx.load(quantized_path)
		graph = inter_model.graph
		nodes_to_remove = []
		for node in graph.node:
			if node.op_type in ["QuantizeLinear", "DequantizeLinear"] and "output0" in node.output[0]:
				nodes_to_remove.append(node)
		for node in nodes_to_remove:
			graph.node.remove(node)
		for node in graph.node:
			if node.op_type == "Concat": 
				graph.output[0].name = node.output[0]  
		onnx.save(inter_model, quantized_path)
		### help func ends

		# after quant val
		q_model = YOLO(quantized_path, task="segment")
		q_performance = q_model.val(data=self.data,workers=0,batch=4)
		### OUTPUT
		q_output = ">>> VAL AFTER", q_performance.box.map50, q_performance.box.map, q_performance.seg.map50, q_performance.seg.map50, "\n MODEL_NAME", quantized_path, "<<<"
		print(q_output)
		### help func starts (ignore)
		q_current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open("data.txt", "a", encoding="utf-8") as f:
			f.write(f"[{q_current_time}] {q_output}\n")
			f.write("-" * 80 + "\n")
		### help func ends
		
		return quantized_path


# ### Step 3. Test using YOLO11 Fish.

# In[24]:


if __name__ == "__main__":
	model = "E:/code dissertation/ultralytics_v11-8.3.9/ultralytics_v11-8.3.9/runs/finetune-fish-combined9/weights/best.onnx"
	data = "D:/app/datasets/fish-combined/data.yaml"
	nodes_exclude = ['/model.23/proto/cv1/conv/Conv', '/model.23/cv4.0/cv4.0.0/conv/Conv', '/model.23/cv2.0/cv2.0.0/conv/Conv', '/model.23/cv3.0/cv3.0.0/cv3.0.0.0/conv/Conv', '/model.23/proto/cv1/act/Sigmoid', '/model.23/cv4.0/cv4.0.0/act/Sigmoid', '/model.23/cv2.0/cv2.0.0/act/Sigmoid', '/model.23/cv3.0/cv3.0.0/cv3.0.0.0/act/Sigmoid', '/model.23/proto/cv1/act/Mul', '/model.23/cv4.0/cv4.0.0/act/Mul', '/model.23/cv2.0/cv2.0.0/act/Mul', '/model.23/cv3.0/cv3.0.0/cv3.0.0.0/act/Mul', '/model.23/proto/upsample/ConvTranspose', '/model.23/cv4.0/cv4.0.1/conv/Conv', '/model.23/cv2.0/cv2.0.1/conv/Conv', '/model.23/cv3.0/cv3.0.0/cv3.0.0.1/conv/Conv', '/model.23/proto/cv2/conv/Conv', '/model.23/cv4.0/cv4.0.1/act/Sigmoid', '/model.23/cv2.0/cv2.0.1/act/Sigmoid', '/model.23/cv3.0/cv3.0.0/cv3.0.0.1/act/Sigmoid', '/model.23/proto/cv2/act/Sigmoid', '/model.23/cv4.0/cv4.0.1/act/Mul', '/model.23/cv2.0/cv2.0.1/act/Mul', '/model.23/cv3.0/cv3.0.0/cv3.0.0.1/act/Mul', '/model.23/proto/cv2/act/Mul', '/model.23/cv4.0/cv4.0.2/Conv', '/model.23/cv2.0/cv2.0.2/Conv', '/model.23/cv3.0/cv3.0.1/cv3.0.1.0/conv/Conv', '/model.23/proto/cv3/conv/Conv', '/model.23/Reshape', '/model.23/cv3.0/cv3.0.1/cv3.0.1.0/act/Sigmoid', '/model.23/proto/cv3/act/Sigmoid', '/model.23/cv3.0/cv3.0.1/cv3.0.1.0/act/Mul', '/model.23/proto/cv3/act/Mul', '/model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv', '/model.23/cv3.0/cv3.0.1/cv3.0.1.1/act/Sigmoid', '/model.23/cv3.0/cv3.0.1/cv3.0.1.1/act/Mul', '/model.23/cv3.0/cv3.0.2/Conv', '/model.23/Concat_1', '/model.23/Reshape_3', '/model.23/cv4.1/cv4.1.0/conv/Conv', '/model.23/cv2.1/cv2.1.0/conv/Conv', '/model.23/cv3.1/cv3.1.0/cv3.1.0.0/conv/Conv', '/model.23/cv4.1/cv4.1.0/act/Sigmoid', '/model.23/cv2.1/cv2.1.0/act/Sigmoid', '/model.23/cv3.1/cv3.1.0/cv3.1.0.0/act/Sigmoid', '/model.23/cv4.1/cv4.1.0/act/Mul', '/model.23/cv2.1/cv2.1.0/act/Mul', '/model.23/cv3.1/cv3.1.0/cv3.1.0.0/act/Mul', '/model.23/cv4.1/cv4.1.1/conv/Conv', '/model.23/cv2.1/cv2.1.1/conv/Conv', '/model.23/cv3.1/cv3.1.0/cv3.1.0.1/conv/Conv', '/model.23/cv4.1/cv4.1.1/act/Sigmoid', '/model.23/cv2.1/cv2.1.1/act/Sigmoid', '/model.23/cv3.1/cv3.1.0/cv3.1.0.1/act/Sigmoid', '/model.23/cv4.1/cv4.1.1/act/Mul', '/model.23/cv2.1/cv2.1.1/act/Mul', '/model.23/cv3.1/cv3.1.0/cv3.1.0.1/act/Mul', '/model.23/cv4.1/cv4.1.2/Conv', '/model.23/cv2.1/cv2.1.2/Conv', '/model.23/cv3.1/cv3.1.1/cv3.1.1.0/conv/Conv', '/model.23/Reshape_1', '/model.23/cv3.1/cv3.1.1/cv3.1.1.0/act/Sigmoid', '/model.23/cv3.1/cv3.1.1/cv3.1.1.0/act/Mul', '/model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv', '/model.23/cv3.1/cv3.1.1/cv3.1.1.1/act/Sigmoid', '/model.23/cv3.1/cv3.1.1/cv3.1.1.1/act/Mul', '/model.23/cv3.1/cv3.1.2/Conv', '/model.23/Concat_2', '/model.23/Reshape_4', '/model.23/cv4.2/cv4.2.0/conv/Conv', '/model.23/cv2.2/cv2.2.0/conv/Conv', '/model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv', '/model.23/cv4.2/cv4.2.0/act/Sigmoid', '/model.23/cv2.2/cv2.2.0/act/Sigmoid', '/model.23/cv3.2/cv3.2.0/cv3.2.0.0/act/Sigmoid', '/model.23/cv4.2/cv4.2.0/act/Mul', '/model.23/cv2.2/cv2.2.0/act/Mul', '/model.23/cv3.2/cv3.2.0/cv3.2.0.0/act/Mul', '/model.23/cv4.2/cv4.2.1/conv/Conv', '/model.23/cv2.2/cv2.2.1/conv/Conv', '/model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv', '/model.23/cv4.2/cv4.2.1/act/Sigmoid', '/model.23/cv2.2/cv2.2.1/act/Sigmoid', '/model.23/cv3.2/cv3.2.0/cv3.2.0.1/act/Sigmoid', '/model.23/cv4.2/cv4.2.1/act/Mul', '/model.23/cv2.2/cv2.2.1/act/Mul', '/model.23/cv3.2/cv3.2.0/cv3.2.0.1/act/Mul', '/model.23/cv4.2/cv4.2.2/Conv', '/model.23/cv2.2/cv2.2.2/Conv', '/model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv', '/model.23/Reshape_2', '/model.23/cv3.2/cv3.2.1/cv3.2.1.0/act/Sigmoid', '/model.23/Concat', '/model.23/cv3.2/cv3.2.1/cv3.2.1.0/act/Mul', '/model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv', '/model.23/cv3.2/cv3.2.1/cv3.2.1.1/act/Sigmoid', '/model.23/cv3.2/cv3.2.1/cv3.2.1.1/act/Mul', '/model.23/cv3.2/cv3.2.2/Conv', '/model.23/Concat_3', '/model.23/Reshape_5', '/model.23/Concat_4', '/model.23/Split', '/model.23/dfl/Reshape', '/model.23/Sigmoid', '/model.23/dfl/Transpose', '/model.23/dfl/Softmax', '/model.23/dfl/conv/Conv', '/model.23/dfl/Reshape_1', '/model.23/Slice', '/model.23/Slice_1', '/model.23/Sub', '/model.23/Add_1', '/model.23/Add_2', '/model.23/Sub_1', '/model.23/Div_1', '/model.23/Concat_5', '/model.23/Mul_2', '/model.23/Concat_6']
	idx = 0
	quantizer = StaticQuantizer(model, data, nodes_exclude, idx, print_original=False)
	quantizer.static_quantize()

