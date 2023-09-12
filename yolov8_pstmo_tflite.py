import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
import torch

def letterbox(new_shape=(640, 640), labels=None, image=None):
    """Resize image and padding for detection, instance segmentation, pose."""

    # Hardcoded parameters
    auto = False
    scaleFill = False
    scaleup = True
    center = True
    stride = 32

    # Initial setup
    if labels is None:
        labels = {}
    img = labels.get('img') if image is None else image
    shape = img.shape[:2]  # current shape [height, width]

    # Adjust new_shape based on input
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2

    # Resize and add border
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

    # Update labels and return
    if len(labels):
        # labels = self._update_labels(labels, ratio, dw, dh)  # Note: _update_labels method is not defined in your code
        labels['img'] = img
        labels['resized_shape'] = new_shape
        return labels
    else:
        return img


def draw_bbox_on_image(image, x, y, w, h):
    # Convert center coordinates (x, y) to top-left corner coordinates
    # x = int(x - w/2)
    # y = int(y - h/2)
    # w = int(w)
    # h = int(h)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)


    # Draw the bounding box
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    return image

def plot_keypoints_on_image(image, keypoints, t):
    # Iterate over the keypoints
    for keypoint in keypoints:
        x, y, visibility = keypoint

        # Check if the visibility is greater than the threshold
        if visibility > t:
            # Denormalize the coordinates
            x = int(x)
            y = int(y)

            # Draw the keypoint
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    return image

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/ubuntu/projects/ultralytics/yolov8n-pose_saved_model/yolov8n-pose_float32.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read the image
image_path = "/home/ubuntu/projects/ultralytics/bus.jpg"
image = cv2.imread(image_path)
im0s = letterbox(new_shape=(640, 640), image=image)
im0s = np.array(im0s).astype(np.float32)
im0s = np.expand_dims(im0s, axis=0)
w, h = 640, 640


# Get the input size from the model's input details and resize the image accordingly
im0s /= 255.0  # normalize

details = input_details[0]
integer = details['dtype'] in (np.int8, np.int16)  # is TFLite quantized int8 or int16 model
print("integer", integer)
if integer:
    scale, zero_point = details['quantization']
    im0s = (im0s / scale + zero_point).astype(details['dtype'])  # de-scale
interpreter.set_tensor(details['index'], im0s)
interpreter.invoke()
y = []
for output in output_details:
    x = interpreter.get_tensor(output['index'])
    if integer:
        scale, zero_point = output['quantization']
        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
    if x.ndim > 2:  # if task is not classification
        # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
        # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
        x[:, [0, 2]] *= w
        x[:, [1, 3]] *= h
    y.append(x)


output_data = y[0][0]
output_data_transposed = output_data.transpose()

# Select the top K bboxes
K = 1  # Change this to your desired number of bboxes
BASE = 0
sorted_indices = np.argsort(output_data_transposed[:, 4])[::-1]
print(f"Max confidence: {output_data_transposed[sorted_indices[0], 4]}")
print(f"Min confidence: {output_data_transposed[sorted_indices[-1], 4]}")
top_K_by_confidence = output_data_transposed[sorted_indices[BASE:BASE+K]]
print("top_K_by_confidence", top_K_by_confidence[0])

# Process each bbox
im0s = im0s[0] * 255.0
im0s = im0s.astype(np.uint8)
top_K_by_confidence[:, :4] = ops.xywh2xyxy(top_K_by_confidence[:, :4])
for bbox in top_K_by_confidence:
    keypoints = bbox[5:].reshape((17, 3))
    xywh = bbox[:4]
    im0s = draw_bbox_on_image(im0s, xywh[0], xywh[1], xywh[2], xywh[3])
    im0s = plot_keypoints_on_image(im0s, keypoints, 0.0)

# Save the image
im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
cv2.imwrite('test-barebones-tflite.png', im0s)