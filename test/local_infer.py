# For benchmarking against DEFER, try this file that uses Single Device Inference

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import time

model = ResNet50(weights='imagenet', include_top=True)

img_path = '../resource/test.jpg' # YOUR IMAGE HERE
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time_run = 10
in_sec = time_run * 60
start = time.time()
task_size = 10
res_count = 0
while res_count < task_size:
    res = model.predict(x)
    res_count += 1
    print(res.shape)
end = time.time()
run = end - start
print(f"{res_count} results in {run} seconds")
print(f"Throughput: {res_count / run} req/s")
exit(0)