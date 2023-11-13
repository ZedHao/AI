import cv2  # 导入OpenCV库
import numpy as np
import joblib
loaded_gs_clf = joblib.load('gs_clf_model.pkl')
# 1. 准备新图片
# 请替换为您自己的图片路径
new_image_path = '006_副本.png'

# 2. 加载新图片并处理
new_image = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式加载图片
new_image = cv2.resize(new_image, (28, 28))  # 调整图片大小为28x28像素
new_image = new_image.astype('float') / 255.0  # 标准化像素值到[0, 1]

# 3. 使用模型进行预测
predicted_digit = loaded_gs_clf.predict([new_image.flatten()])
print("Predicted digit:", predicted_digit)
