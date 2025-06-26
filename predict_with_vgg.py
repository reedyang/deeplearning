import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import Model

# 1. 加载预训练的VGG16模型
model = VGG16(weights='imagenet')

# 2. 创建热力图生成模型
last_conv_layer = model.get_layer('block5_conv3')  # VGG16最后一个卷积层
heatmap_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

# 3. 加载并预处理图像
img_path = 'animals/elephant.jpg'  # 替换为你的图片路径
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 4. 进行预测
preds = model.predict(x)
print("Recognition Results:")
for _, animal, prob in decode_predictions(preds, top=3)[0]:
    print(f"- {animal}: {prob*100:.2f}%")

# 5. 生成热力图
with tf.GradientTape() as tape:
    conv_output, predictions = heatmap_model(x)
    pred_index = np.argmax(predictions[0])
    output = predictions[:, pred_index]
    grads = tape.gradient(output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# 计算权重和热力图
conv_output = conv_output[0]
heatmap = conv_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0)  # ReLU激活
heatmap /= tf.reduce_max(heatmap)  # 归一化

# 6. 调整热力图大小
heatmap = np.array(heatmap)
heatmap = np.uint8(255 * heatmap)
heatmap = plt.cm.jet(heatmap)[..., :3]  # 应用颜色映射
superimposed_img = plt.cm.jet(heatmap)[..., :3]  # 只取RGB通道

# 7. 叠加热力图到原图
img = image.load_img(img_path)
img = image.img_to_array(img)
heatmap = image.array_to_img(heatmap).resize((img.shape[1], img.shape[0]))
heatmap = image.img_to_array(heatmap)
superimposed_img = heatmap * 0.4 + img * 0.6  # 调整透明度
superimposed_img = image.array_to_img(superimposed_img)

# 8. 显示结果
plt.figure(figsize=(10, 8))
plt.imshow(superimposed_img)
plt.title('Attention Heatmap')
plt.axis('off')
plt.tight_layout()
plt.savefig('animals/elephant_heatmap.jpg', bbox_inches='tight')  # 保存结果
plt.show()