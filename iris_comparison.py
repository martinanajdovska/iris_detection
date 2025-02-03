import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import hamming
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt


def detect_iris(image, model):
    image_resized = cv2.resize(image, (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0)
    image_resized = image_resized / 255.0

    pred = model.predict(image_resized)
    center_x, center_y, radius = pred[0]

    center_x, center_y, radius = int(center_x), int(center_y), int(radius)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    iris_region = cv2.bitwise_and(gray_image, gray_image, mask=mask)

    iris_region_resized = cv2.resize(iris_region, (224, 224))

    image_with_iris = image.copy()
    cv2.circle(image_with_iris, (center_x, center_y), radius, (0, 255, 0), 2)

    return iris_region_resized, image_with_iris


def generate_mobilenet_descriptor(iris_region):
    iris_region = np.expand_dims(iris_region, axis=-1)
    iris_region = np.repeat(iris_region, 3, axis=-1)
    iris_region = np.expand_dims(iris_region, axis=0)
    iris_region = tf.keras.applications.mobilenet_v2.preprocess_input(iris_region)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = base_model.predict(iris_region)

    descriptor = features.flatten()

    return descriptor


def generate_sift_descriptor(iris_region):
    max_descriptor_length = 1000

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    iris_region = clahe.apply(iris_region)
    iris_region = cv2.Canny(iris_region, 100, 200)

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=15, nOctaveLayers=64)
    keypoints, descriptors = sift.detectAndCompute(iris_region, None)
    descriptor = descriptors.flatten()

    if descriptor.shape[0] < max_descriptor_length:
        padding = np.zeros((max_descriptor_length - descriptor.shape[0],))
        descriptor = np.concatenate((descriptor, padding))
    elif descriptor.shape[0] > max_descriptor_length:
        descriptor = descriptor[:max_descriptor_length]

    return descriptor


def generate_orb_descriptor(iris_region):
    max_descriptor_length = 1000

    orb = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = orb.detectAndCompute(iris_region, None)
    descriptor = descriptors.flatten()

    if descriptor.shape[0] < max_descriptor_length:
        padding = np.zeros((max_descriptor_length - descriptor.shape[0],))
        descriptor = np.concatenate((descriptor, padding))
    elif descriptor.shape[0] > max_descriptor_length:
        descriptor = descriptor[:max_descriptor_length]

    return descriptor


def compare_images(image_path1, image_path2, model, name):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    iris_region1, iris_drawn1 = detect_iris(image1, model)
    iris_region2, iris_drawn2 = detect_iris(image2, model)

    descriptor1_mobilenet = generate_mobilenet_descriptor(iris_region1)
    descriptor2_mobilenet = generate_mobilenet_descriptor(iris_region2)

    descriptor1_sift = generate_sift_descriptor(iris_region1)
    descriptor2_sift = generate_sift_descriptor(iris_region2)

    descriptor1_orb = generate_orb_descriptor(iris_region1)
    descriptor2_orb = generate_orb_descriptor(iris_region2)

    similarity_mobilenet = cosine_similarity([descriptor1_mobilenet], [descriptor2_mobilenet])
    print(f"MobileNetV2 Iris similarity score: {similarity_mobilenet[0][0]}")

    similarity_sift = cosine_similarity([descriptor1_sift], [descriptor2_sift])
    print(f"SIFT Iris similarity score: {similarity_sift[0][0]}")

    similarity_orb = hamming_distance([descriptor1_orb], [descriptor2_orb])
    print(f"ORB Iris similarity score: {similarity_orb}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].imshow(iris_drawn1)
    axes[0].axis('off')

    axes[1].imshow(iris_drawn2)
    axes[1].axis('off')

    plt.figtext(0.5, 0.3, f"{name}\n"
                        f"MobileNetV2 Iris similarity score: {similarity_mobilenet[0][0]}\n"
                        f"SIFT Iris similarity score: {similarity_sift[0][0]}\n"
                        f"ORB Iris similarity score: {similarity_orb}", ha='center')
    plt.show()


def hamming_distance(descriptor1, descriptor2):
    descriptor1_flat = np.array(descriptor1).flatten()
    descriptor2_flat = np.array(descriptor2).flatten()

    distance = hamming(descriptor1_flat, descriptor2_flat)
    similarity = 1 - distance
    return similarity


mobilenet_model = load_model('./models/mobilenetv2_model.h5', custom_objects={'mse': tf.keras.metrics.MeanSquaredError})
vgg16_model = load_model('./models/vgg16_model.h5', custom_objects={'mse': tf.keras.metrics.MeanSquaredError})

image_path1 = './openEDS/S_5/5.png'
image_path2 = './openEDS/S_5/21.png'
image_path3 = './openEDS/S_130/0.png'
image_path4 = './openEDS/S_81/1.png'
image_path5 = './openEDS/S_118/0.png'
image_path6 = './openEDS/S_110/6.png'
image_path7 = './openEDS/S_99/4.png'
image_path8 = './openEDS/S_79/1.png'


compare_images(image_path1, image_path2, mobilenet_model, 'MobileNetV2')
compare_images(image_path1, image_path2, vgg16_model, 'VGG16')

compare_images(image_path3, image_path4, mobilenet_model, 'MobileNetV2')
compare_images(image_path3, image_path4, vgg16_model, 'VGG16')

compare_images(image_path5, image_path6, mobilenet_model, 'MobileNetV2')
compare_images(image_path5, image_path6, vgg16_model, 'VGG16')

compare_images(image_path7, image_path8, mobilenet_model, 'MobileNetV2')
compare_images(image_path7, image_path8, vgg16_model, 'VGG16')

