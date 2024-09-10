import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import argparse
import numpy as np
import cv2
import tensorflow as tf
import segmentation_models as sm

def get_model(input_shape, backbone):
    model = sm.Unet(backbone, classes=2, activation='softmax', 
                    input_shape=input_shape)
    return model

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = np.expand_dims(image, axis=0)
    return image_array

def infer(model, image_array):
    prediction = model.predict(image_array)
    return prediction.squeeze()

def postprocess_and_save(prediction, output_path):
    binary_mask = (np.argmax(prediction, axis=-1) * 255).astype(np.uint8)
    cv2.imwrite(output_path, binary_mask)

def main():
    parser = argparse.ArgumentParser(description="Inferência com modelo de segmentação.")
    parser.add_argument('--rgb', required=True, help='Caminho para a imagem RGB.')
    parser.add_argument('--modelpath', required=True, help='Caminho para os pesos do modelo.')
    parser.add_argument('--output', required=True, help='Caminho para salvar a imagem segmentada.')
    args = parser.parse_args()

    model = get_model(input_shape=(320,320,3), backbone='mobilenetv2')

    model.load_weights(args.modelpath)

    image_array = preprocess_image(args.rgb, target_size=(320, 320))

    prediction = infer(model, image_array)

    postprocess_and_save(prediction, args.output)

if __name__ == "__main__":
    main()