import cv2
import os
import argparse

def binarize_image(input_path, output_path):
    img = cv2.imread(input_path)

    # Calcula o Excess Green Index (ExG)
    ExG = 2 * img[:, :, 1] - img[:, :, 2] - img[:, :, 0]

    # Aplica o método de Otsu para binarização
    _, binary = cv2.threshold(ExG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.bitwise_not(binary)

    cv2.imwrite(output_path, binary)

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        binarize_image(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binariza imagens usando ExG e Otsu Threshold.')
    parser.add_argument('--input', required=True, help='Caminho para o diretório com as imagens RGB')
    parser.add_argument('--output', required=True, help='Caminho para o diretório onde serão salvas as imagens binarizadas')
    
    args = parser.parse_args()
    
    process_images(args.input, args.output)