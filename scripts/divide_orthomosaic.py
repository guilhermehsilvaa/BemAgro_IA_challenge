import os
import argparse
import cv2
from PIL import Image

def divide_image(image_path, output_dir, block_size):
    with Image.open(image_path) as img:
        img_width, img_height = img.size

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        block_num = 1
        for top in range(0, img_height, block_size):
            for left in range(0, img_width, block_size):
                bottom = min(top + block_size, img_height)
                right = min(left + block_size, img_width)

                block = img.crop((left, top, right, bottom))

                block_filename = os.path.join(output_dir, f"image_{block_num:04d}.png")

                block.save(block_filename, format="PNG")

                block_num += 1

def main():
    parser = argparse.ArgumentParser(description='Divide uma imagem grande em imagens menores.')
    parser.add_argument('--input', type=str, required=True, help='Caminho do arquivo ortomosaico (formato TIFF).')
    parser.add_argument('--output', type=str, required=True, help='Diretório de saída para os blocos de imagem.')
    parser.add_argument('--block-size', type=int, default=320, help='Tamanho das imagens em pixels (padrão: 320).')
    args = parser.parse_args()

    divide_image(args.input, args.output, args.block_size)

if __name__ == "__main__":
    main()