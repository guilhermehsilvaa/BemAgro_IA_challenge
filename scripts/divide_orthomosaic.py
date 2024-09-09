import os
import argparse
import cv2

def divide_image(image_path, output_dir, block_size):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_height, img_width = img.shape[:2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    block_num = 1
    for top in range(0, img_height, block_size):
        for left in range(0, img_width, block_size):
            bottom = min(top + block_size, img_height)
            right = min(left + block_size, img_width)

            block = img[top:bottom, left:right]

            block_filename = os.path.join(output_dir, f"image_{block_num:04d}.png")

            cv2.imwrite(block_filename, block)

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