from os import listdir
from os.path import isfile, join
from multiprocessing import Pool, cpu_count
import cv2
import pylab as plt

from segment_image import get_lines, align_image

images_path = 'binarized'


def process_image(image):
    i, image_file = image
    img_path = join(images_path, image_file)
    img = cv2.imread(img_path, 0)

    img, _, _ = align_image(img)

    lines = get_lines(img)

    plt.imshow(img)
    plt.set_cmap('gray')

    height, width = img.shape
    for line in lines:
        plt.plot([0, width], [line, line])

    plt.savefig(join('output', 'lines_' + image_file.split('.')[0]))
    plt.show()
    print(f"[{i + 1}] Processed file {image_file}")


def run():
    binarized_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    p = Pool(cpu_count())
    p.map(process_image, enumerate(binarized_files))
    # p.map(process_image, enumerate(['P166-Fg007-R-C01-R01-binarized.jpg']))


if __name__ == '__main__':
    run()
