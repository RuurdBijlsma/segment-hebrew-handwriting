from os import listdir
from os.path import isfile, join
from multiprocessing import Pool, cpu_count
import cv2
import pylab as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import math
from line_star import LineStar

from segment_image import get_lines, align_image

images_path = 'binarized'


def process_image(image):
    i, image_file = image
    img_path = join(images_path, image_file)
    img = cv2.imread(img_path, 0)

    img, _, _ = align_image(img)

    lines = get_lines(img)

    height, width = img.shape
    patches = []
    fig, ax = plt.subplots(1)

    # Astar line bounds
    for i, line in enumerate(lines[:-1]):
        crop_top = line  # Lower value than crop_bottom, because (0,0) is top left
        crop_bottom = lines[i + 1]
        crop = img[crop_top:crop_bottom, :]
        crop_height, crop_width = crop.shape
        resize_factor = 1/16
        crop = cv2.resize(crop, (int(crop_width * resize_factor), int(crop_height * resize_factor)))

        # plt.imshow(crop, cmap='gray')
        # plt.show()

        crop_height, crop_width = crop.shape
        crop_mid = math.floor(crop_height / 2)
        start = (0, crop_mid)
        goal = (crop_width - 1, crop_mid)
        line_star = LineStar(crop)
        path = line_star.get_path(start, goal)
        if path is None:
            continue

        original_size_path = [(x / resize_factor, crop_top + y / resize_factor) for x, y in path]
        path_coords = list(zip(*original_size_path))
        ax.plot(*path_coords)

    # Horizontal line bounds
    for i, line in enumerate(lines):
        prev_line = lines[i - 1] if i > 0 else None
        next_line = lines[i + 1] if i < len(lines) - 1 else line + (line - prev_line) * 3
        if prev_line is None:
            prev_line = line - (next_line - line) * 3
        bounds = [line - (line - prev_line) / 2, line + (next_line - line) / 2]
        rect = mpatches.Rectangle((0, bounds[0]), width, bounds[1] - bounds[0], ec="none")
        patches.append(rect)

    # Plot line centers
    # for line in lines:
    #     ax.plot([0, width], [line, line])

    # Plot line bounds
    pc = PatchCollection(patches, linewidths=1, edgecolor='red', facecolor='none')
    # ax.add_collection(pc)

    ax.imshow(img, cmap='gray')
    plt.savefig(join('output', 'lines_' + image_file.split('.')[0]))
    plt.show()
    print(f"[{i + 1}] Processed file {image_file}")


def run():
    binarized_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    p = Pool(cpu_count())
    p.map(process_image, enumerate(binarized_files))
    # p.map(process_image, enumerate(['P166-Fg007-R-C01-R01-binarized.jpg']))


def test():
    process_image((0, 'P123-Fg001-R-C01-R01-binarized.jpg'))


if __name__ == '__main__':
    # test()
    run()
