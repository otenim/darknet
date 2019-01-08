import cv2
import argparse
import os
import tqdm
import imghdr
import darknet


curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('images_dirpath')
parser.add_argument('cfg_path')
parser.add_argument('data_path')
parser.add_argument('weights_path')
parser.add_argument('--prefix', type=str, default='capture')
parser.add_argument('--origindex', type=int, default=1)
parser.add_argument('--imgfmt', type=str, default='jpg')


def main(args):
    args.images_dirpath = os.path.abspath(os.path.expanduser(args.images_dirpath))
    args.cfg_path = os.path.abspath(os.path.expanduser(args.cfg_path))
    args.data_path = os.path.abspath(os.path.expanduser(args.data_path))
    args.weights_path = os.path.abspath(os.path.expanduser(args.weights_path))

    # parse .data file
    names_path = None
    with open(args.data_path) as f:
        lines = f.readlines()
        for line in lines:
            # <type> = <path>
            type, path = line.split('=')
            type = type.strip()
            path = path.strip()
            if type == 'names':
                names_path = path

    # load names file
    names = []
    with open(names_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            names.append(line)
    print('names: %s.' % str(names))

    # load detector
    print('loading detector...')
    net = darknet.load_net(args.cfg_path.encode(), args.weights_path.encode(), 0)
    meta = darknet.load_meta(args.data_path.encode())
    print('the detector was successfully loaded.')

    # count images
    n_images = 0
    for path in os.listdir(args.images_dirpath):
        path = os.path.join(args.images_dirpath, path)
        if imghdr.what(path) == None:
            continue
        n_images += 1
    n_digits = len(str(n_images))
    print('%s input images were loaded.' % n_images)

    # check images
    image_paths = []
    print('validating input images...')
    for i in range(args.origindex, n_images + args.origindex):
        path = os.path.join(args.images_dirpath, args.prefix + str(i).zfill(n_digits) + '.' + args.imgfmt)
        if os.path.exists(path) == False:
            raise Exception('%s was not found.' % path)
        image_paths.append(path)
    print('all the input images are valid.')

    # make video
    print('creating video...')
    pbar = tqdm.tqdm(total=len(image_paths))
    for path in image_paths:
        src_img = cv2.imread(path)
        bboxes = darknet.detect(net, meta, path.encode())
        pbar.update()
    pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
