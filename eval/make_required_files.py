import os
import argparse
import darknet
import tqdm
import imghdr
from PIL import Image


curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('cfg_path')
parser.add_argument('data_path')
parser.add_argument('weights_path')
parser.add_argument('--annotations_dirpath', os.path.join(curdir, 'annotations'))
parser.add_argument('--predicted_dirpath', os.path.join(curdir, 'predicted'))


def main(args):
    args.cfg_path = os.path.abspath(os.path.expanduser(args.cfg_path))
    args.data_path = os.path.abspath(os.path.expanduser(args.data_path))
    args.weights_path = os.path.abspath(os.path.expanduser(args.weights_path))
    args.annotations_dirpath = os.path.abspath(os.path.expanduser(args.annotations_dirpath))
    args.predicted_dirpath = os.path.abspath(os.path.expanduser(args.predicted_dirpath))

    if os.path.exists(args.annotations_dirpath) == False:
        os.makedirs(args.annotations_dirpath)
    if os.path.exists(args.predicted_dirpath) == False:
        os.makedirs(args.predicted_dirpath)

    # parse .data file
    valid_path, names_path = None, None
    with open(args.data_path) as f:
        lines = f.readlines()
        for line in lines:
            # <type> = <path>
            type, _, path = line.split('=')
            type = type.strip()
            path = path.strip()
            if type == 'valid':
                valid_path = path
            elif type == 'names':
                names_path = path

    # load test image paths
    test_img_paths = []
    with open(valid_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            test_img_paths.append(line)

    # load test image's annotations
    labels_dirpath = os.path.join(os.path.dirname(os.path.dirname(pathimg_paths[0])), 'labels')
    test_ano_paths = []
    for path in os.listdir(labels_dirpath):
        if path.split('.')[-1] != 'txt':
            continue
        path = os.path.join(labels_dirpath, path)
        test_ano_paths.append(path)

    # load names file
    names = []
    with open(names_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            names.append(line)

    # convert test annotation files and save them
    print('converting darknet-format annotation files...')
    pbar = tqdm.tqdm(total=len(test_ano_paths))
    for path in test_img_paths:
        img = Image.open(path)
        img_w, img_h = img.width, img.height
        img_format = imghdr.what(path)
        src_ano_path = os.path.join(labels_dirpath, os.path.basename(path).replace('.' + img_format, '.txt'))
        dst_ano_path = os.path.join(args.annotations_dirpath, os.path.basename(path).replace('.' + img_format, '.txt'))
        buffer = []
        with open(src_ano_path) as f:
            lines = f.readlines()
            for line in lines:
                cid, xc, yc, w, h = line.split(' ')
                cid = int(cid.strip())
                xc, yc = float(xc.strip()) * img_w, float(yc.strip()) * img_h
                w, h = float(w.strip()) * img_w, float(h.strip()) * img_h
                buffer.append([cid, xc, yc, w, h])
        with open(dst_ano_path, mode='w') as f:
            for b in buffer:
                cid, xc, yc, w, h = b
                cname = names[cid]
                x_left = int(xc - img_w / 2)
                y_top = int(yc - img_h / 2)
                x_right = int(xc + img_w / 2)
                y_bottom = int(yc + img_h / 2)
                f.write('%s %d %d %d %d\n' % (cname, x_left, y_top, x_right, y_bottom))
        pbar.update()
    pbar.close()

    # load detector
    net = darknet.load_net(args.cfg_path.encode(), args.weights_path.encode(), 0)
    meta = darknet.load_meta(args.data_path.encode())

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
