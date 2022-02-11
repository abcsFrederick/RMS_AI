import torch
import torch.nn as nn


import openslide as op
import argparse
import numpy as np
from skimage.io import imread
import time
import random
import os, glob
from albumentations import Resize
import xlsxwriter
import gc
import timm
import yaml

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def reset_seed(seed):
    """
    ref: https://forums.fast.ai/t/accumulating-gradients/33219/28
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse():
    parser = argparse.ArgumentParser(description='PyTorch RMS Training')
    parser.add_argument('-k', '--kfold', default=1, type=int, metavar='N',
                        help='set the K-Fold interation number (default: 0)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')

    args = parser.parse_args()
    return args


def convert_to_tensor(batch, image_size):
    num_images = batch.shape[0]
    tensor = torch.zeros((num_images, 3, image_size, image_size), dtype=torch.uint8).cuda(non_blocking=True)

    mean = torch.tensor([0.0, 0.0, 0.0]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([255.0, 255.0, 255.0]).cuda().view(1, 3, 1, 1)

    for i, img in enumerate(batch):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] = torch.from_numpy(nump_array)

    tensor = tensor.float()
    tensor = tensor.sub_(mean).div_(std)
    return tensor


def load_best_model(model, path_to_model, best_prec1=0.0):
    if os.path.isfile(path_to_model):
        print("=> loading checkpoint '{}'".format(path_to_model))
        checkpoint = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}), best_precision {}"
              .format(path_to_model, checkpoint['epoch'], best_prec1))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(path_to_model))


class Classifier(nn.Module):
    def __init__(self, n_classes, numgenes):
        super(Classifier, self).__init__()
        self.effnet = timm.create_model('resnet18d', pretrained=True)
        in_features = 1000
        hazard_func = 1

        self.final_act = nn.Tanh()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.alpha_dropout = nn.AlphaDropout(0.25)
        self.l0 = nn.Linear(in_features, 64, bias=True)
        self.l1 = nn.Linear(numgenes, 64, bias=True)
        self.l2 = nn.Linear(64, 64, bias=True)
        self.l3 = nn.Linear(128, hazard_func, bias=True)

    def forward(self, input, gene_muts):
        x = self.effnet(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l0(x)
        x = self.relu(x)  # 64
        x = self.dropout(x)

        y = self.l1(gene_muts)
        y = self.elu(y)
        y = self.alpha_dropout(y)
        y = self.l2(y)
        y = self.elu(y)  # 64
        y = self.alpha_dropout(y)

        z = torch.cat((x, y), dim=1)
        z = self.l3(z)
        z = self.final_act(z)

        return z

def smart_sort(x, permutation):
    ret = x[permutation]
    return ret

def survival_test():
    reset_seed(1)
    args = parse()
    yam_file = open('./arguments.yaml')
    parsed_yaml = yaml.load(yam_file, Loader=yaml.FullLoader)
    num_classes = parsed_yaml['survival']['num_classes']

    weight_path = parsed_yaml['survival']['weight1'] + 'Weight_' + '%02d' % (args.kfold) + '/'
    saved_weights_list = sorted(glob.glob(weight_path + '*.tar'))
    print(saved_weights_list)

    torch.backends.cudnn.benchmark = True

    ## Model instantiation and load model weight.
    ## Currently, my default setup is using 4 GPUs and batch size is 400
    ## Verified with 1 GPU and with the same batch_size of 400
    model = Classifier(num_classes, 4)
    model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()
    model = load_best_model(model, saved_weights_list[-1], 0.)
    print('Loading model is finished!!!!!!!')

    ## inference test svs image and calculate area under the curve
    cal_risk(model, args, parsed_yaml)

def cal_risk(model, args, parsed_yaml):
    model.eval()

    # Task dependent arguments
    IMAGE_SIZE = parsed_yaml['survival']['image_size']
    input_path = parsed_yaml['survival']['image']
    output_path = parsed_yaml['survival']['results']
    cancer_map_path = parsed_yaml['survival']['cancer_maps']

    ## Read input WSIs to be inferenced
    test_ids = sorted(glob.glob(input_path + '*.tif'))
    if len(test_ids) == 0:
        test_ids = sorted(glob.glob(input_path + '*.svs'))
        assert len(test_ids) > 0, "Cannot find svs or tif image"

    print(len(test_ids))

    ## Save results in excel file
    workBook_name = output_path + 'inference_model1_weight_' + '%02d' % (args.kfold) + '.xlsx'
    workBook = xlsxwriter.Workbook(workBook_name, {'nan_inf_to_errors': True})
    workSheet = workBook.add_worksheet('Results')

    workSheet.write(0, 0, 'File ID')
    workSheet.write(0, 1, 'Risk')

    writing_row = 1

    batch_size = args.batch_size

    patients_2nd = torch.zeros(len(test_ids), 1)
    medians = torch.zeros(4096 // batch_size)
    other_index = 0

    for i in range(len(test_ids)):
        file_path = test_ids[i]

        ## imgFile_id => WSI file path's basename
        imgFile_id = os.path.splitext(os.path.basename(file_path))[0]

        ## WSI's segmentation mask (extract patches only from cancerous regions)
        label_path = cancer_map_path + imgFile_id + '.png'

        ## Read WSI
        wholeslide = op.OpenSlide(file_path)

        ## Level 0 optical magnification
        ## If it is 40.0, extract larger patches (IMAGE_SIZE*2) and downsize
        ## If it is 20.0, extract IMAGE_SIZE patch

        objective = float(wholeslide.properties[op.PROPERTY_NAME_OBJECTIVE_POWER])
        print(imgFile_id + ' Objective is: ', objective)
        assert objective >= 20.0, "Level 0 Objective should be greater than 20x"

        ## Extract WSI height and width
        sizes = wholeslide.level_dimensions[0]
        image_height = sizes[1]
        image_width = sizes[0]

        ## Resize WSI's segmentation mask to WSI's size
        label_org = imread(label_path)
        aug = Resize(p=1.0, height=image_height, width=image_width)
        augmented = aug(image=label_org, mask=label_org)
        label = augmented['mask']

        # If the Level 0 objective is 40.0

        for k in range(4096 // batch_size):
            x_coord = 0
            y_coord = 0

            patch_index = 0
            image_batch = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
            gene_batch = torch.zeros((batch_size, 4), dtype=torch.uint8).cuda(non_blocking=True)

            if objective == 40.0:
                image_width_start = 0
                image_width_end = image_width - IMAGE_SIZE * 2 - 1

                image_height_start = 0
                image_height_end = image_height - IMAGE_SIZE * 2 - 1

                for j in range(batch_size):
                    picked = False
                    while (picked == False):
                        x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                        y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                        label_patch = label[y_coord:y_coord + IMAGE_SIZE * 2, x_coord:x_coord + IMAGE_SIZE * 2]

                        if (np.sum(label_patch // 255) > int(IMAGE_SIZE * 2 * IMAGE_SIZE * 2 * 0.50)) and (
                                np.sum(label_patch == 127) == 0):
                            picked = True
                        else:
                            picked = False

                    read_region = wholeslide.read_region((x_coord, y_coord), 0, (IMAGE_SIZE * 2, IMAGE_SIZE * 2))
                    large_image_patch = np.asarray(read_region)[:, :, :3]
                    image_aug = Resize(p=1.0, height=IMAGE_SIZE, width=IMAGE_SIZE)
                    image_augmented = image_aug(image=large_image_patch)
                    image_patch = image_augmented['image']

                    image_batch[patch_index, :, :, :] = image_patch
                    gene_batch[patch_index, :] = torch.from_numpy(np.asarray([0, 0, 0, 0]))
                    patch_index += 1

            elif objective == 20.0:
                image_width_start = 0
                image_width_end = image_width - IMAGE_SIZE - 1

                image_height_start = 0
                image_height_end = image_height - IMAGE_SIZE - 1

                for j in range(batch_size):
                    picked = False
                    while (picked == False):
                        x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                        y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                        label_patch = label[y_coord:y_coord + IMAGE_SIZE, x_coord:x_coord + IMAGE_SIZE]

                        if (np.sum(label_patch // 255) > int(IMAGE_SIZE * IMAGE_SIZE * 0.50)) and (
                                np.sum(label_patch == 127) == 0):
                            picked = True
                        else:
                            picked = False

                    read_region = wholeslide.read_region((x_coord, y_coord), 0, (IMAGE_SIZE, IMAGE_SIZE))
                    image_patch = np.asarray(read_region)[:, :, :3]

                    image_batch[patch_index, :, :, :] = image_patch
                    gene_batch[patch_index, :] = torch.from_numpy(np.asarray([0, 0, 0, 0]))
                    patch_index += 1

            with torch.no_grad():
                image_tensor = convert_to_tensor(image_batch, IMAGE_SIZE)
                gene_tensor = gene_batch.float()
                logits = model(image_tensor, gene_tensor)
                permu = torch.argsort(logits, dim=0, descending=False)

                logits = logits.view(-1)
                logits = smart_sort(logits, permu)
                logits = logits.view(-1, 1)

            median_index = k % (4096 // batch_size)
            medians[median_index] = logits[batch_size // 2, 0]

            if k % (4096 // batch_size) == (4096 // batch_size - 1):
                mediansSorted, _ = medians.sort()
                patients_2nd[other_index, 0] = mediansSorted[1]

                workSheet.write(writing_row, 0, imgFile_id)
                workSheet.write(writing_row, 1, patients_2nd[other_index, 0].item())

                writing_row += 1
                other_index += 1
                medians[:] = 0.

        gc.collect()
    workBook.close()

   
if __name__ == '__main__':
    survival_test()

# CUDA_VISIBLE_DEVICES=0 python survival_inference1.py -k 1 --b 512

# 11_Test_01_Valid_02
# 18_Test_03_Valid_01
# PAMUXL-0BGAOY
# PAPBXZ-0BMUDT
# PAUCAX-0BGD4U

# PASJIW-0BPNQT_1B
# PAKTYJ-0BNA2E_1B
# PAUCLD-0BN9AA
