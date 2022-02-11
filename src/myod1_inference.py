import torch
import torch.nn as nn
import openslide as op
import argparse
import numpy as np
from skimage.io import imread
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
    parser.add_argument('--upperT', default=0.99, type=float, metavar='N',
                        help='upper_threshold')
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
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.effnet = timm.create_model('seresnet50', pretrained=True)
        in_features = 1000

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.alpha_dropout = nn.AlphaDropout(0.25)
        self.l0 = nn.Linear(in_features, 64, bias=True)
        self.l1 = nn.Linear(64, n_classes, bias=True)

    def forward(self, input):
        x = self.effnet(input)
        x = self.elu(x)
        x = self.alpha_dropout(x)
        x = self.l0(x)
        x = self.elu(x)  # 64
        x = self.alpha_dropout(x)
        x = self.l1(x)

        return x

def myod1_test():
    reset_seed(1)
    args = parse()
    yam_file = open('./arguments.yaml')
    parsed_yaml = yaml.load(yam_file, Loader=yaml.FullLoader)
    num_classes = parsed_yaml['myod1']['num_classes']

    ## Three different networks ensembled. You can use other network using command line "--k 1 or 2 or 3"
    weight_path = parsed_yaml['myod1']['weight'] + 'Fold_' + '%02d' % (args.kfold) + '/'
    saved_weights_list = sorted(glob.glob(weight_path + '*.tar'))
    print(saved_weights_list)

    torch.backends.cudnn.benchmark = True

    ## Model instantiation and load model weight.
    ## Currently, my default setup is using 4 GPUs and batch size is 400
    ## Verified with 1 GPU and with the same batch_size of 400
    model = Classifier(num_classes)
    model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()
    model = load_best_model(model, saved_weights_list[-1], 0.)
    print('Loading model is finished!!!!!!!')

    ## inference test svs image and calculate area under the curve
    cal_pos_score(model, args, parsed_yaml)

def cal_pos_score(model, args, parsed_yaml):
    model.eval()
    ml = nn.Softmax(dim=1)
 
    # Task dependent arguments
    IMAGE_SIZE = parsed_yaml['myod1']['image_size']
    input_path = parsed_yaml['myod1']['image']
    output_path = parsed_yaml['myod1']['results']
    cancer_map_path = parsed_yaml['myod1']['cancer_maps']

    ## Read input WSIs to be inferenced
    test_ids = sorted(glob.glob(input_path + '*.tif'))
    if len(test_ids) == 0:
        test_ids = sorted(glob.glob(input_path + '*.svs'))
        assert len(test_ids) > 0, "Cannot find svs or tif image"

    print(len(test_ids))

    ## Save results in excel file
    workBook_name = output_path + 'myod1_inference_Fold_' + '%02d' % (args.kfold) + '.xlsx'
    workBook = xlsxwriter.Workbook(workBook_name, {'nan_inf_to_errors': True})
    workSheet = workBook.add_worksheet('Results')

    workSheet.write(0, 0, 'File ID')
    workSheet.write(0, 1, 'Positive Score')

    writing_row = 1

    ## Patient, labels variables
    patients = np.zeros(len(test_ids))
    other_index = 0

    ## Variables to calculate final outcome value
    correct_count = np.zeros(2)
    correct_probs = np.zeros(2)

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
       
        ## If the Level 0 objective is 40.0
        if objective==40.0:
        ## Retrieve patches from WSI by batch_size but extract no more than 4096 patches
            for k in range(4096 // args.batch_size):
                image_width_start = 0
                image_width_end = image_width - IMAGE_SIZE*2 - 1

                image_height_start = 0
                image_height_end = image_height - IMAGE_SIZE*2 - 1

                x_coord = 0
                y_coord = 0

                patch_index = 0
                image_batch = np.zeros((args.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)

                ## Extract batch_size patches from WSI within cancerous regions
                for j in range(args.batch_size):
                    picked = False

                    while (picked == False):
                        ## Pick random locations withint segmentation masks first
                        x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                        y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                        label_patch = label[y_coord:y_coord + IMAGE_SIZE*2, x_coord:x_coord + IMAGE_SIZE*2]

                        ## Examine whether the random coordinates are within cancerous regions
                        ## If the coordinates are containing enough cancerous region 'picked = True' and If not 'picked=False'
                        if (np.sum(label_patch // 255) > int(IMAGE_SIZE*2 * IMAGE_SIZE*2 * 0.50)) and (
                                np.sum(label_patch == 127) == 0):
                            picked = True
                        else:
                            picked = False

                    ## Using the picked coordinates, extract corresponding WSI patch
                    ## Store patches in the image_batch so that it can be later inferenced at once
                    read_region = wholeslide.read_region((x_coord, y_coord), 0, (IMAGE_SIZE*2, IMAGE_SIZE*2))
                    large_image_patch = np.asarray(read_region)[:, :, :3]
                    image_aug = Resize(p=1.0, height=IMAGE_SIZE, width=IMAGE_SIZE)
                    image_augmented = image_aug(image=large_image_patch)
                    image_patch = image_augmented['image']
                    image_batch[patch_index, :, :, :] = image_patch
                    patch_index += 1

                with torch.no_grad():
                    ## Convert image_batch to pytorch tensor
                    image_tensor = convert_to_tensor(image_batch, IMAGE_SIZE)

                    ## Inference the image_tensor (as a batch)
                    inst_logits = model(image_tensor)

                    ## Model's outcome are logit values for each patch
                    ## Need to conver them into probabilities of being MYOD1+
                    probs = ml(inst_logits)

                    ## Each patch produces two outcomes, MYOD1- and MYOD1+
                    ## Larger value's index will be the prediction for the patch (0, MYOD1-) (1, MYOD1+)
                    _, preds = torch.max(inst_logits, 1)
                    cbatch_size = len(image_tensor)

                    ## Examine all the patch's probability values
                    ## If predicted outcome's probability is greater than args.upperT, use them in the final calculation
                    ## Which means, if the model's outcome is not confident enough, we do not use them in our final calculation
                    for l in range(cbatch_size):
                        ## preds contains each patch's prediction (either 0 or 1)
                        ## index 0 means MYOD1- and index 1 means MYOD1+
                        index = preds[l].item()

                        ## Check the probability of the prediction
                        ## if it is greater than the threshold, it will be counted
                        ## correct_count: (2, ) shape
                        ## correct_count[0] contains total number of patches that are predicted as MYOD1- and has probability >= threshold
                        ## correct_count[1] contains total number of patches that are predicted as MYOD1+ and has probability >= threshold
                        if probs.data[l, index].item() >= args.upperT:
                            correct_count[index] += 1
                            correct_probs[index] += probs.data[l, index].item()

                ## When it arrives at the last iteration
                if k == ((4096 // args.batch_size) - 1):

                    ## If there are no predictions that are made with high conviction, decision is not made
                    if (np.sum(correct_count) == 0):
                        patients[other_index] = np.nan

                    ## If there are predictions that are made with high conviction, decision is made
                    ## Probability of WSI being predicted as MYOD1+ is as below
                    ## (# high conviction MYOD1+ predictions)/(# total number of high convictions)
                    else:
                        patients[other_index] = 1.0 * correct_count[1] / (correct_count[0] + correct_count[1])

                    workSheet.write(writing_row, 0, imgFile_id)
                    workSheet.write(writing_row, 1, patients[other_index])
                    writing_row += 1

                    other_index += 1
                    correct_count[:] = 0.
                    correct_probs[:] = 0.

        ## If the Level 0 objective is 40.0
        if objective == 20.0:
            ## Retrieve patches from WSI by batch_size but extract no more than 4096 patches
            for k in range(4096 // args.batch_size):
                image_width_start = 0
                image_width_end = image_width - IMAGE_SIZE - 1

                image_height_start = 0
                image_height_end = image_height - IMAGE_SIZE - 1

                x_coord = 0
                y_coord = 0

                patch_index = 0
                image_batch = np.zeros((args.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)

                ## Extract batch_size patches from WSI within cancerous regions
                for j in range(args.batch_size):
                    picked = False

                    while (picked == False):
                        ## Pick random locations withint segmentation masks first
                        x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                        y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                        label_patch = label[y_coord:y_coord + IMAGE_SIZE, x_coord:x_coord + IMAGE_SIZE]

                        ## Examine whether the random coordinates are within cancerous regions
                        ## If the coordinates are containing enough cancerous region 'picked = True' and If not 'picked=False'
                        if (np.sum(label_patch // 255) > int(IMAGE_SIZE * IMAGE_SIZE * 0.50)) and (
                                np.sum(label_patch == 127) == 0):
                            picked = True
                        else:
                            picked = False

                    ## Using the picked coordinates, extract corresponding WSI patch
                    ## Store patches in the image_batch so that it can be later inferenced at once
                    read_region = wholeslide.read_region((x_coord, y_coord), 0,
                                                         (IMAGE_SIZE, IMAGE_SIZE))
                    image_patch = np.asarray(read_region)[:, :, :3]
                    image_batch[patch_index, :, :, :] = image_patch
                    patch_index += 1

                with torch.no_grad():
                    ## Convert image_batch to pytorch tensor
                    image_tensor = convert_to_tensor(image_batch, IMAGE_SIZE)

                    ## Inference the image_tensor (as a batch)
                    inst_logits = model(image_tensor)

                    ## Model's outcome are logit values for each patch
                    ## Need to conver them into probabilities of being MYOD1+
                    probs = ml(inst_logits)

                    ## Each patch produces two outcomes, MYOD1- and MYOD1+
                    ## Larger value's index will be the prediction for the patch (0, MYOD1-) (1, MYOD1+)
                    _, preds = torch.max(inst_logits, 1)
                    cbatch_size = len(image_tensor)

                    ## Examine all the patch's probability values
                    ## If predicted outcome's probability is greater than args.upperT, use them in the final calculation
                    ## Which means, if the model's outcome is not confident enough, we do not use them in our final calculation
                    for l in range(cbatch_size):
                        ## preds contains each patch's prediction (either 0 or 1)
                        ## index 0 means MYOD1- and index 1 means MYOD1+
                        index = preds[l].item()

                        ## Check the probability of the prediction
                        ## if it is greater than the threshold, it will be counted
                        ## correct_count: (2, ) shape
                        ## correct_count[0] contains total number of patches that are predicted as MYOD1- and has probability >= threshold
                        ## correct_count[1] contains total number of patches that are predicted as MYOD1+ and has probability >= threshold
                        if probs.data[l, index].item() >= args.upperT:
                            correct_count[index] += 1
                            correct_probs[index] += probs.data[l, index].item()

                ## When it arrives at the last iteration
                if k == ((4096 // args.batch_size) - 1):

                    ## If there are no predictions that are made with high conviction, decision is not made
                    if (np.sum(correct_count) == 0):
                        patients[other_index] = np.nan

                    ## If there are predictions that are made with high conviction, decision is made
                    ## Probability of WSI being predicted as MYOD1+ is as below
                    ## (# high conviction MYOD1+ predictions)/(# total number of high convictions)
                    else:
                        patients[other_index] = 1.0 * correct_count[1] / (correct_count[0] + correct_count[1])

                    workSheet.write(writing_row, 0, imgFile_id)
                    workSheet.write(writing_row, 1, patients[other_index])
                    writing_row += 1

                    other_index += 1
                    correct_count[:] = 0.
                    correct_probs[:] = 0.
    
    gc.collect()
    workBook.close()


if __name__ == '__main__':
    myod1_test()

# CUDA_VISIBLE_DEVICES=0 python myod1_inference.py -k 3 --upperT 0.99 --b 256
