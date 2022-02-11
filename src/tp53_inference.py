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
from efficientnet_pytorch import EfficientNet
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
        self.effnet = EfficientNet.from_pretrained('efficientnet-b3')
        self.l1 = nn.Linear(1000, 512, bias=True) # 6 is number of classes
        self.l2 = nn.Linear(512, n_classes, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.non_linear = nn.LeakyReLU()

    def forward(self, input):
        x = self.effnet(input)
        x = self.l1(x)
        x = self.non_linear(x)
        x = self.drop_out(x)
        x = self.l2(x)
        return x

def tp53_test():
    reset_seed(1)
    args = parse()
    yam_file = open('./arguments.yaml')
    parsed_yaml = yaml.load(yam_file, Loader=yaml.FullLoader)
    num_classes = parsed_yaml['tp53']['num_classes']

    ## Three different networks ensembled. You can use other network using command line "--k 1 or 2 or 3"
    weight_path = parsed_yaml['tp53']['weight'] + 'Fold_' + '%02d' % (args.kfold) + '/'
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
    IMAGE_SIZE = parsed_yaml['tp53']['image_size']
    input_path = parsed_yaml['tp53']['image']
    output_path = parsed_yaml['tp53']['results']
    cancer_map_path = parsed_yaml['tp53']['cancer_maps']

    ## Read input WSIs to be inferenced
    test_ids = sorted(glob.glob(input_path + '*.tif'))
    if len(test_ids) == 0:
        test_ids = sorted(glob.glob(input_path + '*.svs'))
        assert len(test_ids) > 0, "Cannot find svs or tif image"

    print(len(test_ids))

    ## Save results in excel file
    workBook_name = output_path + 'inference_Fold_' + '%02d' % (args.kfold) + '.xlsx'
    workBook = xlsxwriter.Workbook(workBook_name, {'nan_inf_to_errors': True})
    workSheet = workBook.add_worksheet('Results')

    workSheet.write(0, 0, 'File ID')
    workSheet.write(0, 1, 'Total number of predictions')
    workSheet.write(0, 2, 'Positive Score')

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

        batch_size = args.batch_size
        batch_count = 0

        test_patch = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
        total_predictions = 0.0
        positive_score = 0.
        maximum_score = 0.
        total_num_negative_predictions = 0.0
        total_num_positive_predictions = 0.0

        ## If the Level 0 objective is 40.0
        if objective==40.0:
            heights = image_height // IMAGE_SIZE*2
            widths = image_width // IMAGE_SIZE*2
            for h in range(heights):
                for w in range(widths):
                    label_patch = label[h * IMAGE_SIZE*2:(h + 1) * IMAGE_SIZE*2,
                                  w * IMAGE_SIZE*2:(w + 1) * IMAGE_SIZE*2]
                    if (np.sum(label_patch // 255) > int(IMAGE_SIZE*2 * IMAGE_SIZE*2 * 0.50)) and (
                            np.sum(label_patch == 127) == 0):
                        read_region = wholeslide.read_region((w * IMAGE_SIZE*2, h * IMAGE_SIZE*2), 0,
                                                             (IMAGE_SIZE*2, IMAGE_SIZE*2))
                        large_image_patch = np.asarray(read_region)[:, :, :3]
                        image_aug = Resize(p=1.0, height=IMAGE_SIZE, width=IMAGE_SIZE)
                        image_augmented = image_aug(image=large_image_patch)
                        image_patch = image_augmented['image']
                        test_patch[batch_count, :, :, :] = image_patch

                        batch_count += 1

                        if batch_count == batch_size:
                            with torch.no_grad():
                                tensor_test_patch = convert_to_tensor(test_patch, IMAGE_SIZE)
                                # print('Max Input Value: ', torch.max(tensor_test_patch))
                                batch_outcomes = model(tensor_test_patch)
                                _, predicted = torch.max(batch_outcomes.data, 1)
                                ml = nn.Softmax(dim=1)
                                probs = ml(batch_outcomes)
                                # print('Probs: ', probs)

                                negative_predictions = (predicted == 0).sum().item()
                                positive_predictions = (predicted == 1).sum().item()
                                total_num_negative_predictions += negative_predictions
                                total_num_positive_predictions += positive_predictions

                                for m in range(batch_count):
                                    positive_score += probs.data[m, 1].item()
                                    maximum_score = max(probs.data[m, 1].item(), maximum_score)

                            total_predictions += batch_count
                            batch_count = 0
                            test_patch[:, :, :, :] = 0

        elif objective == 20.0:
            heights = image_height // IMAGE_SIZE
            widths = image_width // IMAGE_SIZE
            for h in range(heights):
                for w in range(widths):
                    label_patch = label[h * IMAGE_SIZE:(h + 1) * IMAGE_SIZE,
                                  w * IMAGE_SIZE:(w + 1) * IMAGE_SIZE]
                    if (np.sum(label_patch // 255) > int(IMAGE_SIZE * IMAGE_SIZE * 0.50)) and (
                            np.sum(label_patch == 127) == 0):
                        read_region = wholeslide.read_region((w * IMAGE_SIZE, h * IMAGE_SIZE), 0,
                                                             (IMAGE_SIZE, IMAGE_SIZE))
                        image_patch = np.asarray(read_region)[:, :, :3]
                        test_patch[batch_count, :, :, :] = image_patch

                        batch_count += 1

                        if batch_count == batch_size:
                            with torch.no_grad():
                                tensor_test_patch = convert_to_tensor(test_patch, IMAGE_SIZE)
                                # print('Max Input Value: ', torch.max(tensor_test_patch))
                                batch_outcomes = model(tensor_test_patch)
                                _, predicted = torch.max(batch_outcomes.data, 1)
                                ml = nn.Softmax(dim=1)
                                probs = ml(batch_outcomes)
                                # print('Probs: ', probs)

                                negative_predictions = (predicted == 0).sum().item()
                                positive_predictions = (predicted == 1).sum().item()
                                total_num_negative_predictions += negative_predictions
                                total_num_positive_predictions += positive_predictions

                                for m in range(batch_count):
                                    positive_score += probs.data[m, 1].item()
                                    maximum_score = max(probs.data[m, 1].item(), maximum_score)

                            total_predictions += batch_count
                            batch_count = 0
                            test_patch[:, :, :, :] = 0
 
        if batch_count != 0:
            with torch.no_grad():
                tensor_test_patch = convert_to_tensor(test_patch[0:batch_count, :, :, :], IMAGE_SIZE)
                # print('Max Input Value: ', torch.max(tensor_test_patch))
                batch_outcomes = model(tensor_test_patch)
                _, predicted = torch.max(batch_outcomes.data, 1)
                ml = nn.Softmax(dim=1)
                probs = ml(batch_outcomes)
                # print('Probs: ', probs)

                negative_predictions = (predicted == 0).sum().item()
                positive_predictions = (predicted == 1).sum().item()
                total_num_negative_predictions += negative_predictions
                total_num_positive_predictions += positive_predictions

                for m in range(batch_count):
                    positive_score += probs.data[m, 1].item()
                    maximum_score = max(probs.data[m, 1].item(), maximum_score)

            total_predictions += batch_count
            batch_count = 0
            test_patch[:, :, :, :] = 0

        print('Number of Predictions: ', total_predictions)
        positive_score = positive_score * 1.0 / total_predictions
        print('Positive Score: ', positive_score, ' Flag: ', 0)
        print('Maximum Score: ', maximum_score)

        assert total_predictions == (
                    total_num_negative_predictions + total_num_positive_predictions), 'Prediction numbers are different'
        positive_percent = 100.0 * total_num_positive_predictions / (
                total_num_negative_predictions + total_num_positive_predictions)
        print('Positive Percent: ', positive_percent)
        print('')

        workSheet.write(writing_row, 0, imgFile_id)
        workSheet.write(writing_row, 1, total_predictions)
        workSheet.write(writing_row, 2, positive_percent)

        writing_row += 1

    gc.collect()
    workBook.close()

if __name__ == '__main__':
    tp53_test()

# CUDA_VISIBLE_DEVICES=0 python tp53_inference.py -k 2 --b 64
# PASRNA-0BLRX3