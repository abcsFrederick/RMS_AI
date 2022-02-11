import argparse
import os, glob
import random
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from skimage.io import imread
import xlsxwriter
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from efficientnet_pytorch import EfficientNet

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


def convert_to_tensor(batch, IMAGE_SIZE):
    num_images = batch.shape[0]
    tensor = torch.zeros((num_images, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8).cuda(non_blocking=True)

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
        self.effnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.l1 = nn.Linear(1000, 512, bias=True)  # 6 is number of classes
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

def classify_subtype(model, args, parsed_yaml):
    model.eval()
    ml = nn.Softmax(dim=1)

    # Task dependent arguments
    IMAGE_SIZE = parsed_yaml['subtype']['image_size']
    input_path = parsed_yaml['subtype']['image']
    output_path = parsed_yaml['subtype']['results']

    ## Read input WSIs to be inferenced
    test_ids = sorted(glob.glob(input_path + '*.png'))
    assert len(test_ids) > 0, "Cannot find 10x optical magnification png images"

    print('Number of files: ', len(test_ids))

    ## Save results in excel file
    workBook_name = output_path + 'inference_Fold_' + '%02d' % (args.kfold) + '.xlsx'
    workBook = xlsxwriter.Workbook(workBook_name, {'nan_inf_to_errors': True})
    workSheet = workBook.add_worksheet('Results')

    workSheet.write(0, 0, 'File Name')
    workSheet.write(0, 1, 'Total # of Predictions')
    workSheet.write(0, 2, 'Mean ARMS Score')
    workSheet.write(0, 3, 'Mean ERMS Score')
    workSheet.write(0, 4, 'Prediction')

    writing_row = 1

    for m in range(len(test_ids)):
        file_path = test_ids[m]

        ## imgFile_id => WSI file path's basename
        imgFile_id = os.path.splitext(os.path.basename(file_path))[0]

        ## Read WSI
        wholeslide = imread(file_path)

        image_height = wholeslide.shape[0]
        image_width = wholeslide.shape[1]

        batch_size = args.batch_size
        batch_count = 0

        total_predictions = 0.0
        total_num_arms = 0.0
        total_num_erms = 0.0

        arms_scores = []
        erms_scores = []

        test_patch = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)
        test_patch[:, 0, :, :] = 255
        test_patch[:, 1, :, :] = 255
        test_patch[:, 2, :, :] = 255

        sliding_window_size = (IMAGE_SIZE) // 4
        heights = (image_height - IMAGE_SIZE) // sliding_window_size + 1
        widths = (image_width - IMAGE_SIZE) // sliding_window_size + 1

        for h in range(heights):
            for w in range(widths):
                testing = wholeslide[h * sliding_window_size:h * sliding_window_size + IMAGE_SIZE,
                          w * sliding_window_size:w * sliding_window_size + IMAGE_SIZE, 0:3]
                if (np.average(testing) < 235.0):
                    test_patch[batch_count, :, :, :] = testing
                    batch_count += 1

                if (batch_count == batch_size):
                    with torch.no_grad():
                        tensor_test_patch = convert_to_tensor(test_patch[0:batch_count, :, :, :], IMAGE_SIZE)
                        batch_outcomes = model(tensor_test_patch)
                        _, predicted = torch.max(batch_outcomes.data, 1)
                        batch_probs = ml(batch_outcomes)

                    for k in range(batch_count):
                        arms_scores.append(batch_probs[k][0].item())
                        erms_scores.append(batch_probs[k][1].item())

                    total_num_arms += (predicted == 0).sum().item()
                    total_num_erms += (predicted == 1).sum().item()
                    total_predictions += batch_count

                    test_patch[:, 0, :, :] = 255
                    test_patch[:, 1, :, :] = 255
                    test_patch[:, 2, :, :] = 255

                    batch_count = 0

        if (batch_count > 0):
            with torch.no_grad():
                tensor_test_patch = convert_to_tensor(test_patch[0:batch_count, :, :, :], IMAGE_SIZE)
                batch_outcomes = model(tensor_test_patch)
                _, predicted = torch.max(batch_outcomes.data, 1)
                batch_probs = ml(batch_outcomes)

            # print('Batch predicted shape: ', predicted.shape)
            for k in range(batch_count):
                arms_scores.append(batch_probs[k][0].item())
                erms_scores.append(batch_probs[k][1].item())

            total_num_arms += (predicted == 0).sum().item()
            total_num_erms += (predicted == 1).sum().item()
            total_predictions += batch_count

            test_patch[:, 0, :, :] = 255
            test_patch[:, 1, :, :] = 255
            test_patch[:, 2, :, :] = 255
            batch_count = 0

        print('Number of Predictions: ', total_predictions)
        print('Average ARMS: ', np.average(arms_scores))
        print('Average ERMS: ', np.average(erms_scores))

        final_indice = np.argmax([np.average(arms_scores), np.average(erms_scores)])
        final_prediction = None

        if final_indice == 0:
            final_prediction = 'ARMS'
        if final_indice == 1:
            final_prediction = 'ERMS'

        print('Final Prediction ', final_prediction)
        print('')

        workSheet.write(writing_row, 0, imgFile_id)
        workSheet.write(writing_row, 1, total_predictions)
        workSheet.write(writing_row, 2, np.average(arms_scores))
        workSheet.write(writing_row, 3, np.average(erms_scores))
        workSheet.write(writing_row, 4, final_prediction)

        writing_row += 1

    workBook.close()

def subtype_test():
    reset_seed(1)
    args = parse()
    yam_file = open('./arguments.yaml')
    parsed_yaml = yaml.load(yam_file, Loader=yaml.FullLoader)
    num_classes = parsed_yaml['subtype']['num_classes']

    weight_path = parsed_yaml['subtype']['weight'] + 'Fold_' + '%02d' % (args.kfold) + '/'
    saved_weights_list = sorted(glob.glob(weight_path + '*.tar'))
    print(saved_weights_list)

    torch.backends.cudnn.benchmark = True

    # Model instantiation and load model weight.
    model = Classifier(num_classes)
    model.eval()
    model = nn.DataParallel(model)
    model = model.cuda()
    model = load_best_model(model, saved_weights_list[-1], 0.)
    print('Loading model is finished!!!!!!!')

    classify_subtype(model, args, parsed_yaml)


if __name__ == '__main__':
    subtype_test()

# CUDA_VISIBLE_DEVICES=0 python subtype_inference.py -k 4 --b 64