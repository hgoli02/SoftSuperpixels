# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from superpixelf import SuperSmoothSoft as FastSlic
from superpixelf import *

ABSTAIN = -1

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--tau", default=0.5, type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.1, help="failure probability")
parser.add_argument("--seg", type=bool, help="use superpixel smoothing ")
parser.add_argument("--k", type=int, help="k ", default=5)
parser.add_argument("--c", type=int, default=10, help="compactness ")
parser.add_argument("--ns", type=int, help="num superpixels ")
parser.add_argument("--start", type=int, default=0, help="start ")
parser.add_argument("--noise_factor", type=float, default=1)
parser.add_argument("--alg", type=str, default='slic')
args = parser.parse_args()

device = torch.device('cuda:0')

    
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import math 

if __name__ == "__main__":
    NO_CHECKPOINT_ARCHS = {
        'resnet50', 'eva02_l14', 'vit_l_augreg', 'vit_b_augreg', 'deit3_l16', 'deit3_b16', 'vit_base_patch16_224'
    }
    if args.base_classifier in NO_CHECKPOINT_ARCHS:
        base_classifier = get_architecture(args.base_classifier, args.dataset, device=device)
    else:
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset, device=device)
        
        # Handle different checkpoint formats
        state_dict = checkpoint['state_dict']
        
        # Check if checkpoint has normalization layer (keys starting with "0.")
        has_norm_layer = any(key.startswith('0.') for key in state_dict.keys())
        
        if has_norm_layer:
            # Checkpoint includes normalization layer, load directly
            base_classifier.load_state_dict(state_dict)
        else:
            # Checkpoint doesn't have normalization layer
            # Check if keys start with "1." (saved from Sequential[norm, model])
            has_sequential_prefix = any(key.startswith('1.') for key in state_dict.keys())
            
            if has_sequential_prefix:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('1.'):
                        new_key = key[2:]  # Remove "1." 
                        new_state_dict[f'1.{new_key}'] = value
                    else:
                        new_state_dict[f'1.{key}'] = value
                
                norm_state = base_classifier[0].state_dict()
                for key, value in norm_state.items():
                    new_state_dict[f'0.{key}'] = value
                    
                base_classifier.load_state_dict(new_state_dict)
            else:
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_state_dict[f'1.{key}'] = value
                
                norm_state = base_classifier[0].state_dict()
                for key, value in norm_state.items():
                    new_state_dict[f'0.{key}'] = value
                    
                base_classifier.load_state_dict(new_state_dict)
        
        base_classifier.to(device)


    # create the smooothed classifier g
    if args.seg == True:
        print('using super pixel')
        smoothed_classifier = FastSlic(base_classifier, num_classes=get_num_classes(args.dataset), sigma=args.sigma, k=args.k, c=args.c, ns=args.ns, channel=3, tau=args.tau, device=device, noise_factor=args.noise_factor, alg=args.alg)
    else:
        print('using orignal')
        smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma, device=device)

    # create results directory if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # prepare output file path
    outfile_path = os.path.join(results_dir, args.outfile)

    # prepare output file
    f = open(outfile_path, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)


    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split, device=device)
    for i in range(args.start, len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i > args.max and args.max != -1:
            break

        (x, label) = dataset[i]

        before_time = time.time()
        # certify the prediction of g around x
        x = x.to(device)
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch, label)
        # debug_softness(smoothed_classifier, x.unsqueeze(0))
        after_time = time.time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
