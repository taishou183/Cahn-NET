import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/media/StudentGroup/LZ/Dataset/')
parser.add_argument('--model_dir', type=str, default='./trained_models/ITS_0707')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--crop', default=True, action='store_true')
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--cl_lambda', type=float, default=0.25, help='cl_lambda')
parser.add_argument('--loss_weight', type=float, default=0.3, help='clcr_loss weight')
parser.add_argument('--clcrloss', default=True, action='store_true', help='clcr loss')
parser.add_argument('--norm', default=True, action='store_true', help='normalize')
parser.add_argument('--ablation_type', type=int, default=0, help='ablation_type')
parser.add_argument('--clip', action='store_true', help='use grad clip')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')

opt = parser.parse_args()
