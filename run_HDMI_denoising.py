import utils
from HDMI import HDMI
import torch
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='paths of an image')
parser.add_argument('-s', '--stdv', type=int,default=0.2, help="standard deviation of the noise (default: 0.2)")
parser.add_argument('-w', '--patch_size', type=int,default=10, help="patch size (default: 10)")
parser.add_argument('-n', '--n_iter', type=int, default=100, help="number of iterations of EM algorithm(default: 100)")
parser.add_argument('-k', '--n_groups', type=int, default=40, help="number of groups in the mixture model")
parser.add_argument('--gpu', action='store_true', help='use GPU if possible')
parser.add_argument('--verbose', action='store_true', help='increase verbosity')
args = parser.parse_args()

# select device
if args.gpu:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('cpu')
print('selected device: '+str(DEVICE))

# import image
img = utils.imread(args.image_path, device=DEVICE)

# store image name
path = os.path.basename(args.image_path)
img_name = os.path.splitext(path)[0]

# add gaussian white noise
img_noisy = img + args.stdv*torch.randn(img.shape, device=DEVICE)
utils.imsave(img_name+'_noisy.png', img_noisy)

# extract patches
patch_size = [args.patch_size, args.patch_size]
patches = utils.im2pat(img_noisy, patch_size)

# create HDMI model
model = HDMI(patches, args.n_groups, args.stdv**2, device=DEVICE)

# run the EM algorithm
t = time.time()
print('run EM algorithm...')
for it in range(args.n_iter):
    model.E_step()
    model.M_step()
    if args.verbose:
        print('iteration '+str(it)+'/'+str(args.n_iter)+' done in '+str(int(time.time()-t))+'s')
        
# denoise the patches        
print('denoise patches...')
patches_denoised = model.denoise()

# patch aggregation
img_denoised = utils.pat2im(patches_denoised, img_noisy.shape[2:], patch_size)

# show and save result
print('total runtime: '+str(int(time.time()-t))+'s')
print('PSNR: '+str(utils.psnr(img_denoised,img))+'dB')
utils.imsave(img_name+'_denoised.png', img_denoised)
utils.imshow(img_denoised)