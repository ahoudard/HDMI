# HDMI
High-Dimensional Mixture Models For Unsupervised Image Denoising

Code related to the paper https://hal.archives-ouvertes.fr/hal-01544249

  - utils.py contains custom functions imread, imshow, imsave, im2pat, pat2im, and psnr
  - HDMI.py contains the HDMI model
  - syntax python run_HDMI_denoising.py image_path --options

    replace --options with any:

        -s or -stdv standard deviation of the noise
        -w or --patch_size patch size 
        -n or --n_iter iterations of EM algorithm 
        -k or --n_groups number of groups in the mixture model
        --gpu to use GPU if possible results 
        --verbose to print time at each iteration
