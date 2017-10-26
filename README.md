# Fourier Deconvolution Network

The code in this repository implements image deconvolution as described in the paper:

Jakob Kruse, Carsten Rother, and Uwe Schmidt.  
*Learning to Push the Limits of Efficient FFT-based Image Deconvolution.*  
IEEE International Conference on Computer Vision (ICCV), Venice, Italy, October 2017.

Please cite the paper if you are using this code in your research.

### Dependencies

- Compatible with Python 2 and 3.
- Main requirements: Keras 2+ and TensorFlow 1.3+.  
  Install via `pip install -r requirements.txt`.

### Usage

You can see a demonstration of how to use `fdn_predict.py` in the Jupyter notebook [demo.ipynb](demo.ipynb).  
Please "watch" this repository if you want to be notified of further updates.

    $ python fdn_predict.py -h
    Using TensorFlow backend.
    usage: fdn_predict.py [-h] --image IMAGE --kernel KERNEL --sigma SIGMA
                          [--flip-kernel [FLIP_KERNEL]] [--model-dir MODEL_DIR]
                          [--n-stages N_STAGES] [--finetuned [FINETUNED]]
                          [--output OUTPUT] [--save-all-stages [SAVE_ALL_STAGES]]
                          [--quiet [QUIET]]

    optional arguments:
      -h, --help            show this help message and exit
      --quiet [QUIET]       don't print status messages (default: False)

    input:
      --image IMAGE         blurred image (default: None)
      --kernel KERNEL       blur kernel (default: None)
      --sigma SIGMA         standard deviation of Gaussian noise (default: None)
      --flip-kernel [FLIP_KERNEL]
                            rotate blur kernel by 180 degrees (default: False)

    model:
      --model-dir MODEL_DIR
                            path to model (default: models/sigma_1.0-3.0)
      --n-stages N_STAGES   number of model stages to use (default: 10)
      --finetuned [FINETUNED]
                            use finetuned model weights (default: True)

    output:
      --output OUTPUT       deconvolved result image (default: None)
      --save-all-stages [SAVE_ALL_STAGES]
                            save all intermediate results (if finetuned is false)
                            (default: False)

### Note

**We currently do not recommend to run our code on the CPU.**

- While our deconvolution method is inherently efficient, the implementation is currently not optimized for speed.
In particular, fast Fourier transforms (FFTs) in TensorFlow (TF) are not very fast. Running on the CPU is especially slow, since TF seems to use single-threaded FFTs that are much slower than high-performance implementations like [FFTW](http://fftw.org).
- Even more problematic than the issue of much increased runtime, we found that the results of our models can be substantially worse when the code is executed on the CPU. For the sake of reproducible research, please note that our results in the paper were obtained with an NVIDIA Titan X GPU.