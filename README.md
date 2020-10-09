# Neural Contours: Learning to Draw Lines from 3D Shapes

This repository contains the PyTorch implementation for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "Neural Contours: Learning to Draw Lines from 3D Shapes" by [Difan Liu](https://people.cs.umass.edu/~dliu/), Mohamed Nabail, [Aaron Hertzmann](https://www.dgp.toronto.edu/~hertzman/), [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/).

[[Arxiv]](https://arxiv.org/abs/2003.10333)

### Dependency
- The project is developed on Ubuntu 16.04 with cuda9.0 + cudnn7.0. The code has been tested with PyTorch 1.1.0 (GPU version) and Python 3.6.8. 
- Python packages:
    - OpenCV (tested with 4.2.0)
    - PyYAML (tested with 5.3.1)
    - scikit-image (tested with 0.14.2)

### Dataset and Weights
- Pre-trained model is available [here](https://www.dropbox.com/s/nihzuh524oe0zuu/weights.zip?dl=0), please put it in `data/model_weights`:
    ```
    cd data/model_weights
    unzip weights.zip
    ```

- download example testing data:
    ```
    cd data/example
    wget https://people.cs.umass.edu/~dliu/projects/NeuralContours/example.zip
    unzip example.zip
    ```
- training data is available [here](https://www.dropbox.com/s/ufiu97sn4j4h9z0/dataset.zip?dl=0).

    
### Differentiable Geometry Branch
- we use [rtsc-1.6](https://gfx.cs.princeton.edu/proj/sugcon/) to compute all the input geometric feature maps and lines. See [here](https://github.com/DifanLiu/NeuralContours/blob/master/data/README.md) for details.
- run geometry branch without NRM (Neural Ranking Module), this script takes thresholds of geometric lines as input:
    ```python
    python -m scripts.geometry_branch_demo -sc 10.0 -r 10.0 -v 10.0 -ar 0.1 -model_name bumps_a -save_name data/output/bumps_a.png
    ```

### Testing with NRM and ITB (Image Translation Branch)
- Testing with NRM and ITB:
    ```python
    python -m scripts.test -model_name bumps_a -save_name data/output/bumps_a_NCs.png
    ```
    Note that computation time depends on GPU performance, parameter setting and input 3D model. For reference, tested on GeForce GTX 1080 Ti, under default setting, Neural Contours of `bumps_a` takes about 12 minutes.

### Cite:
```
@InProceedings{Liu_2020_CVPR,
author={Liu, Difan and Nabail, Mohamed and Hertzmann, Aaron and Kalogerakis, Evangelos},
title={Neural Contours: Learning to Draw Lines from 3D Shapes},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

### Contact
To ask questions, please [email](mailto:dliu@cs.umass.edu).
