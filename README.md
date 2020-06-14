# Neural Contours: Learning to Draw Lines from 3D Shapes

This repository contains the PyTorch implementation for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "Neural Contours: Learning to Draw Lines from 3D Shapes" by [Difan Liu](https://people.cs.umass.edu/~dliu/), Mohamed Nabail, [Aaron Hertzmann](https://www.dgp.toronto.edu/~hertzman/), [Evangelos Kalogerakis](https://people.cs.umass.edu/~kalo/).

### Dependency
- The project is developed on Ubuntu 16.04 with cuda9.0 + cudnn7.0. The code has been tested with PyTorch 1.1.0 (GPU version) and Python 3.6.8. 
- Python packages:
    - OpenCV (tested with 4.2.0)
    - PyYAML (tested with 5.3.1)
    - scikit-image (tested with 0.14.2)

### Dataset
- download example testing data:
    ```
    cd data/example
    wget https://people.cs.umass.edu/~dliu/projects/NeuralContours/example.zip
    unzip example.zip
    ```
    
### Differentiable Geometry Branch
- demo:
    ```python
    python -m scripts.geometry_branch_demo -sc 10.0 -r 10.0 -v 10.0 -ar 0.1 -model_name bumps_a -save_name data/output/bumps_a.png
    ```

### Testing with NRM
- Coming soon
    

### Contact
To ask questions, please [email](mailto:dliu@cs.umass.edu).
