# label-noise-theory

## Abstract 
[ICML 2025] Code release for "On the Role of Label Noise in the Feature Learning Process"

> Deep learning with noisy labels presents significant challenges. In this work, we theoretically characterize the role of label noise from a feature learning perspective. Specifically, we consider a *signal-noise* data distribution, where each sample comprises a label-dependent signal and label-independent noise, and rigorously analyze the training dynamics of a two-layer convolutional neural network under this data setup, along with the presence of label noise. Our analysis identifies two key stages. In *Stage I*, the model perfectly fits all the clean samples (i.e., samples without label noise) while ignoring the noisy ones (i.e., samples with noisy labels). During this stage, the model learns the signal from the clean samples, which generalizes well on unseen data. In *Stage II*, as the training loss converges, the gradient in the direction of noise surpasses that of the signal, leading to overfitting on noisy samples. Eventually, the model memorizes the noise present in the noisy samples and degrades its generalization ability. Furthermore, our analysis provides a theoretical basis for two widely used techniques for tackling label noise: early stopping and sample selection. Experiments on both synthetic and real-world setups validate our theory.

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. If you use anaconda3 or miniconda, you can run following instructions to download the required packages in python.
    ```bash
        conda create -n label-noise python=3.9
        conda activate label-noise
        pip install torch torchvision torchaudio numpy matplotlib pandas shap
    ```

---------------------------------------------------------------------------------
Shanghai Jiao Tong University - Email@[zzp1012@sjtu.edu.cn](zzp1012@sjtu.edu.cn)
