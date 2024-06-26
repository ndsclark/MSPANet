# Multi-Scale Spatial Pyramid Attention Mechanism



***The official PyTorch implementation of "Multi-scale spatial pyramid attention mechanism for image recognition: An effective approach".***


## Method

<div align="center">
	<img src="figures/fig1.png">
</div>
<p align="left">
	Figure 1: The overall architecture of the proposed MSPA module.
</p>


**Description.** It contains three core components: the HPC module, the SPR module, and the Softmax operation. The HPC module is designed to extract multi-scale spatial information. The SPR module is responsible for learning channel attention weights to build cross-dimension interaction. The Softmax operation is used to recalibrate channel-wise attention weights to establish long-range channel dependencies.


**MSPA.** Detail of implementations, including modules and the networks, can be found in ``Cifar-100`` and ``ImageNet`` in this repository. 


## Our environments and toolkits

- OS: Ubuntu 18.04.1
- CUDA: 11.6
- Python: 3.9.12
- Toolkit: PyTorch 1.10
- GPU: RTX A6000 (4x)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [ptflops](https://github.com/sovrasov/flops-counter.pytorch)
- For generating GradCAM++ results, please follow the code on this [repository](https://github.com/jacobgil/pytorch-grad-cam)


## How to incorporate the proposed MSPA module into ResNets

<div align="center">
	<img src="figures/fig2.png">
</div>
<p align="left">
	Figure 2: Comparison between the original bottleneck residual block (left) and the basic building block of the proposed MSPANet (right).
</p>


## Overview of Results

### Comparison of the performance of MSPANet-50 with the change of s and ω on CIFAR-100 classification

<div align="center">
	<img src="figures/fig3.png">
</div>
<p align="left">
	Figure 3: Comparison of the performance of MSPANet-50 with the change of s and ω on CIFAR-100 classification. 
</p>


### Comparison of training and validation curves on ImageNet-1K

<div align="center">
	<img src="figures/fig4.png">
</div>
<p align="left">
	Figure 4: Comparisons of training and validation curves on ImageNet-1K for ResNet, MSPANet-S, and MSPANet-B architectures of 50 and 101 layers, respectively.
</p>


### Classification performance on CIFAR-100

<div align="center">
	<img src="figures/fig5.png">
</div>
<p align="left">
	Table 1: Comparisons of various attention methods on the CIFAR-100 test set in terms of network parameters (Parameters), floating-point operations (FLOPs), and Top-1 accuracy (Top-1 Acc), using ResNet-50, ResNeXt-29, and PreActResNet-164 as baselines, respectively.
</p>


### Classification performance on ImageNet-1K

<div align="center">
	<img src="figures/fig6.png">
</div>
<p align="left">
	Table 2: Comparisons of efficiency (i.e., Parameters and FLOPs) and effectiveness (i.e., Top-1/Top-5Acc) of various attention methods and different multi-scale representation architectures on the ImageNet-1K validation set.
</p>


## Citation
If you find MSPA useful in your research, please consider citing:

    @article{2024mspa,
    	title={Multi-scale spatial pyramid attention mechanism for image recognition: An effective approach},
		author={Yu, Yang and Zhang, Yi and Cheng, Zeyu and Song, Zhe and Tang, Chengkai},
		journal={Engineering Applications of Artificial Intelligence},
		volume={133},
		pages={108261},
		year={2024},
		publisher={Elsevier}
	}


## Contact Information

If you have any suggestion or question, you can leave a message here or contact us directly: yang_y9802@163.com. Thanks for your attention!