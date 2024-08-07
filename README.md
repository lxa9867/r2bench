> [**R2-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations**](https://arxiv.org/abs/2403.04924)
>
> Xiang Li, Kai Qiu, Jinglu Wang, Xiaohao Xu, Rita Singh, Kashu Yamazaki, Hao Chen, Xiaonan Huang, Bhiksha Raj

<p align="center"><img src="assets/Illustration.png" width="700"/></p>

# Updates
- **(2024-07-01)** R2-Bench got accepted to ECCV 2024! We are cleaning the remaining code and will release it by the end of July.
- **(2024-03-16)** We release the preview version of the noise synthesis toolbox!

# Instantiated Datasets
| Task | Original Dataset |  Link |
|-----------------|-----------------|-----------------|
| VOS | Youtube-VOS | [Link](https://drive.google.com/file/d/133yl9mIrUrD3g-Geh9_v-t01EB7VRW56/view?usp=drive_link) |
| VOS | DAVIS | [Link](https://drive.google.com/file/d/1JQrPHKr9zzP7W4V9sQgadVEjTSVWlJhS/view?usp=drive_link) |
| RIS | Ref-COCO | [Link](https://drive.google.com/file/d/1CwU1wXmo4wieKIx1q4po_cnqyiBMRg8F/view?usp=drive_link) |



For AVS and R-VOS datasets, please use the 'create_noisy_data.py' (with default random seed).

# Installation
```
conda create -n r2bench python=3.9
conda activate r2bench
pip install -r perturbation_toolbox/requirements.txt
```

# Example Usage
```
augmenter = ModalityAugmentation()
    # Example of using the class for audio
    samples, samplerate = sf.read('data/sample_0.wav')
    samples = samples.transpose()
    for noise_type in augmenter.audio_noise_functions.keys():
        if noise_type == "background_noise":
            samples = stereo_to_mono(samples)
            augmented_samples = augmenter.apply(samples, "audio", noise_type, severity=1, sample_rate=samplerate, background_path='data/sample_0.wav')
            augmented_samples = np.stack([augmented_samples, augmented_samples])
        else:
            augmented_samples = augmenter.apply(samples, "audio", noise_type, sample_rate=samplerate, severity=1)
        sf.write('augmented_samples.wav', augmented_samples.transpose(), samplerate)

    # Example of using the class for image
    image = Image.open('sample.png')
    if image.mode == 'RGBA':
        # Convert to RGB
        image = image.convert('RGB')

    for noise_type in augmenter.image_noise_functions.keys():
        augmented_image = augmenter.apply(image, "image", noise_type, severity=1)
        augmented_image_pil = Image.fromarray(np.uint8(augmented_image))
        augmented_image_pil.save('augmented_image.png')

    # Example of using the class for text
    for noise_type in augmenter.text_noise_functions.keys():
        answer = augmenter.apply("a dog that is running", "text", noise_type, severity=2)
        print(answer)
```
The example method for apply noisy for data refcoco, davis, Youtube VOS can be seen in create_noist_data.py

# Visualization
<p align="center"><img src="assets/vis.jpg" width="900"/></p>

# Related works for robust referring perception:

> [Towards Robust Audiovisual Segmentation in Complex Environments with Quantization-based Semantic Decomposition](https://arxiv.org/abs/2310.00132), CVPR 2024

> [Robust Referring Video Object Segmentation with Cyclic Structural Consensus](https://arxiv.org/abs/2403.04924), ICCV 2023

## Citation
```
@article{li2024text,
  title={$$\backslash$text $\{$R$\}$\^{} 2$-Bench: Benchmarking the Robustness of Referring Perception Models under Perturbations},
  author={Li, Xiang and Qiu, Kai and Wang, Jinglu and Xu, Xiaohao and Singh, Rita and Yamazak, Kashu and Chen, Hao and Huang, Xiaonan and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2403.04924},
  year={2024}
}
```
