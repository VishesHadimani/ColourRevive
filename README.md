# ColourRevive

A CNN-based deep learning model for automatic colorization of black-and-white photographs. Given a grayscale image, the model predicts plausible, contextually appropriate color and reconstructs a full-color output.

![Colorized result](Report%20img/results1.png)
*Left: grayscale input. Right: model output.*

## Overview

ColourRevive works in the **LAB color space** rather than predicting RGB directly. This is a deliberate design choice: LAB separates **luminance (L)** — brightness, which is exactly what a grayscale image already is — from **chrominance (A and B)** — the color information that's missing. The model only has to learn to predict the two missing chrominance channels and reuses the original L channel untouched, which keeps the input image's brightness and detail intact instead of trying to regenerate the whole image from scratch.

**Pipeline:**
1. Load grayscale and color image pairs
2. Convert color images to LAB; normalize L to [0, 1] and A/B channels to [0, 1]
3. Train a CNN to map L → (A, B)
4. At inference, predict (A, B) for a new grayscale image, recombine with its L channel, and convert back to RGB

## Architecture

Encoder–decoder CNN built with TensorFlow/Keras:

- **Encoder**: stacked `Conv2D` layers (8 → 16 → 32 filters) with stride-2 downsampling for feature extraction
- **Decoder**: `UpSampling2D` + `Conv2D` layers (32 → 16 → 2 filters) to reconstruct full resolution
- **Output**: 2-channel sigmoid output representing the predicted A/B chrominance channels

```
Input (grayscale, 1 channel)
  → Conv2D(8)  stride 2
  → Conv2D(16) → Conv2D(16) stride 2
  → Conv2D(32) → Conv2D(32) stride 2
  → UpSampling2D → Conv2D(32)
  → UpSampling2D → Conv2D(16)
  → UpSampling2D → Conv2D(2, sigmoid)   →  predicted A, B channels
```

- **Loss function**: Mean Squared Error (MSE) between predicted and actual A/B channels
- **Optimizer**: RMSprop
- **Training**: 400 epochs, batch size 1, on an NVIDIA RTX 4060 GPU

## Results

Training loss decreased steadily over 400 epochs, reaching a final value of **0.0011**, with a smooth, stable convergence curve (no divergence or instability observed).

![Training loss curve](Image%20colorization/loss_graph.png)

Qualitatively, the model generalized well to unseen test images, particularly landscapes and portraits — vegetation rendered green, skies blue, and skin tones realistic, suggesting the model learned genuine contextual associations between textures/objects and plausible colors rather than memorizing specific images.

**Limitations**: the model struggles with complex textures and visually ambiguous color contexts (objects that could plausibly be several different colors). These cases produced less accurate or muted colorization. This is a known, expected limitation of MSE-based colorization models — MSE optimizes for the statistically "safest" average color, which tends toward desaturated outputs in ambiguous cases. Addressing this typically requires a perceptual or adversarial loss instead of plain MSE.

## Tech Stack

Python, TensorFlow/Keras, NumPy, OpenCV, scikit-image, Matplotlib

## Future Work

- Replace plain MSE loss with a perceptual loss or adversarial (GAN-based) setup to improve color vibrancy and realism on ambiguous cases
- Train on a larger, more diverse dataset to improve generalization on complex textures
- Add user-guided color hints for cases where the "correct" color is genuinely ambiguous from grayscale alone

## Project Context

Built as a mini-project for the Neural Networks and Deep Learning course (BE, AI & Data Science) at Sahyadri College of Engineering & Management. Full academic report included in this repo: [`ColourRevive_Report.pdf`](ColourRevive_Report.pdf).

## Authors

- [Vishesh Hadimani](https://github.com/VishesHadimani)
- P Ashish Rao
