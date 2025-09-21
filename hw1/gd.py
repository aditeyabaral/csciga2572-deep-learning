import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    # You should use this as data augmentation and normalization,
    # convnets expect values to be mean 0 and std 1
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


def gradient_descent(input, model, loss, iterations=256):
    lr = 0.05  # learning rate
    wd = 0.0001  # weight decay
    img_blur_step = 100 # every img_blur_step, apply image blur
    do_weight_decay = True # whether to apply weight decay
    do_gradient_blur = False # whether to apply gaussian blur to the gradient
    do_image_blur = False # whether to apply gaussian blur to the image

    # Define scales
    scales = [32, 64, 128, 224]
    num_scales = len(scales)
    progress_bar = tqdm(total=iterations * num_scales)

    for size in scales:
        # Resize to current scale
        input = torch.nn.functional.interpolate(
            input, size=(size, size), mode="bilinear", align_corners=False
        )
        input = input.clone().detach().requires_grad_(True)

        # Run gradient ascent at this scale
        for step in range(iterations):
            # Add jitter and normalize
            img_norm = normalize_and_jitter(input)

            # Forward pass
            out = model(img_norm)
            J = loss(out)

            # Backward pass
            if input.grad is not None:
                input.grad.zero_()
            J.backward()

            # Gradient ascent step
            with torch.no_grad():
                gradient = input.grad
                
                # Apply gradient blur
                if do_gradient_blur:
                    gradient = F.avg_pool2d(gradient, kernel_size=3, stride=1, padding=1)

                # Apply gradient ascent
                input.data += lr * gradient
                
                # Apply weight decay
                if do_weight_decay:
                    blurred_img = F.avg_pool2d(input.data, kernel_size=3, stride=1, padding=1)
                    input.data -= wd * (input.data - blurred_img)
                
                # Clamp to valid image range
                input.data.clamp_(0, 1)
                
                # Apply image blur
                if do_image_blur and step % img_blur_step == 0 and step > 0:
                    input.data = F.avg_pool2d(input.data, kernel_size=3, stride=1, padding=1)

            # Update progress bar
            progress_bar.update(1)

    return input


def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    Try setting the modeul to `model.features[20]` and the loss to `tensor[0, ind].mean()`
    to see what intermediate activations activate on.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()
