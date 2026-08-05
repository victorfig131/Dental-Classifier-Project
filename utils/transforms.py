from torchvision import transforms


def get_classifier_train_transforms() -> transforms.Compose:
    """Build data augmentations for classifier training."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def get_classifier_eval_transforms() -> transforms.Compose:
    """Build deterministic transforms for validation/inference."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
