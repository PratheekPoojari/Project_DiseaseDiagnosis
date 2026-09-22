# ============================================================
# ConvNeXt-Base Skin Disease Classification
# Kaggle T4 x2 / Free Tier Optimized
# ============================================================

import os
import json
import time
import random
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Base_Weights
from torchvision.transforms import InterpolationMode

from tqdm.auto import tqdm


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 42

# Dataset split
VAL_FRACTION = 0.20

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

# Kaggle T4 x2 has 4 CPU cores.
# 3 workers feed the training GPU workload while 1 handles validation.
TRAIN_WORKERS = 3
VAL_WORKERS = 1
PREFETCH_FACTOR = 2

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MIN_LR = 1e-6

# Loss
LABEL_SMOOTHING = 0.1

# Image dimensions
INPUT_SIZE = 224
RESIZE_SIZE = 232

# Automatic dataset detection
DATASET_HINT = "skin-diseases-image-dataset"

# Resume from a previous checkpoint in /kaggle/working
RESUME = True

# Kaggle output paths
OUTPUT_DIR = Path("/kaggle/working")

BEST_MODEL_PATH = (
    OUTPUT_DIR / "convnext_base_skin_v2.pth"
)

LAST_CHECKPOINT_PATH = (
    OUTPUT_DIR / "convnext_base_skin_v2_last_checkpoint.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR / "convnext_base_skin_v2_history.json"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

    # Good choice for a fixed image size such as 224x224.
    torch.backends.cudnn.benchmark = True


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 72)
print("ENVIRONMENT")
print("=" * 72)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Device          : {device}")

if torch.cuda.is_available():

    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU count       : {torch.cuda.device_count()}")

    for gpu_id in range(torch.cuda.device_count()):
        print(
            f"GPU {gpu_id}          : "
            f"{torch.cuda.get_device_name(gpu_id)}"
        )

print("=" * 72)


# ============================================================
# 3. FIND DATASET AUTOMATICALLY
# ============================================================

def is_image_file(filename: str) -> bool:
    return (
        Path(filename).suffix.lower()
        in IMAGE_EXTENSIONS
    )


def find_imagefolder_root(
    base_dir: str,
    hint: str | None = None
) -> str:
    """
    Find a valid torchvision ImageFolder root.

    ImageFolder expects:

        dataset_root/
            class_1/
                image.jpg
            class_2/
                image.jpg
            ...

    If multiple valid datasets exist, a path containing `hint`
    is preferred, and then the candidate with the largest
    number of images is selected.
    """

    base = Path(base_dir)

    candidates = set()

    for root, dirs, files in os.walk(base):

        if not any(
            is_image_file(f)
            for f in files
        ):
            continue

        candidate = Path(root).parent
        candidates.add(candidate)

    valid_candidates = []

    for candidate in candidates:

        try:

            test_dataset = datasets.ImageFolder(
                str(candidate)
            )

            if (
                len(test_dataset) > 0
                and len(test_dataset.classes) >= 2
            ):
                valid_candidates.append(
                    (
                        candidate,
                        len(test_dataset),
                        len(test_dataset.classes)
                    )
                )

        except (
            FileNotFoundError,
            RuntimeError,
            OSError
        ):
            continue

    if not valid_candidates:

        raise FileNotFoundError(
            "Could not find a valid ImageFolder dataset "
            f"under {base_dir}"
        )

    # Prefer the requested dataset if its name appears in path.
    if hint:

        hinted_candidates = [
            item
            for item in valid_candidates
            if hint.lower()
            in str(item[0]).lower()
        ]

        if hinted_candidates:
            valid_candidates = hinted_candidates

    # Largest valid dataset wins.
    valid_candidates.sort(
        key=lambda x: (x[1], x[2]),
        reverse=True
    )

    return str(valid_candidates[0][0])


DATA_DIR = find_imagefolder_root(
    "/kaggle/input",
    DATASET_HINT
)

print(f"\nDataset root:")
print(DATA_DIR)


# ============================================================
# 4. TRANSFORMS
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# Training augmentation
train_transform = transforms.Compose([

    transforms.Resize(
        RESIZE_SIZE,
        interpolation=InterpolationMode.BILINEAR
    ),

    transforms.RandomCrop(
        INPUT_SIZE
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15,
        interpolation=InterpolationMode.BILINEAR
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# Validation preprocessing
val_transform = transforms.Compose([

    transforms.Resize(
        RESIZE_SIZE,
        interpolation=InterpolationMode.BILINEAR
    ),

    transforms.CenterCrop(
        INPUT_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ============================================================
# 5. BASE DATASET
# ============================================================

full_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=None
)

class_names = full_dataset.classes
targets = full_dataset.targets

num_classes = len(class_names)

if num_classes < 2:

    raise ValueError(
        f"Expected at least 2 classes, "
        f"but found {num_classes}."
    )


print("\nClasses:")
print("-" * 72)

for class_index, class_name in enumerate(class_names):

    print(
        f"{class_index:2d} : {class_name}"
    )

print("-" * 72)

print(
    f"Total images: {len(full_dataset):,}"
)

print(
    f"ImageFolder mapping:\n"
    f"{full_dataset.class_to_idx}"
)


# ============================================================
# 6. STRATIFIED 80/20 SPLIT
# ============================================================

def stratified_indices(
    targets: list[int],
    val_fraction: float,
    seed: int
):
    """
    Create a deterministic stratified train/validation split.

    Each class contributes approximately the same percentage
    to the validation set.
    """

    rng = random.Random(seed)

    class_to_indices = {}

    for index, target in enumerate(targets):

        if target not in class_to_indices:
            class_to_indices[target] = []

        class_to_indices[target].append(index)

    train_indices = []
    val_indices = []

    for target, indices in class_to_indices.items():

        indices = indices.copy()

        rng.shuffle(indices)

        val_count = max(
            1,
            int(round(
                len(indices) * val_fraction
            ))
        )

        # Keep at least one sample in training.
        val_count = min(
            val_count,
            len(indices) - 1
        )

        val_indices.extend(
            indices[:val_count]
        )

        train_indices.extend(
            indices[val_count:]
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


train_indices, val_indices = (
    stratified_indices(
        targets,
        VAL_FRACTION,
        SEED
    )
)


# ============================================================
# 7. TRANSFORM-AWARE SUBSET
# ============================================================

class TransformSubset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform
    ):

        self.dataset = dataset
        self.indices = indices
        self.transform = transform


    def __len__(self):

        return len(self.indices)


    def __getitem__(self, index):

        original_index = self.indices[index]

        image, label = self.dataset[
            original_index
        ]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


train_dataset = TransformSubset(
    full_dataset,
    train_indices,
    train_transform
)

val_dataset = TransformSubset(
    full_dataset,
    val_indices,
    val_transform
)


print(
    f"\nTraining images   : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation images : "
    f"{len(val_dataset):,}"
)


# ============================================================
# 8. VERIFY STRATIFICATION
# ============================================================

def count_classes(
    indices,
    targets,
    number_of_classes
):

    counts = [
        0
        for _ in range(number_of_classes)
    ]

    for index in indices:

        counts[
            targets[index]
        ] += 1

    return counts


train_counts = count_classes(
    train_indices,
    targets,
    num_classes
)

val_counts = count_classes(
    val_indices,
    targets,
    num_classes
)


print("\nClass distribution:")
print("-" * 72)

print(
    f"{'Class':45s}"
    f"{'Train':>10s}"
    f"{'Validation':>12s}"
)

print("-" * 72)

for i, name in enumerate(class_names):

    print(
        f"{name[:44]:45s}"
        f"{train_counts[i]:10d}"
        f"{val_counts[i]:12d}"
    )

print("-" * 72)


# ============================================================
# 9. DATA LOADERS
# ============================================================

pin_memory = torch.cuda.is_available()


def create_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers
):

    loader_kwargs = {

        "dataset": dataset,

        "batch_size": batch_size,

        "shuffle": shuffle,

        "num_workers": num_workers,

        "pin_memory": pin_memory,

        "drop_last": False
    }

    # These options only exist when workers > 0.
    if num_workers > 0:

        loader_kwargs.update({

            "persistent_workers": True,

            "prefetch_factor": PREFETCH_FACTOR

        })

    return DataLoader(
        **loader_kwargs
    )


train_loader = create_loader(
    train_dataset,
    BATCH_SIZE,
    shuffle=True,
    num_workers=TRAIN_WORKERS
)

val_loader = create_loader(
    val_dataset,
    BATCH_SIZE,
    shuffle=False,
    num_workers=VAL_WORKERS
)


print("\nDataLoader configuration:")
print(f"  Batch size       : {BATCH_SIZE}")
print(f"  Train workers    : {TRAIN_WORKERS}")
print(f"  Val workers      : {VAL_WORKERS}")
print(f"  Prefetch factor  : {PREFETCH_FACTOR}")
print(f"  Pin memory       : {pin_memory}")


# ============================================================
# 10. MODEL
# ============================================================

print("\nLoading pretrained ConvNeXt-Base...")

weights = ConvNeXt_Base_Weights.DEFAULT

model = models.convnext_base(
    weights=weights
)


# Replace ImageNet classifier
num_features = (
    model.classifier[2].in_features
)

model.classifier[2] = nn.Linear(
    num_features,
    num_classes
)


model = model.to(device)


# ============================================================
# 11. MULTI-GPU
# ============================================================

if torch.cuda.device_count() > 1:

    print(
        f"\nUsing "
        f"{torch.cuda.device_count()} GPUs "
        f"with DataParallel."
    )

    model = nn.DataParallel(
        model
    )

else:

    print(
        "\nUsing a single GPU."
    )


# ============================================================
# 12. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer_kwargs = {

    "lr": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY
}


# Fused AdamW can be faster on supported CUDA builds.
try:

    optimizer = optim.AdamW(
        model.parameters(),
        fused=True,
        **optimizer_kwargs
    )

    print(
        "Optimizer: Fused AdamW"
    )

except (
    TypeError,
    RuntimeError
):

    optimizer = optim.AdamW(
        model.parameters(),
        **optimizer_kwargs
    )

    print(
        "Optimizer: Standard AdamW"
    )


# ============================================================
# 14. LEARNING-RATE SCHEDULER
# ============================================================

scheduler = (
    optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LR
    )
)


# ============================================================
# 15. MIXED PRECISION
# ============================================================

AMP_ENABLED = (
    device.type == "cuda"
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=AMP_ENABLED
)

print(
    f"Mixed precision: "
    f"{AMP_ENABLED}"
)


# ============================================================
# 16. MODEL HELPER FUNCTIONS
# ============================================================

def unwrap_model(model):

    if isinstance(
        model,
        nn.DataParallel
    ):

        return model.module

    return model


def get_cpu_state_dict(model):

    base_model = unwrap_model(
        model
    )

    return {

        key: value.detach().cpu().clone()

        for key, value
        in base_model.state_dict().items()

    }


# ============================================================
# 17. ATOMIC CHECKPOINT SAVING
# ============================================================

def atomic_torch_save(
    obj,
    path: Path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fd, temporary_name = tempfile.mkstemp(
        suffix=".tmp",
        dir=str(path.parent)
    )

    os.close(fd)

    try:

        torch.save(
            obj,
            temporary_name
        )

        os.replace(
            temporary_name,
            path
        )

    finally:

        if os.path.exists(
            temporary_name
        ):

            os.remove(
                temporary_name
            )


# ============================================================
# 18. RESUME TRAINING
# ============================================================

start_epoch = 0

best_val_accuracy = -1.0

best_epoch = 0

history = []


if (
    RESUME
    and LAST_CHECKPOINT_PATH.exists()
):

    print(
        "\nCheckpoint detected:"
    )

    print(
        LAST_CHECKPOINT_PATH
    )

    try:

        checkpoint = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False
        )

        checkpoint_classes = (
            checkpoint.get(
                "class_names",
                []
            )
        )

        if (
            checkpoint_classes
            != class_names
        ):

            print(
                "Checkpoint class mapping "
                "does not match this dataset."
            )

            print(
                "Starting a fresh run."
            )

        else:

            # Load model
            unwrap_model(
                model
            ).load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            # Load optimizer
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

            # Load scheduler
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

            # Load AMP scaler
            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

            start_epoch = int(
                checkpoint[
                    "epoch"
                ]
            )

            best_val_accuracy = float(
                checkpoint.get(
                    "best_val_acc",
                    -1.0
                )
            )

            best_epoch = int(
                checkpoint.get(
                    "best_epoch",
                    0
                )
            )

            history = checkpoint.get(
                "history",
                []
            )

            print(
                f"Resuming from epoch "
                f"{start_epoch}."
            )

            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy:.4f}"
            )

    except Exception as error:

        print(
            "\nCould not safely resume "
            f"from checkpoint.\n"
            f"Reason: {error}"
        )

        print(
            "Starting a fresh training run."
        )

        start_epoch = 0
        best_val_accuracy = -1.0
        best_epoch = 0
        history = []


# ============================================================
# 19. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader
):

    model.train()

    running_loss = 0.0
    running_corrects = 0
    samples_seen = 0

    progress_bar = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    for inputs, labels in progress_bar:

        # ----------------------------------------------------
        # CPU -> GPU
        # ----------------------------------------------------

        inputs = inputs.to(
            device,
            non_blocking=pin_memory
        )

        labels = labels.to(
            device,
            non_blocking=pin_memory
        )


        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.amp.autocast(

            device_type=device.type,

            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),

            enabled=AMP_ENABLED
        ):

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                labels
            )


        # ----------------------------------------------------
        # Backward + optimizer step
        # ----------------------------------------------------

        if AMP_ENABLED:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        current_batch_size = (
            inputs.size(0)
        )

        running_loss += (
            loss.detach().item()
            * current_batch_size
        )

        running_corrects += (
            (
                predictions
                == labels
            )
            .sum()
            .item()
        )

        samples_seen += (
            current_batch_size
        )


        # ----------------------------------------------------
        # Live progress
        # ----------------------------------------------------

        progress_bar.set_postfix({

            "loss":
                f"{running_loss / samples_seen:.4f}",

            "acc":
                f"{running_corrects / samples_seen:.4f}"

        })


    epoch_loss = (
        running_loss
        / samples_seen
    )

    epoch_accuracy = (
        running_corrects
        / samples_seen
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 20. VALIDATION FUNCTION
# ============================================================

@torch.inference_mode()
def validate_one_epoch(
    model,
    loader
):

    model.eval()

    running_loss = 0.0
    running_corrects = 0
    samples_seen = 0

    progress_bar = tqdm(
        loader,
        desc="Val",
        leave=False
    )

    for inputs, labels in progress_bar:

        inputs = inputs.to(
            device,
            non_blocking=pin_memory
        )

        labels = labels.to(
            device,
            non_blocking=pin_memory
        )


        with torch.amp.autocast(

            device_type=device.type,

            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),

            enabled=AMP_ENABLED
        ):

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                labels
            )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        current_batch_size = (
            inputs.size(0)
        )

        running_loss += (
            loss.item()
            * current_batch_size
        )

        running_corrects += (
            (
                predictions
                == labels
            )
            .sum()
            .item()
        )

        samples_seen += (
            current_batch_size
        )


        progress_bar.set_postfix({

            "loss":
                f"{running_loss / samples_seen:.4f}",

            "acc":
                f"{running_corrects / samples_seen:.4f}"

        })


    epoch_loss = (
        running_loss
        / samples_seen
    )

    epoch_accuracy = (
        running_corrects
        / samples_seen
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 21. MAIN TRAINING LOOP
# ============================================================

if start_epoch >= NUM_EPOCHS:

    print(
        "\nThe checkpoint already contains "
        f"{start_epoch} completed epochs."
    )

else:

    total_training_start = (
        time.time()
    )

    for epoch in range(
        start_epoch,
        NUM_EPOCHS
    ):

        epoch_number = (
            epoch + 1
        )

        epoch_start = (
            time.time()
        )


        print("\n")
        print("=" * 72)
        print(
            f"EPOCH "
            f"{epoch_number}/{NUM_EPOCHS}"
        )
        print("=" * 72)


        # ----------------------------------------------------
        # Reset peak memory measurement
        # ----------------------------------------------------

        if device.type == "cuda":

            for gpu_id in range(
                torch.cuda.device_count()
            ):

                torch.cuda.reset_peak_memory_stats(
                    gpu_id
                )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate_one_epoch(
                model,
                val_loader
            )
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        current_lr = (
            scheduler.get_last_lr()[0]
        )


        # ----------------------------------------------------
        # Epoch timing
        # ----------------------------------------------------

        epoch_seconds = (
            time.time()
            - epoch_start
        )


        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        epoch_record = {

            "epoch":
                epoch_number,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "learning_rate":
                current_lr,

            "epoch_seconds":
                epoch_seconds
        }

        history.append(
            epoch_record
        )


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"\nTrain Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.8f}"
        )

        print(
            f"Epoch Time      : "
            f"{epoch_seconds / 60:.2f} min"
        )


        # ----------------------------------------------------
        # GPU memory report
        # ----------------------------------------------------

        if device.type == "cuda":

            print(
                "\nPeak GPU memory:"
            )

            for gpu_id in range(
                torch.cuda.device_count()
            ):

                peak_memory_gb = (

                    torch.cuda
                    .max_memory_allocated(
                        gpu_id
                    )
                    / (1024 ** 3)

                )

                print(
                    f"  GPU {gpu_id}: "
                    f"{peak_memory_gb:.2f} GiB"
                )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = (
                epoch_number
            )


            best_model_checkpoint = {

                "model_state_dict":
                    get_cpu_state_dict(
                        model
                    ),

                "class_names":
                    class_names,

                "num_classes":
                    num_classes,

                "best_val_accuracy":
                    best_val_accuracy,

                "best_epoch":
                    best_epoch,

                "input_size":
                    INPUT_SIZE,

                "normalize_mean":
                    IMAGENET_MEAN,

                "normalize_std":
                    IMAGENET_STD
            }


            atomic_torch_save(

                best_model_checkpoint,

                BEST_MODEL_PATH
            )


            print(
                "\n*** NEW BEST MODEL ***"
            )

            print(
                f"Best Val Accuracy: "
                f"{best_val_accuracy:.4f}"
            )


        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        last_checkpoint = {

            "epoch":
                epoch_number,

            "model_state_dict":
                get_cpu_state_dict(
                    model
                ),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "best_val_acc":
                best_val_accuracy,

            "best_epoch":
                best_epoch,

            "class_names":
                class_names,

            "history":
                history,

            "train_indices":
                train_indices,

            "val_indices":
                val_indices
        }


        atomic_torch_save(

            last_checkpoint,

            LAST_CHECKPOINT_PATH

        )


        # ----------------------------------------------------
        # Save history separately
        # ----------------------------------------------------

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as history_file:

            json.dump(
                history,
                history_file,
                indent=2
            )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        epochs_without_improvement = (

            epoch_number
            - best_epoch

        )


        print(
            f"\nBest Val Accuracy: "
            f"{best_val_accuracy:.4f}"
            f"  @ epoch {best_epoch}"
        )


        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"No improvement for "
                f"{EARLY_STOPPING_PATIENCE} "
                f"epochs."
            )

            break


    total_training_seconds = (
        time.time()
        - total_training_start
    )


    print(
        "\nTraining time this session: "
        f"{total_training_seconds / 60:.2f} min"
    )


# ============================================================
# 22. LOAD THE BEST MODEL
# ============================================================

if not BEST_MODEL_PATH.exists():

    raise FileNotFoundError(
        "Best model checkpoint was not created."
    )


best_checkpoint = torch.load(

    BEST_MODEL_PATH,

    map_location="cpu",

    weights_only=False

)


# Make sure the saved classifier matches this dataset.
saved_class_names = (
    best_checkpoint.get(
        "class_names",
        []
    )
)

if (
    saved_class_names
    != class_names
):

    raise RuntimeError(
        "Saved best model class mapping "
        "does not match the current dataset."
    )


unwrap_model(
    model
).load_state_dict(

    best_checkpoint[
        "model_state_dict"
    ]

)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("TRAINING COMPLETE")
print("=" * 72)

print(
    f"Dataset              : "
    f"{DATA_DIR}"
)

print(
    f"Number of classes    : "
    f"{num_classes}"
)

print(
    f"Training images      : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation images    : "
    f"{len(val_dataset):,}"
)

print(
    f"Batch size           : "
    f"{BATCH_SIZE}"
)

print(
    f"Best validation acc  : "
    f"{best_checkpoint['best_val_accuracy']:.4f}"
)

print(
    f"Best epoch           : "
    f"{best_checkpoint['best_epoch']}"
)

print(
    f"\nBest model:"
)

print(
    BEST_MODEL_PATH
)

print(
    f"\nLast resumable checkpoint:"
)

print(
    LAST_CHECKPOINT_PATH
)

print(
    f"\nTraining history:"
)

print(
    HISTORY_PATH
)

print("=" * 72)
