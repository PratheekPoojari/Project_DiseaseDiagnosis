# ============================================================
# ConvNeXt-Base DDP From-Scratch Training Script for Kaggle
# Architecture: ConvNeXt-Base (88M params, ImageNet Pretrained)
# Compute: 2x NVIDIA T4 GPUs via PyTorch DistributedDataParallel
# ============================================================

import argparse
import ctypes
import gc
import json
import os
import random
import tempfile
import time
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Base_Weights
from torchvision.transforms import InterpolationMode


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

OUTPUT_DIR = Path("/kaggle/working")
BEST_MODEL_PATH = OUTPUT_DIR / "convnext_base_skin_v2.pth"
LAST_CHECKPOINT_PATH = OUTPUT_DIR / "convnext_base_skin_v2_last_checkpoint.pth"
HISTORY_PATH = OUTPUT_DIR / "convnext_base_skin_v2_history.json"

# Training configuration
GLOBAL_BATCH_SIZE = 128
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

# With 2 DDP processes: global batch = 128 (64 per GPU)
TRAIN_WORKERS_PER_RANK = 1
VAL_WORKERS_PER_RANK = 0
PREFETCH_FACTOR = 1

INPUT_SIZE = 224
RESIZE_SIZE = 232

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MIN_LR = 1e-6
LABEL_SMOOTHING = 0.1

CUDNN_BENCHMARK = True

# RAM safety limits (emergency tripwires)
MAX_RANK_RSS_GIB = 20.0
MIN_SYSTEM_AVAILABLE_GIB = 6.0

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# 2. DATASET PATH AUTO-DETECTION
# ============================================================

def find_data_dir() -> Path:
    """Finds the dataset directory dynamically regardless of Kaggle mount structure."""
    candidates = [
        Path("/kaggle/input/skin-diseases-image-dataset/IMG_CLASSES"),
        Path("/kaggle/input/datasets/ismailpromus/skin-diseases-image-dataset/IMG_CLASSES"),
        Path("/kaggle/input/skin-diseases-image-dataset"),
    ]
    for c in candidates:
        if c.exists() and any("Eczema" in d.name for d in c.iterdir() if d.is_dir()):
            return c

    # Fallback search inside /kaggle/input
    if Path("/kaggle/input").exists():
        for root, dirs, _ in os.walk("/kaggle/input"):
            for d in dirs:
                if "Eczema" in d:
                    return Path(root)

    return candidates[0]


# ============================================================
# 3. ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ConvNeXt-Base DDP Training From Scratch for Kaggle")
    parser.add_argument(
        "--mode",
        choices=["diagnostic", "train"],
        default="train",
        help="diagnostic = test DDP without saving; train = run full training",
    )
    parser.add_argument("--diag-train-batches", type=int, default=50)
    parser.add_argument("--diag-val-batches", type=int, default=43)
    return parser.parse_args()


# ============================================================
# 4. DDP INITIALIZATION
# ============================================================

def setup_distributed():
    if "RANK" not in os.environ:
        raise RuntimeError("This script must be launched with torchrun.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    gpu_count = torch.cuda.device_count()
    if gpu_count < world_size:
        raise RuntimeError(f"World size={world_size}, but only {gpu_count} GPUs visible.")

    if world_size != 2:
        raise RuntimeError(f"Expected exactly 2 DDP processes for Kaggle T4x2, got {world_size}.")

    if GLOBAL_BATCH_SIZE % world_size != 0:
        raise RuntimeError("GLOBAL_BATCH_SIZE must be divisible by WORLD_SIZE.")

    torch.cuda.set_device(local_rank)
    torch.multiprocessing.set_start_method("spawn", force=True)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# 5. GENERAL HELPERS & MEMORY CLEANUP
# ============================================================

def seed_everything(seed, rank):
    process_seed = seed + rank
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK


def gib_from_bytes(value):
    return value / (1024 ** 3)


def get_rss_gib():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def get_child_rss_gib():
    parent = psutil.Process(os.getpid())
    total = 0
    try:
        for child in parent.children(recursive=True):
            try:
                if child.is_running():
                    total += child.memory_info().rss
            except Exception:
                pass
    except Exception:
        pass
    return gib_from_bytes(total)


def get_system_memory_gib():
    total_kib = None
    available_kib = None
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total_kib = float(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kib = float(line.split()[1])
    return total_kib / 1024 / 1024, available_kib / 1024 / 1024


def malloc_trim():
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        return bool(libc.malloc_trim(0))
    except Exception:
        return False


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
    gc.collect()
    malloc_trim()
    gc.collect()


def get_cuda_memory(device):
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    return gib_from_bytes(allocated), gib_from_bytes(reserved)


def distributed_memory_state(device):
    local_rss = get_rss_gib()
    _, local_available = get_system_memory_gib()
    local_child = get_child_rss_gib()
    local_cuda_alloc, local_cuda_reserved = get_cuda_memory(device)

    values = torch.tensor(
        [local_rss, local_available, local_child, local_cuda_alloc, local_cuda_reserved],
        dtype=torch.float64,
        device=device,
    )
    maximum = values.clone()
    minimum = values.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)

    return {
        "max_rank_rss": maximum[0].item(),
        "min_available": minimum[1].item(),
        "max_child_rss": maximum[2].item(),
        "max_cuda_alloc": maximum[3].item(),
        "max_cuda_reserved": maximum[4].item(),
    }


def print_memory_state(label, device, rank):
    state = distributed_memory_state(device)
    if rank == 0:
        print(f"\n[{label}]")
        print(f"  Max rank RSS       : {state['max_rank_rss']:.2f} GiB")
        print(f"  Min system avail.  : {state['min_available']:.2f} GiB")
        print(f"  Max worker RSS     : {state['max_child_rss']:.2f} GiB")
        print(f"  Max CUDA allocated : {state['max_cuda_alloc']:.2f} GiB")
        print(f"  Max CUDA reserved  : {state['max_cuda_reserved']:.2f} GiB")
    return state


def memory_safety_check(device):
    state = distributed_memory_state(device)
    unsafe = (
        state["max_rank_rss"] >= MAX_RANK_RSS_GIB
        or state["min_available"] <= MIN_SYSTEM_AVAILABLE_GIB
    )
    flag = torch.tensor([1 if unsafe else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item()), state


# ============================================================
# 6. ATOMIC CHECKPOINT SAVE
# ============================================================

def atomic_save(obj, path):
    path = Path(path)
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 7. TRANSFORMS & DATASET (STRATIFIED SPLIT FROM SCRATCH)
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, label = self.dataset[original_index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_datasets(rank):
    data_dir = find_data_dir()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at {data_dir}")

    full_dataset = datasets.ImageFolder(str(data_dir), transform=None)

    # Stratified 80/20 Train/Val Split from Scratch
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_indices, val_indices = next(sss.split(range(len(full_dataset)), full_dataset.targets))
    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()

    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    if rank == 0:
        print()
        print("=" * 72)
        print("DATASET (STRATIFIED 80/20 SPLIT INITIALIZED FROM SCRATCH)")
        print("=" * 72)
        print(f"Path       : {data_dir}")
        print(f"Total      : {len(full_dataset):,}")
        print(f"Classes    : {len(full_dataset.classes)}")
        print(f"Training   : {len(train_dataset):,}")
        print(f"Validation : {len(val_dataset):,}")

    cleanup_memory()
    return full_dataset, train_dataset, val_dataset, train_indices, val_indices


# ============================================================
# 8. DATALOADERS
# ============================================================

def create_dataloaders(train_dataset, val_dataset, rank, world_size):
    local_batch_size = GLOBAL_BATCH_SIZE // world_size

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=SEED,
        drop_last=False,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=SEED,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=TRAIN_WORKERS_PER_RANK,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=VAL_WORKERS_PER_RANK,
        pin_memory=True,
        drop_last=False,
    )

    if rank == 0:
        print()
        print("=" * 72)
        print("DDP DATALOADERS")
        print("=" * 72)
        print(f"Global batch size     : {GLOBAL_BATCH_SIZE}")
        print(f"Local batch size      : {local_batch_size}")
        print(f"Train workers/rank    : {TRAIN_WORKERS_PER_RANK}")
        print(f"Validation workers    : {VAL_WORKERS_PER_RANK}")
        print(f"Prefetch factor       : {PREFETCH_FACTOR}")
        print("Persistent workers    : False")
        print("Pin memory            : True")
        print(f"Train batches/rank    : {len(train_loader)}")
        print(f"Validation batches    : {len(val_loader)}")

    return train_loader, val_loader, train_sampler, val_sampler


# ============================================================
# 9. MODEL & OPTIMIZER (IMAGENET TRANSFER LEARNING)
# ============================================================

def create_model(device):
    # Load official ImageNet-1K pretrained weights for Transfer Learning
    weights = ConvNeXt_Base_Weights.DEFAULT
    model = models.convnext_base(weights=weights)

    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, 10)
    model = model.to(device)
    return model


def create_optimizer_state(model):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=True,
        fused=False,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LR,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return criterion, optimizer, scheduler, scaler


# ============================================================
# 10. CHECKPOINT SERIALIZATION
# ============================================================

def cpu_model_state_dict(ddp_model):
    return {k: v.detach().cpu() for k, v in ddp_model.module.state_dict().items()}


def save_best_model(ddp_model, class_names, num_classes, epoch_number, val_accuracy):
    model_state = cpu_model_state_dict(ddp_model)
    checkpoint = {
        "model_state_dict": model_state,
        "class_names": class_names,
        "num_classes": num_classes,
        "best_val_accuracy": float(val_accuracy),
        "best_epoch": int(epoch_number),
        "input_size": INPUT_SIZE,
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
        "model_type": "convnext_base",
        "parallelism": "ddp",
    }
    atomic_save(checkpoint, BEST_MODEL_PATH)
    del model_state
    del checkpoint
    cleanup_memory()


def save_last_checkpoint(
    ddp_model, optimizer, scheduler, scaler, class_names, num_classes,
    epoch_number, best_val_accuracy, best_epoch, history, train_indices, val_indices,
):
    model_state = cpu_model_state_dict(ddp_model)
    checkpoint = {
        "epoch": int(epoch_number),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc": float(best_val_accuracy),
        "best_epoch": int(best_epoch),
        "class_names": class_names,
        "num_classes": num_classes,
        "input_size": INPUT_SIZE,
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
        "history": history,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "model_type": "convnext_base",
        "parallelism": "ddp",
        "global_batch_size": GLOBAL_BATCH_SIZE,
    }
    atomic_save(checkpoint, LAST_CHECKPOINT_PATH)
    del model_state
    del checkpoint
    cleanup_memory()


# ============================================================
# 11. TRAIN & VALIDATE ONE EPOCH
# ============================================================

def train_one_epoch(
    model, loader, sampler, criterion, optimizer, scaler, device,
    epoch_index, rank, max_batches=None,
):
    model.train()
    sampler.set_epoch(epoch_index)

    running_loss = 0.0
    running_correct = 0
    samples_seen = 0
    total_batches = len(loader)
    batches_to_run = total_batches if max_batches is None else min(max_batches, total_batches)

    start_time = time.time()

    for batch_index, (inputs, labels) in enumerate(loader):
        if batch_index >= batches_to_run:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.size(0)
        running_loss += loss.detach().item() * batch_size
        predictions = outputs.argmax(dim=1)
        running_correct += int((predictions == labels).sum().item())
        samples_seen += batch_size

        del inputs, labels, outputs, predictions, loss

        current_batch = batch_index + 1
        if rank == 0 and (current_batch == 1 or current_batch % 25 == 0 or current_batch == batches_to_run):
            loss_now = running_loss / samples_seen
            accuracy_now = running_correct / samples_seen
            elapsed = (time.time() - start_time) / 60
            print(
                f"  Train {current_batch:3d}/{batches_to_run} | "
                f"loss={loss_now:.4f} | acc={accuracy_now:.4f} | time={elapsed:.1f}m"
            )

        if current_batch == 1 or current_batch % 25 == 0:
            unsafe, state = memory_safety_check(device)
            if unsafe:
                if rank == 0:
                    print("\n!!! HOST RAM SAFETY STOP !!!")
                    print(f"Max rank RSS: {state['max_rank_rss']:.2f} GiB")
                    print(f"Min available: {state['min_available']:.2f} GiB")
                return None, None, True

    metrics = torch.tensor([running_loss, float(running_correct), float(samples_seen)], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0].item() / metrics[2].item(), metrics[1].item() / metrics[2].item(), False


@torch.inference_mode()
def validate_one_epoch(model, loader, criterion, device, rank, max_batches=None):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    samples_seen = 0
    total_batches = len(loader)
    batches_to_run = total_batches if max_batches is None else min(max_batches, total_batches)

    for batch_index, (inputs, labels) in enumerate(loader):
        if batch_index >= batches_to_run:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        running_correct += int((predictions == labels).sum().item())
        samples_seen += batch_size

        del inputs, labels, outputs, predictions, loss

        current_batch = batch_index + 1
        if rank == 0 and (current_batch == 1 or current_batch % 20 == 0 or current_batch == batches_to_run):
            loss_now = running_loss / samples_seen
            accuracy_now = running_correct / samples_seen
            print(f"  Val   {current_batch:3d}/{batches_to_run} | loss={loss_now:.4f} | acc={accuracy_now:.4f}")

    metrics = torch.tensor([running_loss, float(running_correct), float(samples_seen)], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0].item() / metrics[2].item(), metrics[1].item() / metrics[2].item()


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rank, world_size, local_rank, device = setup_distributed()

    try:
        seed_everything(SEED, rank)

        if rank == 0:
            print()
            print("=" * 72)
            print("CONVNEXT-BASE DDP (FROM-SCRATCH TRANSFER LEARNING)")
            print("=" * 72)
            print(f"Mode               : {args.mode}")
            print(f"PyTorch            : {torch.__version__}")
            print(f"CUDA               : {torch.version.cuda}")
            print(f"World size         : {world_size}")
            print(f"Global batch       : {GLOBAL_BATCH_SIZE}")
            print(f"Local batch        : {GLOBAL_BATCH_SIZE // world_size}")
            print(f"Rank 0 GPU         : {torch.cuda.get_device_name(0)}")
            print(f"Rank 1 GPU         : {torch.cuda.get_device_name(1)}")
            print("\nWeights Pretraining: ImageNet-1K (ConvNeXt_Base_Weights.DEFAULT)")
            print("Parallelism        : DistributedDataParallel (DDP)")
            print("AMP                : Enabled (FP16)")

        dist.barrier()

        # 1. Dataset (Stratified 80/20 from scratch)
        full_dataset, train_dataset, val_dataset, train_indices, val_indices = load_datasets(rank)
        class_names = full_dataset.classes
        num_classes = len(class_names)

        train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
            train_dataset, val_dataset, rank, world_size
        )

        # 2. Model & Optimizer
        base_model = create_model(device)
        criterion, optimizer, scheduler, scaler = create_optimizer_state(base_model)

        # Training starts freshly at Epoch 0
        start_epoch = 0
        best_val_accuracy = 0.0
        best_epoch = 0
        history = []

        # 3. DDP Wrap
        ddp_model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        del base_model
        cleanup_memory()

        print_memory_state("AFTER DDP INITIALIZATION", device, rank)

        # 4. Mode: Diagnostic
        if args.mode == "diagnostic":
            if rank == 0:
                print()
                print("=" * 72)
                print("DDP DIAGNOSTIC MODE (50 TRAIN / 43 VAL BATCHES)")
                print("=" * 72)

            dist.barrier()
            diag_train_loss, diag_train_acc, aborted = train_one_epoch(
                ddp_model, train_loader, train_sampler, criterion, optimizer, scaler,
                device, epoch_index=0, rank=rank, max_batches=args.diag_train_batches,
            )
            if aborted:
                raise RuntimeError("Diagnostic hit RAM safety limit.")

            dist.barrier()
            diag_val_loss, diag_val_acc = validate_one_epoch(
                ddp_model, val_loader, criterion, device, rank, max_batches=args.diag_val_batches,
            )
            if rank == 0:
                print(f"\nDiagnostic train loss: {diag_train_loss:.4f} | acc: {diag_train_acc:.4f}")
                print(f"Diagnostic val loss  : {diag_val_loss:.4f} | acc: {diag_val_acc:.4f}")
                print("\nDiagnostic complete! If RAM is healthy, run with: --mode train")
            return

        # 5. Mode: Full Training
        if rank == 0:
            print()
            print("=" * 72)
            print(f"STARTING FULL TRAINING (1 TO {NUM_EPOCHS} EPOCHS)")
            print("=" * 72)

        dist.barrier()
        total_start = time.time()

        for epoch_index in range(start_epoch, NUM_EPOCHS):
            epoch_number = epoch_index + 1
            print_memory_state(f"EPOCH {epoch_number} START", device, rank)
            torch.cuda.reset_peak_memory_stats(local_rank)

            # --- Train ---
            train_start = time.time()
            train_loss, train_accuracy, aborted = train_one_epoch(
                ddp_model, train_loader, train_sampler, criterion, optimizer, scaler,
                device, epoch_index, rank,
            )

            if aborted:
                if rank == 0:
                    print("\n!!! Training stopped because of RAM safety limit !!!")
                return

            train_minutes = (time.time() - train_start) / 60

            # --- Validate ---
            val_start = time.time()
            val_loss, val_accuracy = validate_one_epoch(
                ddp_model, val_loader, criterion, device, rank,
            )
            val_minutes = (time.time() - val_start) / 60

            # --- Scheduler ---
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # --- History ---
            history.append({
                "epoch": epoch_number,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "learning_rate": float(current_lr),
                "train_minutes": float(train_minutes),
                "val_minutes": float(val_minutes),
                "parallelism": "ddp",
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "local_batch_size": GLOBAL_BATCH_SIZE // world_size,
            })

            local_peak_gpu = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)
            peak_tensor = torch.tensor([local_peak_gpu], dtype=torch.float64, device=device)
            dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
            max_peak_gpu = peak_tensor.item()

            if rank == 0:
                print()
                print("=" * 72)
                print(f"EPOCH {epoch_number} RESULTS")
                print("=" * 72)
                print(f"Train Loss      : {train_loss:.4f} | Accuracy: {train_accuracy:.4f}")
                print(f"Val Loss        : {val_loss:.4f} | Accuracy: {val_accuracy:.4f}")
                print(f"Learning Rate   : {current_lr:.8f}")
                print(f"Train Time      : {train_minutes:.2f} min | Val Time: {val_minutes:.2f} min")
                print(f"Peak GPU memory : {max_peak_gpu:.2f} GiB")

            print_memory_state("AFTER EPOCH", device, rank)

            # --- Save Best Model ---
            improved = val_accuracy > best_val_accuracy
            if improved:
                best_val_accuracy = val_accuracy
                best_epoch = epoch_number
                if rank == 0:
                    print("\nSaving BEST model...")
                    save_best_model(ddp_model, class_names, num_classes, epoch_number, val_accuracy)
                    print(f"*** NEW BEST MODEL SAVED (Val Acc: {best_val_accuracy:.4f}) ***")
                dist.barrier()

            # --- Save Resumable Checkpoint ---
            if rank == 0:
                print("\nSaving resumable checkpoint...")
                save_last_checkpoint(
                    ddp_model, optimizer, scheduler, scaler, class_names, num_classes,
                    epoch_number, best_val_accuracy, best_epoch, history, train_indices, val_indices,
                )
            dist.barrier()

            # --- Save History JSON ---
            if rank == 0:
                with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            dist.barrier()

            cleanup_memory()
            print_memory_state("AFTER CLEANUP", device, rank)

            # --- Early Stopping ---
            epochs_without_improvement = epoch_number - best_epoch
            if rank == 0:
                print(f"\nBest Val Accuracy: {best_val_accuracy:.4f} @ epoch {best_epoch}")
                print(f"No improvement for: {epochs_without_improvement} epoch(s)")

            stop_tensor = torch.tensor(
                [1 if epochs_without_improvement >= EARLY_STOPPING_PATIENCE else 0],
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)

            if bool(stop_tensor.item()):
                if rank == 0:
                    print("\nEarly stopping triggered.")
                break

        total_minutes = (time.time() - total_start) / 60
        cleanup_memory()

        if rank == 0:
            print()
            print("=" * 72)
            print("DDP TRAINING FINISHED")
            print("=" * 72)
            print(f"Best Val Accuracy : {best_val_accuracy:.4f}")
            print(f"Best Epoch        : {best_epoch}")
            print(f"Total runtime     : {total_minutes:.2f} min")
            print(f"Best model        : {BEST_MODEL_PATH}")
            print(f"Resume checkpoint : {LAST_CHECKPOINT_PATH}")
            print(f"History           : {HISTORY_PATH}")

        dist.barrier()

    except Exception as error:
        if rank == 0:
            print("\n" + "=" * 72)
            print("DDP ERROR")
            print("=" * 72)
            print(repr(error))
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
