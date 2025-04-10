import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
import os
import csv
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC 

from tcav_ranking import rank_images_by_concept, display_ranked_images
from class_lookup import subset_index_to_details, get_imagenet_index_and_label



def log_results(filename, 
                concept_name,
                target_class_name,
                target_class_index,
                layer_name,
                tcav_score,
                positive_count,
                total_count,
                max_rank_score,  
                min_rank_score,  
                avg_rank_score): 

    file_exists = os.path.isfile(filename)
    is_empty = (not file_exists) or (os.path.getsize(filename) == 0)

    columns = [
        "Concept", "Target Class", "Target Index", "Layer",
        "TCAV Score", "Positive Count", "Total Count",
        "Max Rank Score", "Min Rank Score", "Avg Rank Score" 
    ]

    data_row = [
        concept_name, target_class_name, target_class_index, layer_name,
        f"{tcav_score:.4f}", positive_count, total_count,
        f"{max_rank_score:.4f}" if max_rank_score is not None else "N/A", 
        f"{min_rank_score:.4f}" if min_rank_score is not None else "N/A",
        f"{avg_rank_score:.4f}" if avg_rank_score is not None else "N/A"
    ]

    with open(filename, 'a+', newline='') as csvfile:
        csvwrite = csv.writer(csvfile)
        if is_empty:
            csvwrite.writerow(columns)
        csvwrite.writerow(data_row)



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_TARGET_IMG_SIZE = 224 

def load_pretrained_resnet18(device='cpu'):
    print("loading ResNet")
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model = model.to(device)
    model.eval() 
    print("resnet loaded")
    return model


# # preprocessing function needed for resnet input
# def get_imagenet_transforms(target_size=DEFAULT_TARGET_IMG_SIZE,
#                             mean=IMAGENET_MEAN,
#                             std=IMAGENET_STD):
#     return transforms.Compose([
#         transforms.Resize(target_size),       
#         transforms.CenterCrop(target_size), 
#         transforms.ToTensor(),             
#         transforms.Normalize(mean=mean, std=std) 
#     ])


class SingleFolderDataset(Dataset):
    def __init__(self, file_list, transform, base_path): 
        self.file_list = file_list
        self.transform = transform
        self.samples = [(f, 0) for f in file_list]
        folder_name = os.path.basename(base_path) 
        self.classes = [folder_name]
        self.class_to_idx = {folder_name: 0}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = 0
        return image, label


def load_imagenet_data(data_path,
                      batch_size=32,
                      target_size=DEFAULT_TARGET_IMG_SIZE,
                      num_workers=0,
                      shuffle=False):
    
    if not os.path.isdir(data_path):
        return None, None

    print(f"load data from: {data_path}")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    is_single_folder_mode = False
    items = [item for item in os.listdir(data_path)] 
    is_single_folder_mode = False
    if items:
        potential_images = [item for item in items if '.' in item and not os.path.isdir(os.path.join(data_path, item))]
        potential_dirs = [item for item in items if os.path.isdir(os.path.join(data_path, item))]
        if not potential_dirs or (potential_images and len(potential_dirs) <= 1):
                exts_img = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                image_like_files = [f for f in potential_images if f.lower().endswith(exts_img)]
                if image_like_files:
                    is_single_folder_mode = True


    dataset = None
    if is_single_folder_mode:
        image_files = []
        exts = ('.png', '.jpg', '.jpeg', '.JPEG', '.bmp', '.tif', '.tiff')
        for filename in os.listdir(data_path):
            if filename.lower().endswith(exts):
                image_files.append(os.path.join(data_path, filename))

        dataset = SingleFolderDataset(image_files, preprocess, data_path)
        print(f"loaded {len(dataset)} images, {filename}")

    else: 
        dataset = datasets.ImageFolder(root=data_path, transform=preprocess)       
        print(f"classes found: {dataset.classes}")


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader, dataset



activations_storage = {}
def make_hook(layer_name):
    def hook(model, input, output):
        activations_storage[layer_name] = output.detach()
    return hook

# function used to get activations from the model, is also passed to ranking function
def extract_activations(model, layer_name, input_batch, device='cpu'):
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    hook_handle = target_layer.register_forward_hook(make_hook(layer_name)) # this stores the activations 
    input_batch = input_batch.to(device)
    with torch.no_grad():
        model(input_batch)

    hook_handle.remove()
    activation = activations_storage.pop(layer_name, None)
    return activation



def calculate_cav(model, layer_name, concept_loader, random_loader, device='cpu'):
    print(f"\nCalculating CAV for layer '{layer_name}'...")
    concept_activations = []
    random_activations = []

    # concept activations
    for batch_data, _ in concept_loader:
        batch_acts = extract_activations(model, layer_name, batch_data, device)
        if batch_acts is not None:
            if batch_acts.dim() > 2:
                batch_acts = batch_acts.view(batch_acts.size(0), -1)
            concept_activations.append(batch_acts.cpu().numpy())

    # random images activations
    for batch_data, _ in random_loader:
        batch_acts = extract_activations(model, layer_name, batch_data, device)
        if batch_acts is not None:
            if batch_acts.dim() > 2:
                batch_acts = batch_acts.view(batch_acts.size(0), -1)
            random_activations.append(batch_acts.cpu().numpy())

    all_concept_acts = np.concatenate(concept_activations, axis=0)
    all_random_acts = np.concatenate(random_activations, axis=0)
    all_acts = np.concatenate([all_concept_acts, all_random_acts], axis=0)
    concept_labels = np.ones(all_concept_acts.shape[0])
    random_labels = np.zeros(all_random_acts.shape[0])
    all_labels = np.concatenate([concept_labels, random_labels], axis=0)

    classifier = LogisticRegression(solver='liblinear', C=1.0) 
    classifier.fit(all_acts, all_labels)

    accuracy = classifier.score(all_acts, all_labels)
    print(f"linear classifier acc: {accuracy}")

    cav_vector = classifier.coef_[0]

    norm = np.linalg.norm(cav_vector)

    normalized_cav = cav_vector / norm
    print(f"CAV shape: {normalized_cav.shape}")

    return normalized_cav

activations_grad_storage = {}
def make_hook_for_grad(layer_name):
    def hook(model, input, output):
        activations_grad_storage[layer_name] = output
    return hook

def calculate_tcav_score(model, layer_name, target_class_loader, target_class_index, cav, device='cpu'):
    print(f"\n TCAV score for idx class {target_class_index}")
    positive_sensitivity_count = 0
    total_examples = 0
    cav_tensor = torch.tensor(cav, dtype=torch.float32).to(device)

    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    hook_handle = target_layer.register_forward_hook(make_hook_for_grad(layer_name))

    for i, (batch_data, _) in enumerate(target_class_loader):
        print(f"Processing batch {i+1}/{len(target_class_loader)} for sensitivity...")
        batch_data = batch_data.to(device)
        total_examples += batch_data.size(0)

        activations_grad_storage.pop(layer_name, None)
        logits = model(batch_data) 
        intermediate_activation = activations_grad_storage.pop(layer_name, None)
        intermediate_activation.requires_grad_(True)

        target_logits = logits[:, target_class_index] 

        grads = torch.autograd.grad(
            outputs=target_logits,
            inputs=intermediate_activation,
            grad_outputs=torch.ones_like(target_logits),
            only_inputs=True, # only compute grad for the inputs
            retain_graph=False
        )[0] 


        if grads.dim() > 2:
            grads = grads.view(grads.size(0), -1)

        if cav_tensor.shape[0] != grads.shape[1]:
                print(f"dimension btween cav tensor and grads is not the same {cav_tensor.shape[0]}, {grads.shape[1]}")
                continue

        sensitivities = torch.matmul(grads, cav_tensor) 
        positive_sensitivity_count += torch.sum(sensitivities > 0).item()
        del grads, intermediate_activation, logits, target_logits, sensitivities

    hook_handle.remove()

    tcav_score = positive_sensitivity_count / total_examples
    print(f"TCAV score calculated: {tcav_score} ({positive_sensitivity_count}/{total_examples} examples sensitive)")

    return tcav_score, positive_sensitivity_count, total_examples


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    steps = ['setup', 'calculate_cav', 'score_class', 'ranking', 'log']


    CONCEPT_IMG_DIR = "./concepts/feathers"
    RANDOM_IMG_DIR = "./concepts/random_images" 
    IMAGENET_PARENT_DIR = "./data" 
    BATCH_SIZE = 64
    CONCEPT_NAME = "feathers"
    LAYER_NAME_TO_ANALYZE = 'layer4'
    LOG_FILENAME = "tcav_ranking_log_cosine.csv" 

    model = None
    concept_loader, random_loader = None, None
    cav = None


    if 'setup' in steps:
        model = load_pretrained_resnet18(device=DEVICE)
        concept_loader, _ = load_imagenet_data(CONCEPT_IMG_DIR, batch_size=BATCH_SIZE)
        random_loader, _ = load_imagenet_data(RANDOM_IMG_DIR, batch_size=BATCH_SIZE)

    if 'calculate_cav' in steps:
        print("start tcav calculations")
        if concept_loader and random_loader and model:
            cav = calculate_cav(model, LAYER_NAME_TO_ANALYZE, concept_loader, random_loader, device=DEVICE)

    for i in range(1, 5): # this is needed because the dataset is somehow split up into 4 parts
        source_split_dir = os.path.join(IMAGENET_PARENT_DIR, f"train.X{i}")

        for class_folder_name in sorted(os.listdir(source_split_dir)): 
            class_folder_path = os.path.join(source_split_dir, class_folder_name)

            if not os.path.isdir(class_folder_path): continue 

            print(f"{class_folder_name}")
            target_class_index = None
            target_class_name = "Unknown"
            tcav_score = 0.0
            positive_count = 0
            total_count = 0
            max_rank_score = None
            min_rank_score = None
            avg_rank_score = None
            ranked_target_images = None 

            target_class_index, target_class_name = get_imagenet_index_and_label(
                wnid_to_find=class_folder_name,
                full_index_map=subset_index_to_details
            )

            target_loader, target_dataset = load_imagenet_data(
                class_folder_path, batch_size=BATCH_SIZE, shuffle=False
            )

            if 'score_class' in steps:
                if cav is not None and model:
                    tcav_score, positive_count, total_count = calculate_tcav_score(
                        model, LAYER_NAME_TO_ANALYZE, target_loader,
                        target_class_index, cav, device=DEVICE
                    )

            if 'ranking' in steps:
                print("start ranking images")
                if cav is not None and target_dataset and model:
                    ranked_target_images = rank_images_by_concept(
                        model, LAYER_NAME_TO_ANALYZE, cav,
                        target_loader, target_dataset,
                        extract_fn=extract_activations, device=DEVICE, use_cosine_similarity=True
                    )
                    scores_only = [score for score, filename in ranked_target_images]
                    if scores_only: 
                        max_rank_score = np.max(scores_only)
                        min_rank_score = np.min(scores_only)
                        avg_rank_score = np.mean(scores_only)
                        print(f"anking: max={max_rank_score:.4f}, min={min_rank_score:.4f}, avg={avg_rank_score:.4f}")

                    #create the ranked images summaries for the class
                    display_ranked_images(
                        ranked_target_images,
                        title=f"'{target_class_name}' Ranked by '{CONCEPT_NAME}'",
                        num_to_show=5,
                        class_name=target_class_name
                    )

            if 'log' in steps:
                    print(f"--- Logging Results ---")
                    log_results(
                        filename=LOG_FILENAME,
                        concept_name=CONCEPT_NAME,
                        target_class_name=target_class_name,
                        target_class_index=target_class_index,
                        layer_name=LAYER_NAME_TO_ANALYZE,
                        tcav_score=tcav_score,
                        positive_count=positive_count,
                        total_count=total_count,
                        max_rank_score=max_rank_score, 
                        min_rank_score=min_rank_score,
                        avg_rank_score=avg_rank_score
                     )






