import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def rank_images_by_concept(model,
                           layer_name,
                           cav,
                           dataloader_to_rank,
                           dataset_to_rank,
                           extract_fn, 
                           device='cpu',
                           use_cosine_similarity=False):
  
    print(f"\nranking images based on concept in {layer_name}")
    image_scores = []
    cav_tensor = torch.tensor(cav, dtype=torch.float32).to(device)
    current_idx = 0 
    model.eval() 

    with torch.no_grad(): 
        for i, batch_tuple in enumerate(dataloader_to_rank):
            if isinstance(batch_tuple, (list, tuple)):
                batch_data = batch_tuple[0]
            else:
                batch_data = batch_tuple

            print(f"batch {i+1}/{len(dataloader_to_rank)} for ranking")
            batch_acts = extract_fn(model, layer_name, batch_data, device)

            if batch_acts.dim() > 2:
                batch_acts_flat = batch_acts.view(batch_acts.size(0), -1)
            else:
                batch_acts_flat = batch_acts


            if use_cosine_similarity:
                act_norms = torch.linalg.norm(batch_acts_flat, dim=1, keepdim=True)
                act_norms = torch.clamp(act_norms, min=1e-6)
                similarities = torch.matmul(batch_acts_flat, cav_tensor) / act_norms.squeeze()
            else: # Use dot product
                similarities = torch.matmul(batch_acts_flat, cav_tensor)

            scores_list = similarities.cpu().tolist()
            batch_size = batch_data.size(0)
            for j in range(batch_size):
                filename = dataset_to_rank.samples[current_idx][0]
                image_scores.append((scores_list[j], filename))
                current_idx += 1

    image_scores.sort(key=lambda x: x[0], reverse=True)

    print(f"ranking done, {len(image_scores)} images.")
    return image_scores


def display_ranked_images(ranked_list, title, num_to_show=5, target_size=224, class_name="None"):

    # crops to match resnet input aspect ratios
    resize_dim = 256 
    display_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(target_size),
    ])

    print(f"\n{title}")
    print(f"top, {num_to_show} images:")
    fig, axes = plt.subplots(2, num_to_show, figsize=(num_to_show * 3, 6))
    for i in range(num_to_show):
        if i < len(ranked_list):
            score, filename = ranked_list[i]
            print(f"  {i+1}. Score: {score:.4f}, File: {os.path.basename(filename)}")
            img_pil = Image.open(filename).convert('RGB')
            img_transformed_pil = display_transform(img_pil)
            axes[0, i].imshow(img_transformed_pil) 
            axes[0, i].set_title(f"Rank {i+1}\nScore: {score:.2f}")
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off') 

    print(f"\nbottom, {num_to_show} images:")
    for i in range(num_to_show):
        idx = len(ranked_list) - 1 - i
        if idx >= 0 and idx < len(ranked_list): 
            score, filename = ranked_list[idx]
            print(f"  {len(ranked_list)-i}. Score: {score:.4f}, File: {os.path.basename(filename)}")
            img_pil = Image.open(filename).convert('RGB')
            img_transformed_pil = display_transform(img_pil)
            axes[1, i].imshow(img_transformed_pil) 
            axes[1, i].set_title(f"Rank {len(ranked_list)-i}\nScore: {score:.2f}")
            axes[1, i].axis('off')
        else:
             axes[1, i].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f"./scores/cosine/{class_name}.png")
    plt.close()