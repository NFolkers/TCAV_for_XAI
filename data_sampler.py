import os
import random
import shutil
import argparse



FLAMINGO = ["n02007558"]

NON_BIRD_CLASSES = [
    "n01968897", # chambered nautilus
    "n01770081", # harvestman
    "n01496331", # electric ray
    "n01687978", # agama
    "n01740131", # night snake
    "n01491361", # tiger shark
    "n01735189", # garter snake
    "n01630670", # common newt
    "n01440764", # tench
    "n01667778", # terrapin
    "n01755581", # diamondback rattlesnake
    "n01924916", # flatworm
    "n01751748", # sea snake
    "n01984695", # spiny lobster
    "n01729977", # green snake
    "n01443537", # goldfish
    "n01770393", # scorpion
    "n01914609", # sea anemone
    "n01667114", # mud turtle
    "n01985128", # crayfish
    "n01773797", # garden spider
    "n01986214", # hermit crab
    "n01484850", # great white shark
    "n01749939", # green mamba
    "n01695060", # Komodo dragon
    "n01729322", # hognose snake
    "n01677366", # common iguana
    "n01734418", # king snake
    "n01773549", # barn spider
    "n01775062", # wolf spider
    "n01728572", # thunder snake
    "n01978287", # Dungeness crab
    "n01930112", # nematode
    "n01739381", # vine snake
    "n01883070", # wombat
    "n01774384", # black widow
    "n01944390", # snail
    "n01494475", # hammerhead shark
    "n01632458", # spotted salamander
    "n01698640", # American alligator
    "n01675722", # banded gecko
    "n01877812", # wallaby
    "n01910747", # jellyfish
    "n01685808", # whiptail lizard
    "n01756291", # sidewinder rattlesnake
    "n01753488", # horned viper
    "n01632777", # axolotl
    "n01644900", # tailed frog
    "n01664065", # loggerhead turtle
    "n01776313", # tick
    "n02077923", # sea lion
    "n01774750", # tarantula
    "n01742172", # boa constrictor
    "n01943899", # conch
    "n01955084", # chiton
    "n01773157", # black and gold garden spider
    "n01665541", # leatherback turtle
    "n01498041", # stingray
    "n01978455", # rock crab
    "n01693334", # green lizard
    "n01950731", # sea slug
]


def sample_non_bird_images(
    data_root,
    split,
    non_bird_classes,
    num_samples,
    output_dir):
    source_split_dir = os.path.join(data_root, split)

    all_non_bird_image_paths = []
    found_class_folders = 0
    for i in range(1, 5):
        source_split_dir_ = source_split_dir + f'{i}'

        # for class_name in os.listdir(source_split_dir_):
        for class_name in os.listdir(source_split_dir_) and class_name in non_bird_classes:
            class_path = os.path.join(source_split_dir_, class_name)
            if os.path.isdir(class_path): 
                found_class_folders += 1
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                        full_path = os.path.join(class_path, filename)
                        all_non_bird_image_paths.append(full_path)


    sampled_paths = random.sample(all_non_bird_image_paths, num_samples)

    output_subdir = os.path.join(output_dir, "random_images/nrandomclass/") 
    os.makedirs(output_subdir, exist_ok=True)
    print(f"output dir: {output_subdir}")

    copied_count = 0
    for src_path in sampled_paths:
        dest_path = os.path.join(output_subdir, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
        copied_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample non-bird images from dataset.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory of the dataset")
    parser.add_argument("--split", type=str, default="train.X", help="Data split to sample from")
    parser.add_argument("--num_samples", type=int, default=250, help="Number of random samples to draw")
    parser.add_argument("--output_dir", type=str, default="./concepts", help="Directory to save sampled images")
    args = parser.parse_args()

    sample_non_bird_images(
        data_root=args.data_root,
        split=args.split,
        non_bird_classes=NON_BIRD_CLASSES,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )