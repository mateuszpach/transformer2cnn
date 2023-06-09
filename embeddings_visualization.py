import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm, trange

from DistilledResnetModel import DistilledResNetModel
from CUB200DataModule import CUB200DataModule
from dataset.vits_finetune import ViTLightningModule

cifar100_labels = {
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver", 5: "bed", 6: "bee", 7: "beetle", 8: "bicycle",
    9: "bottle", 10: "bowl",
    11: "boy", 12: "bridge", 13: "bus", 14: "butterfly", 15: "camel", 16: "can", 17: "castle", 18: "caterpillar",
    19: "cattle", 20: "chair",
    21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach", 25: "couch", 26: "cra", 27: "crocodile", 28: "cup",
    29: "dinosaur", 30: "dolphin",
    31: "elephant", 32: "flatfish", 33: "forest", 34: "fox", 35: "girl", 36: "hamster", 37: "house", 38: "kangaroo",
    39: "keyboard", 40: "lamp",
    41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard", 45: "lobster", 46: "man", 47: "maple_tree",
    48: "motorcycle", 49: "mountain", 50: "mouse",
    51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid", 55: "otter", 56: "palm_tree", 57: "pear",
    58: "pickup_truck", 59: "pine_tree", 60: "plain",
    61: "plate", 62: "poppy", 63: "porcupine", 64: "possum", 65: "rabbit", 66: "raccoon", 67: "ray", 68: "road",
    69: "rocket", 70: "rose", 71: "sea",
    72: "seal", 73: "shark", 74: "shrew", 75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
    80: "squirrel", 81: "streetcar",
    82: "sunflower", 83: "sweet_pepper", 84: "table", 85: "tank", 86: "telephone", 87: "television", 88: "tiger",
    89: "tractor", 90: "train",
    91: "trout", 92: "tulip", 93: "turtle", 94: "wardrobe", 95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman",
    99: "worm",
}


def cub_batch_extractor(batch):
    imgs, batch_labels = batch['img'], batch['label']
    return imgs, batch_labels


def cifar_batch_extractor(batch):
    imgs, batch_labels = batch
    return imgs, batch_labels


def create_tsne_visualization(model, test_loader, num_classes, class_names, out_file, batch_extractor, device):
    # Iterate over the test set
    all_student_cls = []
    all_batch_labels = []
    tl = tqdm(test_loader, desc='Batch')
    with torch.no_grad():
        for i, batch in enumerate(tl):
            imgs, batch_labels = batch_extractor(batch)
            imgs = imgs.to(device)
            outs = model(imgs)
            student_cls = outs['cls'].detach()

            all_student_cls.append(student_cls.cpu().numpy())
            all_batch_labels.append(batch_labels.cpu().numpy())

            if i > 130:
                break

    all_student_cls = np.concatenate(all_student_cls, axis=0)
    all_batch_labels = np.concatenate(all_batch_labels, axis=0)

    labels = all_batch_labels
    embeddings = all_student_cls

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create a scatter plot of the t-SNE embeddings
    plt.figure(figsize=(20, 16))
    color_map = plt.cm.get_cmap('tab10', num_classes)  # Colormap for coloring samples

    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap=color_map, s=150)

    plt.colorbar(scatter)  # Add colorbar

    plt.savefig(out_file)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Visualize t-sne of embeddings")
    parser.add_argument('--model', type=str, help='Name of the model')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--name', type=str, help='output prefix', default=None)
    args = parser.parse_args()

    if args.model == 'dino2resnet':
        resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model = DistilledResNetModel(resnet, replace_fc=False)
        model.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))
    elif args.model == 'dinovit':
        model = ViTLightningModule()
        model.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))

    if args.dataset == 'CUB200':
        datamodule = CUB200DataModule(subset=[3, 15, 21, 46, 51, 84, 106, 111, 187, 200], batch_size=2)
        datamodule.prepare_data()
        datamodule.setup()
        dataloader = datamodule.val_dataloader()
        num_labels = 10
        labels = './dataset/caltech_birds2011/CUB_200_2011/classes.txt'
        class_names = []
        with open(labels, 'r') as f:
            for line in f:
                class_names.append(line.strip().split('.', 1)[1])
        extractor = cub_batch_extractor
    elif args.dataset == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='dataset/cifar', download=True, train=False,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((224, 224)),
                                                ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        num_labels = 100
        class_names = cifar100_labels
        extractor = cifar_batch_extractor
    elif args.dataset == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='dataset/cifar', download=True, train=False,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Resize((224, 224)),
                                               ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        num_labels = 10
        class_names = cifar100_labels
        extractor = cifar_batch_extractor

    model = model.to(device)
    if args.name is None:
        name = f'{args.model}_{args.dataset}'
    else:
        name = f'{args.name}_{args.model}_{args.dataset}'
    create_tsne_visualization(model,
                              dataloader,
                              num_labels,
                              class_names,
                              name,
                              extractor,
                              device)
