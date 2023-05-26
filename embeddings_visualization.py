import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from DistilledResnetModel import DistilledResNetModel
from CUB200DataModule import CUB200DataModule


def create_tsne_visualization(model, test_loader, num_classes, class_names_file, out_file):
    embeddings = []
    labels = []

    # Read class names from the file
    class_names = []
    with open(class_names_file, 'r') as f:
        for line in f:
            class_names.append(line.strip().split('.', 1)[1])

    # Iterate over the test set
    all_student_cls = []
    all_batch_labels = []
    for i, batch in enumerate(test_loader):
        print(f'Batch {i}/{len(test_loader)}')
        imgs, batch_labels = batch['img'], batch['label']
        outs = model(imgs)
        student_cls = outs['cls'].detach()

        all_student_cls.append(student_cls.cpu().numpy())
        all_batch_labels.append(batch_labels.cpu().numpy())

    all_student_cls = np.concatenate(all_student_cls, axis=0)
    all_batch_labels = np.concatenate(all_batch_labels, axis=0)

    # Collect embeddings and labels for each class
    for class_label in range(num_classes):
        print(f'Label {class_label}/{num_classes}')
        class_indices = all_batch_labels == class_label
        class_embeddings = all_student_cls[class_indices]
        if class_embeddings.size == 0:
            continue
        class_label = all_batch_labels[class_indices][0]
        class_avg_embedding = np.mean(class_embeddings, axis=0)

        embeddings.append(class_avg_embedding)
        labels.append(class_label.item())

    # Convert the lists to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create a scatter plot of the t-SNE embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])

    # Add labels to the data points
    for i, label in enumerate(labels):
        plt.annotate(class_names[label], (embeddings_tsne[i, 0], embeddings_tsne[i, 1]), fontsize=8)

    plt.savefig(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize t-sne of embeddings")
    parser.add_argument('--model', type=str, help='Name of the model')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    args = parser.parse_args()

    if args.model == 'dino2resnet':
        resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        model = DistilledResNetModel(resnet, replace_fc=False)
        model.load_from_checkpoint(args.checkpoint)

    if args.dataset == 'CUB200':
        datamodule = CUB200DataModule()
        datamodule.prepare_data()
        datamodule.setup()
        dataloader = datamodule.val_dataloader()
        num_labels = 200
        labels = './dataset/caltech_birds2011/CUB_200_2011/classes.txt'

    create_tsne_visualization(model,
                              dataloader,
                              num_labels,
                              labels,
                              f'{args.model}_{args.dataset}')
