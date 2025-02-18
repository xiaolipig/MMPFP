import argparse
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import torch
import torch.nn as torch_nn
from mindspore import Model
from MMPFP import FinalProteinModel
from data_loader import ProteinDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test FinalProteinModel with MindSpore")
    parser.add_argument('--data_path', type=str, default='./data/val_paths.txt', help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--n_pdb_files', type=int, default=10, help='Number of PDB files to use')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                        help='Device target')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')

    return parser.parse_args()


def test(args):
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = ProteinDataset(args.data_path, args.n_pdb_files)
    val_loader = ds.GeneratorDataset(val_dataset, ['protein', 'gcn', 'repvgg', 'label'], shuffle=False).batch(
        args.batch_size)

    model = FinalProteinModel(protein_dim=512, fusion_dim=256, output_dim=3)  # CC, MF, BP 三分类
    loss_fn = nn.CrossEntropyLoss()

    # loss_fn = torch_nn.CrossEntropyLoss()

    ms.load_checkpoint(args.ckpt_path, net=model)
    # model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    mindspore_model = Model(model, loss_fn=loss_fn, metrics={"accuracy"})

    total_loss = 0
    correct_preds = 0
    total_samples = 0

    for batch_data in val_loader.create_tuple_iterator():
        protein, gcn, repvgg, label = batch_data
        output = mindspore_model.predict((protein, gcn, repvgg))

        loss = loss_fn(output, label)
        total_loss += loss.asnumpy()

        predicted = ms.ops.Argmax(axis=1)(output)
        correct_preds += (predicted == label).sum().asnumpy()
        total_samples += label.shape[0]

        # with torch.no_grad():
        #     output = model(protein.to(device))
        #     loss = loss_fn(output, label.to(device))
        #     total_loss += loss.item()
        #     predicted = torch.argmax(output, dim=1)
        #     correct_preds += (predicted == label.to(device)).sum().item()
        #     total_samples += label.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_preds / total_samples
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    test(args)

# python test.py --batch_size 32 --device_target GPU --ckpt_path ./checkpoints/best_model.ckpt
