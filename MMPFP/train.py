import os
import argparse
import mindspore as ms
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import torch
import torch.nn as torch_nn
import torch.optim as optim
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import Model
from MMPFP import FinalProteinModel
from data_loader import ProteinDataset
from callback import StateMonitor
from mindspore.communication import init, get_group_size, get_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train FinalProteinModel with MindSpore")
    parser.add_argument('--data_path', type=str, default='./your/data/txt', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_pdb_files', type=int, default=10, help='Number of PDB files to use')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                        help='Device target')
    parser.add_argument('--ckpt_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval to save checkpoints')
    parser.add_argument('--keep_max_ckpts', type=int, default=5, help='Maximum number of checkpoints to keep')

    return parser.parse_args()


def train(args):
    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)
    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ms.set_context(mode=ms.PYNATIVE_MODE)

    # parallel mode
    # init()
    # rank_id, device_num = get_rank(), get_group_size()
    # mindspore.set_auto_parallel_context(parallel_mode= mindspore.ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num)

    train_dataset = ProteinDataset(args.data_path, args.n_pdb_files)
    val_dataset = ProteinDataset(args.data_path, args.n_pdb_files)

    train_loader = ds.GeneratorDataset(train_dataset, ['protein', 'gcn', 'repvgg', 'label'], shuffle=True).batch(
        args.batch_size)
    val_loader = ds.GeneratorDataset(val_dataset, ['protein', 'gcn', 'repvgg', 'label'], shuffle=False).batch(
        args.batch_size)

    model = FinalProteinModel(protein_dim=512, fusion_dim=256, output_dim=3)  # CC, MF, BP
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.learning_rate)

    # loss_fn = torch_nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    mindspore_model = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy"})

    state_cb = StateMonitor(
        model,
        last_epoch=0,
        dataset_sink_mode=False,
        dataset_val=None,
        metric_name=("accuracy",),
        val_interval=100,
        val_start_epoch=100,
        save_best_ckpt=True,
        ckpt_save_dir=args.ckpt_dir,
        ckpt_save_interval=1,
        ckpt_save_policy='latest_k',
        ckpt_keep_max=args.keep_max_ckpts,
        summary_dir='./',
        log_interval=1,
    )

    loss_monitor = LossMonitor(per_print_times=1)
    checkpoint_config = CheckpointConfig(save_checkpoint_steps=args.save_interval,
                                         keep_checkpoint_max=args.keep_max_ckpts)
    checkpoint_cb = ModelCheckpoint(directory=args.ckpt_dir, config=checkpoint_config)

    mindspore_model.train(args.epochs, train_loader, callbacks=[loss_monitor, checkpoint_cb, state_cb],
                          dataset_sink_mode=False)

    # for epoch in range(args.epochs):
    #     model.train()
    #     for batch in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(batch['protein'])
    #         loss = loss_fn(outputs, batch['label'])
    #         loss.backward()
    #         optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    train(args)
