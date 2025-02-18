
import os
import copy
import pickle
import warnings
import argparse
import numpy as np
import sys
# sys.setrecursionlimit(3000)
import luojianet
import luojianet.nn as nn
import luojianet.ops as ops
from luojianet import load_param_into_net,load_checkpoint
from luojianet.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from luojianet.train.loss_scale_manager import FixedLossScaleManager
from luojianet import nn, ops, Parameter, Tensor, context, Model
from luojianet.communication import init, get_group_size, get_rank
from luojianet.parallel import set_algo_parameters
from luojianet.parallel._cost_model_context import _set_multi_subgraphs
import stat
from tqdm import tqdm
from configs import BuildConfig
from callback import StateMonitor
from luojianet.train import LearningRateScheduler
# from modules import (
#     BuildDataset, BuildDistributedDataloader,  BuildOptimizer, BuildScheduler, initslurm,
#     BuildLoss, BuildBackbone, BuildSegmentor, BuildPixelSampler, Logger, setRandomSeed, BuildPalette, checkdir, loadcheckpoints, savecheckpoints
# )
from modules import (
    BuildDataset, BuildDistributedDataloader,  BuildOptimizer, BuildScheduler, EvalCallback,
    BuildLoss, BuildBackbone, BuildSegmentor, Logger, checkdir
)
warnings.filterwarnings('ignore')

luojianet.set_seed(1)

'''parse arguments in command line'''
def parseArgs():
    parser = argparse.ArgumentParser(description='SSSegmentation is an open source supervised semantic segmentation toolbox based on PyTorch')
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed training', default=0, type=int)
    parser.add_argument('--slurm', dest='slurm', help='please add --slurm if you are using slurm', default=False, action='store_true')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_pdb_files', type=int, default=10, help='Number of PDB files to use')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'],
                        help='Device target')
    parser.add_argument('--ckpt_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval to save checkpoints')
    parser.add_argument('--keep_max_ckpts', type=int, default=5, help='Maximum number of checkpoints to keep')
    args = parser.parse_args()
    if args.slurm: initslurm(args, '29000')
    return args

def learning_rate_function(lr, cur_step_num):
    if cur_step_num % 7000  == 0:
        lr = lr * 0.1
    return lr

'''Trainer'''
class Trainer():
    def __init__(self, cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path):
        # set attribute
        self.cfg = cfg
        self.ngpus_per_node = ngpus_per_node
        self.logger_handle = logger_handle
        self.cmd_args = cmd_args
        self.cfg_file_path = cfg_file_path
        # assert torch.cuda.is_available(), 'cuda is not available'
        # init distributed training
        # dist.init_process_group(backend=self.cfg.SEGMENTOR_CFG.get('backend', 'nccl'))
    # def init_context(self):
    #     cfg = self.cfg
    #     luojianet.set_context(mode=luojianet.PYNATIVE_MODE)
    #     #luojianet.context.set_context(mode=luojianet.GRAPH_MODE)
    #     init()
    #     rank_id, device_num = get_rank(), get_group_size()
    #     luojianet.context.set_auto_parallel_context(parallel_mode= luojianet.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        
    #     luojianet.common.set_seed(3047)
    #     print('--------------distribute init done!--------------')
    '''start trainer'''
    def start(self):

        luojianet.set_context(mode=luojianet.GRAPH_MODE,device_target="Ascend", max_device_memory="5GB")
        #luojianet.set_context(mode=luojianet.PYNATIVE_MODE)
        #init()
        #rank_id, device_num = get_rank(), get_group_size()
        #luojianet.set_auto_parallel_context(parallel_mode= luojianet.ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num)
        #luojianet.set_seed(3047)
        print('--------------distribute init done!--------------')
        cfg, ngpus_per_node, logger_handle, cmd_args, cfg_file_path = self.cfg, self.ngpus_per_node, self.logger_handle, self.cmd_args, self.cfg_file_path
        # build dataset and dataloader
        dataset = BuildDataset(mode='TRAIN', logger_handle=logger_handle, dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        # assert dataset.source.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        assert dataset.num_classes == cfg.SEGMENTOR_CFG['num_classes'], 'parsed config file %s error' % cfg_file_path
        dataloader_cfg = copy.deepcopy(cfg.DATALOADER_CFG)
        # batch_size, num_workers = dataloader_cfg['train']['batch_size'], dataloader_cfg['train']['num_workers']
        # batch_size_per_node = batch_size // ngpus_per_node
        # num_workers_per_node = num_workers // ngpus_per_node
        # dataloader_cfg['train'].update({'batch_size': batch_size_per_node, 'num_workers': num_workers_per_node})
        dataloader = BuildDistributedDataloader(dataset=dataset, dataloader_cfg=dataloader_cfg['train'], dataset_cfg=copy.deepcopy(cfg.DATASET_CFG))
        # build segmentor
        segmentor = BuildSegmentor(segmentor_cfg=copy.deepcopy(cfg.SEGMENTOR_CFG), losses_cfg=copy.deepcopy(cfg.LOSSES_CFG), mode='TRAIN')
        if segmentor.__class__.__name__ == 'FastFCN':
           segmentor = segmentor.segmentor
        else:
            segmentor = segmentor
        
        segmentor.set_train(True)
        

        
        optimizer_cfg = copy.deepcopy(cfg.OPTIMIZER_CFG)
        optimizer = BuildOptimizer(segmentor, optimizer_cfg)
        
        scheduler_cfg = copy.deepcopy(cfg.SCHEDULER_CFG)
       
        start_epoch, end_epoch = 1, scheduler_cfg['max_epochs']
        # load checkpoints
        if cmd_args.checkpointspath and os.path.exists(cmd_args.checkpointspath):
           
            print("************************")
            print(cmd_args.checkpointspath)
            print("************************")
            param_dict = load_checkpoint(cmd_args.checkpointspath)
            load_param_into_net(net=segmentor, parameter_dict=param_dict)

        
        else:
            cmd_args.checkpointspath = ''
       
        if (cmd_args.local_rank == 0) and (int(os.environ.get('SLURM_PROCID', 0)) == 0):
            logger_handle.info(f'Config file path: {cfg_file_path}')
            logger_handle.info(f'DATASET_CFG: \n{cfg.DATASET_CFG}')
            logger_handle.info(f'DATALOADER_CFG: \n{cfg.DATALOADER_CFG}')
            logger_handle.info(f'OPTIMIZER_CFG: \n{cfg.OPTIMIZER_CFG}')
            logger_handle.info(f'SCHEDULER_CFG: \n{cfg.SCHEDULER_CFG}')
            logger_handle.info(f'LOSSES_CFG: \n{cfg.LOSSES_CFG}')
            logger_handle.info(f'SEGMENTOR_CFG: \n{cfg.SEGMENTOR_CFG}')
            logger_handle.info(f'INFERENCE_CFG: \n{cfg.INFERENCE_CFG}')
            logger_handle.info(f'COMMON_CFG: \n{cfg.COMMON_CFG}')
            logger_handle.info(f'Resume from: {cmd_args.checkpointspath}')
       
        model = Model(segmentor, optimizer=optimizer)
   
        loss_cb = LossMonitor(1)

        state_cb = StateMonitor(
        model,
        cfg.SEGMENTOR_CFG['type'],
        last_epoch=0,
        dataset_sink_mode=False,
        dataset_val=None,
        metric_name=("accuracy",),
        val_interval=100,
        val_start_epoch=100,
        save_best_ckpt=True,
        ckpt_save_dir=cfg.COMMON_CFG['work_dir'],
        ckpt_save_interval=1,
        ckpt_save_policy='latest_k',
        ckpt_keep_max=10,
        summary_dir=cfg.COMMON_CFG['work_dir'],
        log_interval=1,
        #rank_id=rank_id,
        #device_num=device_num

        )
        # Save-checkpoint callback
        savepath = os.path.join(cfg.COMMON_CFG['work_dir'])
        # set_permissions(cfg.COMMON_CFG['work_dir'])
        ckpt_config = CheckpointConfig(save_checkpoint_steps=500,
                                       keep_checkpoint_max=100,async_save=True)
        ckpt_cb = ModelCheckpoint(prefix='{}'.format("epoch"),
                                  directory=savepath,
                                  config=ckpt_config)
        #cb = [loss_cb, ckpt_cb, LearningRateScheduler(learning_rate_function)]
        cb = [state_cb,LearningRateScheduler(learning_rate_function)]


        train_epoch = end_epoch - start_epoch
        from luojianet import SummaryCollector
        summ = SummaryCollector(summary_dir=cfg.COMMON_CFG['work_dir'])
        #cb = [loss_cb, ckpt_cb, LearningRateScheduler(learning_rate_function)]
        model.train(train_epoch, dataloader, callbacks=cb, dataset_sink_mode=False)        
        



def set_permissions(folder_path):
    files = os.listdir(folder_path)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        os.chmod(file_path, stat.S_IWRITE | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | 
                               stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | 
                               stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)




'''main'''
def main():
    
    args = parseArgs()
    cfg, cfg_file_path = BuildConfig(args.cfgfilepath)
    # check work dir
    checkdir(cfg.COMMON_CFG['work_dir'])
    # initialize logger_handle
    logger_handle = Logger(cfg.COMMON_CFG['logfilepath'])
    ngpus_per_node = 1
    client = Trainer(cfg=cfg, ngpus_per_node=ngpus_per_node, logger_handle=logger_handle, cmd_args=args, cfg_file_path=cfg_file_path)
    client.start()


'''debug'''
if __name__ == '__main__':
    main()

# python train.py  --epochs 100 --batch_size 32 --learning_rate 0.0001 --device_target GPU --ckpt_dir ./checkpoints