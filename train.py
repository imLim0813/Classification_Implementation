import argparse
import albumentations as A
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.AlexNet as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from data_loader.data_loaders import make_dataloder
from albumentations.pytorch import ToTensorV2
from logger.logger import Logger


# 랜덤 시드를 고정함으로써 동일한 조건으로 시작.
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config_):

    config = ConfigParser(config=config_, run_id='ver_0')
    train_config = {}

    # 1. 데이터 로더 생성
    loader_dict = dict(config.data_loader['args'])

    # Train / Test 전처리 객체 생성
    train_transform = A.Compose([
        A.Resize(loader_dict['img_size'][0], loader_dict['img_size'][1]),
        A.HorizontalFlip(p=0.5),
        A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(loader_dict['img_size'][0], loader_dict['img_size'][1]),
        A.augmentations.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_config['Image size'] = loader_dict['img_size']

    # 배치 사이즈를 리스트로 받아 리스트 원소 개수만큼 반복
    for batch_idx, batch_size in enumerate(loader_dict['batch_size']):
        train_config['Batch size'] = batch_size

        # 배치사이즈에 따른 데이터 로더 생성
        train_dataloader = make_dataloder(transform=train_transform, train_=True, batch_size=batch_size)

        valid_dataloader = make_dataloder(transform=test_transform, train_=False, batch_size=batch_size)

        # Learning rate를 리스트로 받아 리스트 원소 개수만큼 반복
        for lr_idx, learning_rate in enumerate(config.optimizer['args']['lr']):
            train_config['Initial LR'] = learning_rate

            # GPU 설정
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            train_config['GPU'] = True if torch.cuda.is_available() else 'False'

            # 손실 함수, 평가지표 함수 생성
            criterion = getattr(module_loss, config.loss_fn)
            metrics = [getattr(module_metric, met) for met in config.metric_fn]
            train_config['Loss function'] = config.loss_fn
            train_config['Metric'] = [metric for metric in config.metric_fn]

            for ep_idx, epoch in enumerate(config.trainer['epochs']):
                train_config['Epoch'] = epoch

                # 모델 생성
                model = getattr(module_arch, config.module_name)()
                model = model.to(device)

                # 옵티마이저 생성 및 learning rate scheduler 생성
                trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = getattr(torch.optim, config.optimizer['type'])(trainable_params, lr=learning_rate,
                                                                           weight_decay=config.optimizer['args'][
                                                                               'weight_decay'])
                lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler['type'])(optimizer=optimizer,
                                                                                              gamma=config.lr_scheduler[
                                                                                                  'args']['gamma'],
                                                                                              step_size=
                                                                                              config.lr_scheduler[
                                                                                                  'args']['step_size'])
                train_config['Optimizer'] = {'Name': config.optimizer['type'],
                                             'Weight decay': config.optimizer['args']['weight_decay']}
                train_config['LR scheduler'] = {'Name': config.lr_scheduler['type'],
                                                'gamma': config.lr_scheduler['args']['gamma'],
                                                'Step size': config.lr_scheduler['args']['step_size']}

                logger = Logger(config_file=train_config, p_name=config.module_name,
                                r_name=f"ver_{len(config.trainer['epochs']) * len(config.optimizer['args']['lr']) * batch_idx + len(config.trainer['epochs']) * lr_idx + ep_idx}")

                trainer = Trainer(model, criterion, metrics, optimizer, device, epoch, logger, config.save_dir,
                                  data_loader=train_dataloader, valid_data_loader=valid_dataloader,
                                  lr_scheduler=lr_scheduler)
                trainer.train()

                config = ConfigParser(config=config_,
                                      run_id=f"ver_{len(config.trainer['epochs']) * len(config.optimizer['args']['lr']) * batch_idx + len(config.trainer['epochs']) * lr_idx + ep_idx + 1}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Segmentation Template')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: ./config.json)')
    parser = args.parse_args()

    main(parser.config)