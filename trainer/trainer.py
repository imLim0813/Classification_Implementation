import numpy as np
import torch
import wandb
from torch.utils.data import dataloader
from typing import List, Union
from timeit import default_timer as timer


class Trainer:
    """
    세그멘테이션 모델 학습을 위한 클래스
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, metric_fn: Union[torch.nn.Module, torch.nn.Module],
                 optimizer: torch.optim.Optimizer, device: str, len_epoch: int, logger, save_dir,
                 data_loader: torch.utils.data.DataLoader, valid_data_loader: torch.utils.data.DataLoader = None,
                 lr_scheduler: torch.optim.lr_scheduler = None):

        # CUDA // device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # make_dataloder 함수의 결과 값
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        # config.json 파일로부터 파라미터를 호출받고, getattr로 생성된 객체
        self.lr_scheduler = lr_scheduler

        # model/metric.py의 함수들 전부 호출
        self.metric_fn = metric_fn

        # model/model.py -> U-Net, DeepLab v3, FCN, etc..
        self.model = model

        # config.json 파일로부터 파라미터를 호출받고, getattr로 생성된 객체
        self.criterion = criterion
        self.optimizer = optimizer

        # config.json 파일로부터 파라미터를 호출받아 생성된 int 변수
        self.epochs = len_epoch

        # logger/logger.py의 Logger 클래스를 이용하여 학습로그 기록 객체 생성
        self.logger = logger

        # 모델을 저장 할 경로 설정
        self.save_dir = save_dir

        # Early Stopping을 위한 딕셔너리 생성
        self.es_log = {'train_loss' : [], 'val_loss' : []}

        self.not_improved = 0
        self.early_stop = 10
        self.save_period = 10
        self.mnt_best = np.inf

    def _train_epoch(self, epoch: int):
        train_loss = 0
        train_metric = {f'{metric.__name__}': [] for metric in self.metric_fn}

        self.model.train()
        for batch, (x_train, y_train) in enumerate(self.data_loader):
            x_train, y_train = x_train.to(self.device), y_train.to(self.device).long()
            y_pred = self.model(x_train)
            loss = self.criterion(y_pred, y_train)

            train_loss += loss.item() * x_train.size(0)
            met_ = {f'{metric.__name__}': metric(x_train, y_train, self.model, self.device) for metric in self.metric_fn}
            for key, value in met_.items():
                train_metric[key].append(value)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.data_loader.dataset)
        train_acc = list(map(lambda x: sum(x) / len(self.data_loader), train_metric.values()))[0]
        print(f'Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}% | ', end='')
        self.es_log['train_loss'].append(train_loss)

        if self.do_validation:
            val_loss, val_acc = self._valid_epoch(epoch)
            self.logger.record({'Train Loss': train_loss, 'Train Acc': train_acc, 'Val Loss': val_loss, 'Val Acc': val_acc})
        else:
            self.logger.record(
                {'Train Loss': train_loss, 'Train Acc': train_acc})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Early_Stopping
        best = False

        improved = (self.es_log['val_loss'][-1] <= self.mnt_best)
        if improved:
            self.mnt_best = self.es_log['val_loss'][-1]
            self.not_improved_count = 0
            best = True
        else:
            self.not_improved_count += 1

        if epoch % self.save_period == 0:
            self._save_checkpoint(epoch, save_best=best)

    def _valid_epoch(self, epoch: int):
        val_loss = 0
        val_metric = {f'{metric.__name__}': [] for metric in self.metric_fn}

        self.model.eval()
        with torch.inference_mode():
            for (x_test, y_test) in self.valid_data_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device).long()
                y_pred = self.model(x_test)
                loss = self.criterion(y_pred, y_test)

                val_loss += loss.item() * x_test.size(0)
                met_ = {f'{metric.__name__}': metric(x_test, y_test, self.model, self.device) for metric in self.metric_fn}
                for key, value in met_.items():
                    val_metric[key].append(value)

            val_loss /= len(self.valid_data_loader.dataset)
            val_acc = list(map(lambda x: sum(x) / len(self.valid_data_loader), val_metric.values()))[0]
            print(f'Val Loss : {val_loss:.4f} | Val Acc : {val_acc:.4f}% | ', end='')
            self.es_log['val_loss'].append(val_loss)

            return val_loss, val_acc

    def train(self):
        for epoch in range(self.epochs):
            print(f'\nEpoch : {epoch} | ', end='')
            start_time = timer()
            self._train_epoch(epoch)
            end_time = timer()
            print(f'Training Time : {(end_time-start_time):.2f}sec')

            if self.not_improved_count > self.early_stop:
                print("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                break

        self.logger.finish()

    def _save_checkpoint(self, epoch, save_best=False):
        filename = str(self.save_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(self.model.state_dict(), filename)
        if save_best:
            best_path = str(self.save_dir / 'model_best.pth')
            torch.save(self.model.state_dict(), best_path)