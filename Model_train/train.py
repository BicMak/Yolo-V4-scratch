"""Unified class to make training pipeline for deep neural networks."""
import os
import datetime

# Union은 여러 가지 가능한 타입을 변수나 함수의 반환 타입에 지정할 때 사용
# Callable은 외부에서 함수에 대한 변수 타입을 강제로 지정할 때 사용
# Path 경로 결합, 존재 여부 확인, 디렉토리 탐색, 파일 읽기/쓰기 등 다양한 기능을 제공
from typing import Union, Callable
from pathlib import Path




import torch

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:  # pylint: disable=too-many-instance-attributes
    """ Generic class for training loop.

    Parameters
    ----------
    initiatialize the trainer class.
    it control the training process.

    Parameters:
        model(torch.nn.Module): torch model to train.
        loader_train(torch.utils.DataLoader): train dataset loader.
        loader_test(torch.utils.DataLoader): test dataset loader.
        loss_fn(callable): loss function.
        metric_fn(callable): evaluation metric function.
        optimizer(torch.optim.Optimizer): Optimizer.
        lr_scheduler(torch.optim.LrScheduler): Learning Rate scheduler.
        device(Union[torch.device, str]): device to use for training.
        model_save_best(bool): save best model or not.
        model_saving_frequency(int): frequency of saving model.
        save_dir(Union[str, Path]): directory to save model.
    """
    def __init__( # pylint: disable=too-many-arguments
        self,
        model: torch.nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        device: Union[torch.device, str] = "cuda",
        model_save_best: bool = True,
        model_saving_frequency: int = 1,
        save_dir: Union[str, Path] = "checkpoints",
        stage_progress: bool = True,
    ):
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_save_best = model_save_best
        self.model_saving_frequency = model_saving_frequency
        self.save_dir = save_dir
        self.stage_progress = stage_progress
        self.hooks = {}
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "test_metric": []}

    def fit(self, epochs):
        """ Fit model method.

        Arguments:
            epochs (int): number of epochs to train model.
        """
        iterator = tqdm(range(epochs), dynamic_ncols=True)
        for epoch in iterator:
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
            )
            output_test = self.hooks["test"](
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                get_key_metric=self.get_key_metric
            )
            if self.visualizer:
                self.visualizer.update_charts(
                    None, output_train['loss'], output_test['metric'], output_test['loss'],
                    self.optimizer.param_groups[0]['lr'], epoch
                )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()

            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train, output_test)

            if self.model_save_best:
                best_acc = max([self.get_key_metric(item) for item in self.metrics['test_metric']])
                current_acc = self.get_key_metric(output_test['metric'])

                if current_acc >= best_acc:
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, self.model.__class__.__name__) + '_best.pth'
                    )
            else:
                if (epoch + 1) % self.model_saving_frequency == 0:
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_dir, self.model.__class__.__name__) + '_' +
                        str(datetime.datetime.now()) + '.pth'
                    )

        return self.metrics

    def register_hook(self, hook_type, hook_fn):
        """ Register hook method.

        Arguments:
            hook_type (string): hook type.
            hook_fn (callable): hook function.
        """
        self.hooks[hook_type] = hook_fn

    def _register_default_hooks(self):
        self.register_hook("train", train_hook_default)
        self.register_hook("test", test_hook_default)
        self.register_hook("end_epoch", None)
