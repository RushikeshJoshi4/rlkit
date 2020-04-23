import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn as nn

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    # def __init__(
    #         self,
    #         trainer,
    #         exploration_env,
    #         evaluation_env,
    #         exploration_data_collector: PathCollector,
    #         evaluation_data_collector: PathCollector,
    #         replay_buffer: ReplayBuffer,
    #         batch_size,
    #         max_path_length,
    #         num_epochs,
    #         num_eval_steps_per_epoch,
    #         num_expl_steps_per_train_loop,
    #         num_trains_per_train_loop,
    #         num_train_loops_per_epoch=1,
    #         min_num_steps_before_training=0,
    #         initial_epoch=0):
    #     super().__init__(
    #         trainer,
    #         exploration_env,
    #         evaluation_env,
    #         exploration_data_collector,
    #         evaluation_data_collector,
    #         replay_buffer,
    #         batch_size,
    #         max_path_length,
    #         num_epochs,
    #         num_eval_steps_per_epoch,
    #         num_expl_steps_per_train_loop,
    #         num_trains_per_train_loop,
    #         num_train_loops_per_epoch=1,
    #         min_num_steps_before_training=0,
    #         initial_epoch=0
    #     )


    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
