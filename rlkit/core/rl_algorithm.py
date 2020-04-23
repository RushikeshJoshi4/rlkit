import abc
from collections import OrderedDict

import gtimer as gt
import torch
import os
import copy

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
            initial_epoch
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = initial_epoch

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        # print("\n\nhello world\n\n")
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def get_cur_best_metric_val(self):
        cur_best_metric_val = None
        if os.path.exists('cur_best_avg_rewards.pkl'):
            cur_best_metric_val = torch.load('cur_best_avg_rewards.pkl')
        else:
            cur_best_metric_val = -1* float('inf')
        return cur_best_metric_val

    def _end_epoch(self, epoch):
        print('in _end_epoch, epoch is: {}'.format(epoch))
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        # trainer_obj = self.trainer
        # ckpt_path='ckpt.pkl'
        # logger.save_ckpt(epoch, trainer_obj, ckpt_path)
        # gt.stamp('saving')
        if epoch%1==0:
            self.save_snapshot_2(epoch)
        expl_paths = self.expl_data_collector.get_epoch_paths()
        d = eval_util.get_generic_path_information(expl_paths)
        # print(d.keys())
        metric_val = d['Rewards Mean']
        
        cur_best_metric_val = self.get_cur_best_metric_val()
        if epoch!=0: self.save_snapshot_2_best_only(metric_val=metric_val, cur_best_metric_val=cur_best_metric_val, min_or_max='max')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def save_snapshot_2(self, epoch):
        # print('Saving s')
        print('Saving snapshot 2')
        torch.save({'algorithm':self, 'epoch':epoch}, 'ckpt.pkl')

    def get_snapshot_2(self):
        print('in get_snapshot_2')
        ckpt = {}
        ckpt = torch.load('ckpt.pkl')
        self = copy.deepcopy(ckpt['algorithm'])
        epoch = ckpt['epoch']
        return epoch

    def save_snapshot_2_best_only(self, metric_val, cur_best_metric_val, min_or_max='min'):
        if min_or_max == 'min' and metric_val < cur_best_metric_val \
            or min_or_max == 'max' and metric_val > cur_best_metric_val:
            print('Saving snapshot best')
            torch.save(self, 'ckpt-best.pkl')
            cur_best_metric_val = metric_val
            torch.save('cur_best_avg_rewards.pkl')

    # def _resume_training(self):

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        # gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
