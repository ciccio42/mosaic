import wandb
from train_utils import *
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from mosaic.utils.early_stopping import EarlyStopping
from copy import deepcopy
import learn2learn as l2l
import os
import json
import copy
import torch
import hydra
import torch.nn as nn
from os.path import join
from omegaconf import OmegaConf
from mosaic.utils.lr_scheduler import build_scheduler
from collections import defaultdict
torch.autograd.set_detect_anomaly(True)


class Trainer:

    def __init__(self, allow_val_grad=False, hydra_cfg=None):
        assert hydra_cfg is not None, "Need to start with hydra-enabled yaml file!"
        self.config = hydra_cfg
        self.train_cfg = hydra_cfg.train_cfg
        # initialize device
        def_device = hydra_cfg.device if hydra_cfg.device != -1 else 0
        self._device = torch.device("cuda:{}".format(def_device))
        self._device_list = None
        self._allow_val_grad = allow_val_grad
        # set of file saving

        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'
        self._best_validation_loss = float('inf')
        self._best_validation_weights = None

        append = "-Batch{}".format(int(self.config.bsize))
        if 'target_obj_detector' in hydra_cfg.policy._target_:
            append = "-Batch{}-{}gpu-Attn{}ly{}-Act{}ly{}mix{}".format(
                int(self.config.bsize), int(torch.cuda.device_count()),
                int(self.config.policy.attn_cfg.n_attn_layers), int(
                    self.config.policy.attn_cfg.attn_ff),
                int(self.config.policy.mlp_cfg.n_layers), int(
                    self.config.policy.mlp_cfg.out_dim),
                int(self.config.policy.mlp_cfg.n_mixtures))

            if self.config.policy.concat_demo_head:
                append += "-headCat"
            elif self.config.policy.concat_demo_act:
                append += "-actCat"
            else:
                append += "-noCat"

        self.config.exp_name += append

        save_dir = join(self.config.get('save_path', './'),
                        str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        self._step = None
        if self.config.wandb_log:
            config_keys = ['train_cfg', 'tasks',
                           'samplers', 'dataset_cfg', 'policy']
            # for k in config_keys:
            #     print(k, self.config.get(k))
            #     print(k, dict(self.config.get(k)))
            #     print('-'*20)
            wandb_config = {k: self.config.get(k) for k in config_keys}
            run = wandb.init(project=self.config.project_name,
                             name=self.config.exp_name, config=wandb_config)

        # create early stopping object
        self._early_stopping = EarlyStopping(patience=self.train_cfg.early_stopping.patience,
                                             verbose=True,
                                             delta=self.train_cfg.early_stopping.delta,
                                             path=self.save_dir
                                             )

    def train(self, model, weights_fn=None, save_fn=None, optim_weights=None, optimizer_state_dict=None):

        self._train_loader, self._val_loader = make_data_loaders(
            self.config, self.train_cfg.dataset)
        # wrap model in DataParallel if needed and transfer to correct device
        print('Training stage \n Found {} GPU devices \n'.format(self.device_count))
        model = model.to(self._device)
        if self.device_count > 1 and not isinstance(model, nn.DataParallel):
            print("Training stage \n Device list: {}".format(self.device_list))
            model = nn.DataParallel(model, device_ids=self.device_list)

        # initialize optimizer and lr scheduler
        optim_weights = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(
            self.config.train_cfg.optimizer, optim_weights, optimizer_state_dict, self.train_cfg)

        # initialize constants:
        # compute epochs
        if self.config.resume:
            epochs = self.config.epochs - \
                int(self.config.resume_step/len(self._train_loader))
            print(f"Remaining epochs {epochs}")
            self._step = int(self.config.resume_step)
            print(f"Starting step {self._step}")
        else:
            epochs = self.train_cfg.get('epochs', 1)
            self._step = 0

        vlm_alpha = self.train_cfg.get('vlm_alpha', 0.6)
        log_freq = self.train_cfg.get('log_freq', 1000)

        self.tasks = self.config.tasks
        num_tasks = len(self.tasks)
        sum_mul = sum([task.get('loss_mul', 1) for task in self.tasks])
        task_loss_muls = {task.name:
                          float("{:3f}".format(task.get('loss_mul', 1) / sum_mul)) for task in self.tasks}
        print(" Weighting each task loss separately:", task_loss_muls)
        self.generated_png = False
        val_iter = iter(self._val_loader)
        # log stats to both 'task_name/loss_name' AND 'loss_name/task_name'
        raw_stats = dict()
        print(f"Training for {epochs} epochs train dataloader has length {len(self._train_loader)}, \
                which sums to {epochs * len(self._train_loader)} total train steps, \
                validation loader has length {len(self._val_loader)}")

        train_loss = nn.CrossEntropyLoss(reduction="mean")
        train_accuracy = Accuracy(
            task="multiclass", num_classes=4).to(device=0)
        val_loss = nn.CrossEntropyLoss(reduction="mean")
        val_accuracy = Accuracy(task="multiclass", num_classes=4).to(device=0)
        torch.autograd.set_detect_anomaly(True)
        for e in range(epochs):
            frac = e / epochs

            for i, inputs in tqdm(enumerate(self._train_loader), total=len(self._train_loader), leave=False):
                tolog = {}

                # Save stats
                if (self._step % len(self._train_loader) == 0):  # stats
                    stats_save_name = join(
                        self.save_dir, 'stats', '{}.json'.format('train_val_stats'))
                    json.dump({k: str(v) for k, v in raw_stats.items()},
                              open(stats_save_name, 'w'))

                optimizer.zero_grad()
                # self.batch_distribution(inputs)

                # calculate loss here:
                task_losses, task_accuracy = calculate_obj_pos_loss(
                    self.config, self.train_cfg, self._device, model, inputs, train_loss, train_accuracy)
                task_names = sorted(task_losses.keys())
                weighted_task_loss = sum([l["ce_loss"] * task_loss_muls.get(name) for name, l
                                          in task_losses.items()])
                weighted_accuracy = sum(
                    [acc["accuracy"] * task_loss_muls.get(name) for name, acc in task_accuracy.items()])
                weighted_task_loss.backward()
                optimizer.step()
                ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
                # calculate train iter stats
                if self._step % log_freq == 0:
                    if self.config.wandb_log:
                        tolog['Train Step'] = self._step
                        for task_name, losses in task_losses.items():
                            for loss_name, loss_val in losses.items():
                                tolog[f'train/{loss_name}/{task_name}'] = loss_val
                                tolog[f'train/{task_name}/{loss_name}'] = loss_val

                        for task_name, acc in task_accuracy.items():
                            for acc_name, acc_val in acc.items():
                                tolog[f'train/{acc_name}/{task_name}'] = acc_val
                                tolog[f'train/{task_name}/{acc_name}'] = acc_val

                    if (self._step % len(self._train_loader) == 0):
                        print(
                            'Training epoch {1}/{2}, step {0}: \t '.format(self._step, e, epochs))
                        print(
                            f"Train Weighted CE Loss {weighted_task_loss} - Train Accuracy {weighted_accuracy}")

                #### ---- Validation step ----####
                if (self._step % len(self._train_loader) == 0):
                    # exhaust all data in val loader and take avg loss
                    all_val_losses = {task: defaultdict(
                        list) for task in task_names}
                    all_val_accuracy = {task: defaultdict(
                        list) for task in task_names}

                    val_iter = iter(self._val_loader)
                    model = model.eval()
                    for val_inputs in val_iter:
                        with torch.no_grad():
                            val_task_losses, val_task_accuracy = calculate_obj_pos_loss(
                                self.config, self.train_cfg, self._device, model, val_inputs, val_loss, val_accuracy)

                        for task, losses in val_task_losses.items():
                            for k, v in losses.items():
                                all_val_losses[task][k].append(v)

                        for task, accuracy in val_task_accuracy.items():
                            for k, v in accuracy.items():
                                all_val_accuracy[task][k].append(v)

                    # take average across all batches in the val loader
                    avg_losses = dict()
                    for task, losses in all_val_losses.items():
                        avg_losses[task] = {
                            k: torch.mean(torch.stack(v)) for k, v in losses.items()}

                    avg_accuracy = dict()
                    for task, accuracy in all_val_accuracy.items():
                        avg_accuracy[task] = {
                            k: torch.mean(torch.stack(v)) for k, v in accuracy.items()}

                    if self.config.wandb_log:
                        tolog['Validation Step'] = self._step
                        for task_name, losses in avg_losses.items():
                            for loss_name, loss_val in losses.items():
                                tolog[f'val/{loss_name}/{task_name}'] = loss_val
                                tolog[f'val/{task_name}/{loss_name}'] = loss_val

                            for task_name, acc in avg_accuracy.items():
                                for acc_name, acc_val in acc.items():
                                    tolog[f'val/{acc_name}/{task_name}'] = acc_val
                                    tolog[f'val/{task_name}/{acc_name}'] = acc_val

                    # compute the sum of validation losses
                    weighted_task_loss_val = sum(
                        [l["ce_loss"] * task_loss_muls.get(name) for name, l in avg_losses.items()])
                    weighted_accuracy_val = sum(
                        [acc["accuracy"] * task_loss_muls.get(name) for name, acc in avg_accuracy.items()])

                    if (self._step % len(self._train_loader) == 0):
                        print('Validation step {}:'.format(self._step))
                        print(
                            f"CE val loss {weighted_task_loss_val} - Validation accuracy {weighted_accuracy_val}")

                    if self.config.train_cfg.lr_schedule['type'] is not None:
                        # perform lr-scheduling step
                        scheduler.step(val_loss=weighted_task_loss_val)
                        if self.config.wandb_log:
                            # log learning-rate
                            tolog['Validation Step'] = self._step
                            tolog['learning_rate'] = scheduler._schedule.optimizer.param_groups[0]['lr']

                    # check for early stopping
                    self._early_stopping(
                        weighted_task_loss_val, model, self._step, optimizer)

                    model = model.train()
                    if self._early_stopping.early_stop:
                        break

                    self.save_checkpoint(
                        model, optimizer, weights_fn, save_fn)

                if self.config.wandb_log:
                    wandb.log(tolog)

                self._step += 1
                # update target params
                # mod = model.module if isinstance(model, nn.DataParallel) else model
                # if self.train_cfg.target_update_freq > -1:
                #     mod.momentum_update(frac)
                #     if self._step % self.train_cfg.target_update_freq == 0:
                #         mod.soft_param_update()

            if self._early_stopping.early_stop:
                print("----Stop training for early-stopping----")
                break

        # when all epochs are done, save model one last time
        self.save_checkpoint(model, optimizer, weights_fn, save_fn)

    def save_checkpoint(self, model, optimizer, weights_fn=None, save_fn=None):
        if save_fn is not None:
            save_fn(self._save_fname, self._step)
        else:
            save_module = model
            if weights_fn is not None:
                save_module = weights_fn()
            elif isinstance(model, nn.DataParallel):
                save_module = model.module
            torch.save(save_module.state_dict(),
                       self._save_fname + '-{}.pt'.format(self._step))
        if self.config.get('save_optim', False):
            torch.save(optimizer.state_dict(), self._save_fname +
                       '-optim-{}.pt'.format(self._step))
        print(f'Model checkpoint saved at step {self._step}')
        return

    @property
    def device_count(self):
        if self._device_list is None:
            return torch.cuda.device_count()
        return len(self._device_list)

    @property
    def device_list(self):
        if self._device_list is None:
            return [i for i in range(torch.cuda.device_count())]
        return copy.deepcopy(self._device_list)

    @property
    def device(self):
        return copy.deepcopy(self._device)

    def _build_optimizer_and_scheduler(self, optimizer, optim_weights, optimizer_state_dict, cfg):
        assert self.device_list is not None, str(self.device_list)
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                optim_weights, cfg.lr, weight_decay=cfg.get('weight_decay', 0))
        elif optimizer == 'RMSProp':
            optimizer = torch.optim.RMSprop(
                optim_weights, cfg.lr, weight_decay=cfg.get('weight_decay', 0))

        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        print(
            f"Creating {optimizer}, with lr {optimizer.param_groups[0]['lr']}")

        return optimizer, build_scheduler(optimizer, cfg.get('lr_schedule', {}))

    def _loss_to_scalar(self, loss):
        """For more readable logging"""
        x = loss.item()
        return float("{:.3f}".format(x))

    @property
    def step(self):
        if self._step is None:
            raise Exception("Optimization has not begun!")
        return self._step

    @property
    def is_img_log_step(self):
        return self._step % self._img_log_freq == 0


class Workspace(object):
    """ Initializes the policy model and prepare for Trainer.train() """

    def __init__(self, cfg):
        self.trainer = Trainer(allow_val_grad=False, hydra_cfg=cfg)
        print("Finished initializing trainer")
        config = self.trainer.config
        resume = config.get('resume', False)
        self.action_model = hydra.utils.instantiate(config.policy)
        config.use_daml = 'DAMLNetwork' in cfg.policy._target_
        if config.use_daml:
            print("Switching to l2l.algorithms.MAML")
            self.action_model = l2l.algorithms.MAML(
                self.action_model,
                lr=config['policy']['maml_lr'],
                first_order=config['policy']['first_order'],
                allow_unused=True)

        print("Action model initialized to: {}".format(config.policy._target_))
        if resume:
            self._rpath = join(cfg.save_path, cfg.resume_path,
                               f"model_save-{cfg.resume_step}.pt")
            assert os.path.exists(self._rpath), "Can't seem to find {} anywhere".format(
                config.resume_path)
            print('load model from ...%s' % self._rpath)
            self.action_model.load_state_dict(torch.load(
                self._rpath, map_location=torch.device('cpu')))
            # create path for loading state dict
            optimizer_state_dict = join(
                cfg.save_path, cfg.resume_path, f"model_save-optim-{cfg.resume_step}.pt")
            self.optimizer_state_dict = torch.load(
                optimizer_state_dict, map_location=torch.device('cpu'))
        else:
            self.optimizer_state_dict = None

        self.config = config
        self.train_cfg = config.train_cfg

        # move log path to here!
        print('\n Done initializing Workspace, saving config.yaml to directory: {}'.format(
            self.trainer.save_dir))

        try:
            os.makedirs(self.trainer.save_dir, exist_ok=(
                'burn' in self.trainer.save_dir))
            os.makedirs(join(self.trainer.save_dir, 'stats'), exist_ok=True)
        except:
            pass

        save_config = copy.deepcopy(self.trainer.config)
        OmegaConf.save(config=save_config, f=join(
            self.trainer.save_dir, 'config.yaml'))

    def run(self):
        self.trainer.train(model=self.action_model,
                           optimizer_state_dict=self.optimizer_state_dict)
        print("Done training")


@hydra.main(
    version_base=None,
    config_path="experiments",
    config_name="config.yaml")
def main(cfg):
    print("---- Target Obj pos ----")
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    from train_target_obj_pos import Workspace as W
    all_tasks_cfgs = [cfg.tasks_cfgs.nut_assembly, cfg.tasks_cfgs.door, cfg.tasks_cfgs.drawer,
                      cfg.tasks_cfgs.button, cfg.tasks_cfgs.pick_place, cfg.tasks_cfgs.stack_block, cfg.tasks_cfgs.basketball]

    if cfg.task_names:
        cfg.tasks = [
            tsk for tsk in all_tasks_cfgs if tsk.name in cfg.task_names]

    if cfg.use_all_tasks:
        print("Loading all 7 tasks to the dataset!  obs_T: {} demo_T: {}".format(
            cfg.dataset_cfg.obs_T, cfg.dataset_cfg.demo_T))
        cfg.tasks = all_tasks_cfgs

    if cfg.exclude_task:
        print(f"Training with 6 tasks and exclude {cfg.exclude_task}")
        cfg.tasks = [
            tsk for tsk in all_tasks_cfgs if tsk.name != cfg.exclude_task]

    if cfg.set_same_n > -1:
        for tsk in cfg.tasks:
            tsk.n_per_task = cfg.set_same_n
        cfg.bsize = sum([tsk.n_tasks * cfg.set_same_n for tsk in cfg.tasks])
        cfg.vsize = cfg.bsize
        print(
            f'To construct a training batch, set n_per_task of all tasks to {cfg.set_same_n}, new train/val batch sizes: {cfg.train_cfg.batch_size}/{cfg.train_cfg.val_size}')

    if cfg.limit_num_traj > -1:
        print('Only using {} trajectory for each sub-task'.format(cfg.limit_num_traj))
        for tsk in cfg.tasks:
            tsk.traj_per_subtask = cfg.limit_num_traj
    if cfg.limit_num_demo > -1:
        print(
            'Only using {} demon. trajectory for each sub-task'.format(cfg.limit_num_demo))
        for tsk in cfg.tasks:
            tsk.demo_per_subtask = cfg.limit_num_demo

    print(cfg.policy)
    if 'target_obj_detector' not in cfg.policy._target_:
        print(f'Running baseline method: {cfg.policy._target_}')
        cfg.target_update_freq = -1
    workspace = W(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
