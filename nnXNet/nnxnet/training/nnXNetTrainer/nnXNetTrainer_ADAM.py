"""
ADAM Dataset Trainer — Binary Aneurysm Segmentation on TOF-MRA

Based on BraveCoWCoW's nnXNet framework, adapted for:
  - Single modality (TOF-MRA only, no modality classification head)
  - Binary segmentation (aneurysm vs background, no 26-class anatomy labels)
  - No CSV metadata dependency
  - 32GB VRAM friendly (batch_size=1, deep_supervision=OFF)

Usage:
  nnXNet_train Dataset001_ADAM 3d_fullres 0 -tr nnXNetTrainer_ADAM
"""

import os
import numpy as np
import torch
from torch import autocast
from typing import Union, Tuple, List
from torch.nn.parallel import DistributedDataParallel as DDP

from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer
from nnxnet.training.nnXNetTrainer.variants.network_architecture.ResEncoderUNet_two_seg import (
    ResEncoderUNet_two_seg,
)
from nnxnet.training.loss.compound_losses import DC_and_CE_loss
from nnxnet.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnxnet.training.loss.deep_supervision import DeepSupervisionWrapper
from nnxnet.utilities.helpers import dummy_context
from nnxnet.utilities.collate_outputs import collate_outputs
from nnxnet.utilities.label_handling.label_handling import determine_num_input_channels


class nnXNetTrainer_ADAM(nnXNetTrainer):
    """
    ADAM-adapted trainer for binary intracranial aneurysm segmentation.

    Key differences from RSNA trainers:
      - Uses base nnXNetDataLoader3D (no CSV metadata)
      - ResEncoderUNet_two_seg with 2-class outputs
      - No classification losses, no modality head
      - Deep supervision OFF by default (memory saving)
      - batch_size forced to 1 for 32GB VRAM
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)

        # ---- Memory-saving defaults for 32GB VRAM ----
        self.enable_deep_supervision = False
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.save_every = 50

        # ---- ADAM segmentation heads ----
        # seg_index_1: binary aneurysm segmentation
        # seg_index_2: binary aneurysm segmentation (dual decoder)
        self.seg_index_1 = [[1]]  # class 1 = aneurysm
        self.seg_index_2 = [[1]]
        self.seg_ce_class_weights_1 = [10]  # high weight for aneurysm
        self.seg_ce_class_weights_2 = [10]

    # ------------------------------------------------------------------
    # Override: ResEncoderUNet_two_seg stores deep_supervision at root,
    # not at .decoder.deep_supervision like standard nnU-Net
    # ------------------------------------------------------------------
    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, torch.nn.Module):
            # torch.compile wraps in OptimizedModule
            from torch._dynamo import OptimizedModule
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
        mod.deep_supervision = enabled

        self.print_to_log_file(
            "nnXNetTrainer_ADAM initialized: "
            f"deep_supervision={self.enable_deep_supervision}, "
            f"batch_size={self.configuration_manager.batch_size}"
        )

    # ------------------------------------------------------------------
    # Network architecture
    # ------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels_1: int = 2,
        num_output_channels_2: int = 2,
        enable_deep_supervision: bool = False,
    ) -> torch.nn.Module:
        """
        Build ResEncoderUNet_two_seg for binary aneurysm segmentation.

        Uses the same architecture as BraveCoWCoW but with:
          - 2 output classes per head (bg + aneurysm)
          - No classification/modality heads
          - Reduced feature dims optionally
        """
        network = ResEncoderUNet_two_seg(
            in_channels=num_input_channels,
            out_channels_1=num_output_channels_1,
            out_channels_2=num_output_channels_2,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            kernel_sizes=[(3, 3, 1), (3, 3, 3), (3, 3, 3),
                           (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2),
                      (2, 2, 2), (2, 2, 2), (2, 2, 2)],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            deep_supervision=enable_deep_supervision,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05, "affine": True},
            conv_op=torch.nn.Conv3d,
            conv_bias=True,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
        )
        return network

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                len(self.seg_index_1) + 1,  # 2 (bg + aneurysm)
                len(self.seg_index_2) + 1,  # 2
                self.enable_deep_supervision,
            ).to(self.device)

            # torch.compile disabled: incompatible with anisotropic patch sizes
            # (ResEncoderUNet residual connections have mismatched shapes under
            #  compile's fake tensor tracing with do_dummy_2d_data_aug=True)
            if self._do_i_compile():
                self.print_to_log_file(
                    'ADAM trainer: skipping torch.compile '
                    '(incompatible with anisotropic patches)')

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.seg_loss_1, self.seg_loss_2 = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "Trainer already initialized. This should not happen.")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _build_loss(self):
        """DC + CE loss for binary segmentation, one per decoder head."""
        seg_ce_class_weights_1_gpu = torch.tensor(
            [1] + self.seg_ce_class_weights_1, dtype=torch.float32
        ).to(self.device)
        seg_ce_class_weights_2_gpu = torch.tensor(
            [1] + self.seg_ce_class_weights_2, dtype=torch.float32
        ).to(self.device)

        seg_loss_1 = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'weight': seg_ce_class_weights_1_gpu},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        seg_loss_2 = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'weight': seg_ce_class_weights_2_gpu},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))]
            )
            weights[-1] = 0
            weights = weights / weights.sum()
            seg_loss_1 = DeepSupervisionWrapper(seg_loss_1, weights)
            seg_loss_2 = DeepSupervisionWrapper(seg_loss_2, weights)

        return seg_loss_1, seg_loss_2

    # ------------------------------------------------------------------
    # Train step (simplified — no classification)
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) \
                if self.device.type == 'cuda' else dummy_context():
            output_1, output_2 = self.network(data)
            l_1 = self.seg_loss_1(output_1, target)
            l_2 = self.seg_loss_2(output_2, target)
            l = l_1 + l_2

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # Validation step (simplified — no classification)
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        keys = batch['keys']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        validation_dict = {}

        with autocast(self.device.type, enabled=True) \
                if self.device.type == 'cuda' else dummy_context():
            output_1, output_2 = self.network(data)
            l_1 = self.seg_loss_1(output_1, target)
            l_2 = self.seg_loss_2(output_2, target)
            l = l_1 + l_2

        # ---- Compute Dice metrics ----
        with torch.no_grad():
            if self.enable_deep_supervision:
                output_1_eval = output_1[0]
                output_2_eval = output_2[0]
                target_eval_1 = target[0] if isinstance(target, list) else target
                target_eval_2 = target_eval_1
            else:
                output_1_eval = output_1
                output_2_eval = output_2
                target_eval_1 = target
                target_eval_2 = target

            axes = [0] + list(range(2, output_1_eval.ndim))

            for name_suffix, output, tgt in [
                ('1', output_1_eval, target_eval_1),
                ('2', output_2_eval, target_eval_2),
            ]:
                output_seg = output.argmax(1)[:, None]
                predicted_onehot = torch.zeros(
                    output.shape, device=output.device, dtype=torch.float32)
                predicted_onehot.scatter_(1, output_seg, 1)

                tp, fp, fn, _ = get_tp_fp_fn_tn(
                    predicted_onehot, tgt, axes=axes, mask=None)
                validation_dict[f'tp_hard_{name_suffix}'] = \
                    tp.detach().cpu().numpy()[1:]
                validation_dict[f'fp_hard_{name_suffix}'] = \
                    fp.detach().cpu().numpy()[1:]
                validation_dict[f'fn_hard_{name_suffix}'] = \
                    fn.detach().cpu().numpy()[1:]

        validation_dict['loss'] = l.detach().cpu().numpy()
        validation_dict['keys'] = keys
        return validation_dict

    # ------------------------------------------------------------------
    # Epoch-end callbacks
    # ------------------------------------------------------------------
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        if self.is_ddp:
            losses_tr = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        # Aggregate Dice metrics
        for suffix in ['1', '2']:
            tp = np.sum(outputs_collated[f'tp_hard_{suffix}'], 0)
            fp = np.sum(outputs_collated[f'fp_hard_{suffix}'], 0)
            fn = np.sum(outputs_collated[f'fn_hard_{suffix}'], 0)

            dice_per_class = [
                2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0
                for i, j, k in zip(tp, fp, fn)
            ]
            mean_dice = np.nanmean(dice_per_class)

            log_key = f'mean_fg_dice_{suffix}'
            if log_key not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging[log_key] = list()
            self.logger.log(log_key, mean_dice, self.current_epoch)

            self.print_to_log_file(
                f"Mean Dice head_{suffix}: {mean_dice:.4f}  "
                f"per_class: {[round(x, 4) for x in dice_per_class]}"
            )

        # Validation loss
        if self.is_ddp:
            losses_val = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(
                losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        """Log metrics and save checkpoints."""
        import time as time_module
        self.logger.log('epoch_end_timestamps',
                        time_module.time(), self.current_epoch)

        self.print_to_log_file(
            'train_loss',
            np.round(self.logger.my_fantastic_logging['train_losses'][-1],
                     decimals=4))
        self.print_to_log_file(
            'val_loss',
            np.round(self.logger.my_fantastic_logging['val_losses'][-1],
                     decimals=4))

        for suffix in ['1', '2']:
            log_key = f'mean_fg_dice_{suffix}'
            if log_key in self.logger.my_fantastic_logging:
                self.print_to_log_file(
                    f'val_dice_head_{suffix}',
                    np.round(self.logger.my_fantastic_logging[log_key][-1],
                             decimals=4))

        epoch_time = (
            self.logger.my_fantastic_logging['epoch_end_timestamps'][-1]
            - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        )
        self.print_to_log_file(f"Epoch time: {np.round(epoch_time, decimals=2)} s")

        # Save checkpoints
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 \
                and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(
                os.path.join(self.output_folder, 'checkpoint_latest.pth'))

        # Track best based on mean Dice of head 1
        if 'mean_fg_dice_1' in self.logger.my_fantastic_logging:
            current_dice = self.logger.my_fantastic_logging[
                'mean_fg_dice_1'][-1]
            if self._best_ema is None or current_dice > self._best_ema:
                self._best_ema = current_dice
                self.print_to_log_file(
                    f"New best Dice: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(
                    os.path.join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    # ------------------------------------------------------------------
    # Batch size override — force 1 for 32GB VRAM safety
    # ------------------------------------------------------------------
    def _set_batch_size_and_oversample(self):
        """Force batch_size=1 regardless of plans configuration."""
        if not self.is_ddp:
            self.batch_size = 1
            self.oversample_foreground_percent = 0.33
            self.print_to_log_file(
                f"ADAM trainer: forcing batch_size={self.batch_size}")
        else:
            super()._set_batch_size_and_oversample()
