import os
# os.environ["CUDA_VISIBLE_DEVICES"]='6'
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sacred import Experiment

from config_super import setup, init_environment
from constants import on_cloud
from core.base_trainer import BaseTrainer, BaseEvaluator
from core.losses import get as get_loss_obj
from data_kits import datasets
from networks import load_model
from utils_ import misc

ex = setup(
    Experiment(name="Super", save_git_info=False, base_dir="./")
)
torch.set_printoptions(precision=8)


class Evaluator(BaseEvaluator):
    def test_step(self, batch, step):
        sup_rgb = batch['sup_rgb'].cuda()
        sup_msk = batch['sup_msk'].cuda()
        qry_rgb = batch['qry_rgb'].cuda()
        qry_msk = batch['qry_msk'].cuda()
        classes = batch['cls'].cuda()
        class_name = batch['class_name']
        qry_names = batch['qry_names']

        output = self.model_DP(qry_rgb, sup_rgb, sup_msk, qry_msk, class_name,classes,qry_names)
        qry_pred = output['out']

        # Compute loss
        loss = self.loss_obj(qry_pred, qry_msk.squeeze(1))

        # Compute prediction
        qry_pred = qry_pred.argmax(dim=1).detach().cpu().numpy()
        return qry_pred, {'loss': loss.item()}


class Trainer(BaseTrainer):
    def _train_step(self, batch, step, epoch):
        sup_rgb = batch['sup_rgb'].cuda()
        sup_msk = batch['sup_msk'].cuda()
        qry_rgb = batch['qry_rgb'].cuda()
        qry_msk = batch['qry_msk'].cuda()
        classes = batch['cls'].cuda()
        class_name = batch['class_name']
        kwargs = {}
        if 'weights' in batch:
            kwargs['weight'] = batch['weights'].cuda()

        output = self.model_DP(qry_rgb, sup_rgb, sup_msk, qry_msk, class_name, classes)
        qry_msk_reshape = qry_msk.view(-1, *qry_msk.shape[-2:])
        sup_msk_reshape = sup_msk.view(-1, *sup_msk.shape[-2:])

        loss = self.loss_obj(output['out'], qry_msk_reshape, **kwargs)
        aux_loss = torch.zeros(1).to(device=loss.device)
        for i in range(len(output['aux_out'])):
            aux_loss_i = self.loss_obj(output['aux_out'][i], qry_msk_reshape, **kwargs)
            aux_loss+=aux_loss_i
        aux_loss = aux_loss /len(output['aux_out'])


        aux_loss = aux_loss * 0.8 # default 0.8
        total_loss = loss  + aux_loss

        return total_loss, loss, aux_loss

    def train_step(self, batch, step, epoch):
        self.optimizer.zero_grad()

        total_loss, loss, aux_loss= self._train_step(batch, step, epoch)
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'aux' : aux_loss.item(),
        }


@ex.main
def train(_run, _config):
    opt, logger, device = init_environment(ex, _run, _config)  # 定义参数等，参数在config.py里面更改

    ds_train, data_loader, _ = datasets.load(opt, logger, "train")  # 加载训练集
    ds_eval_online, data_loader_val, num_classes = datasets.load(opt, logger, "eval_online")  # 加载验证集-online验证
    logger.info(f'     ==> {len(ds_train)} training samples')
    logger.info(f'     ==> {len(ds_eval_online)} eval_online samples')

    model = load_model(opt, logger)
    if opt.exp_id >= 0 or opt.ckpt:
        ckpt = misc.find_snapshot(_run.run_dir.parent, opt.exp_id, opt.ckpt, afs=on_cloud)
        model.load_weights(ckpt, logger, strict=opt.strict)

    trainer = Trainer(opt, logger, device, model, data_loader, data_loader_val, _run)
    evaluator = Evaluator(opt, logger, device, trainer.model_DP, None, "EVAL_ONLINE")

    logger.info("Start training.")
    start_epoch = 1
    trainer.start_training_loop(start_epoch, evaluator, num_classes)

    logger.info(f"============ Training finished - id {_run._id} ============\n")
    if _run._id is not None:
        return test(_run, _config, _run._id, ckpt=None, strict=False, eval_after_train=True)


@ex.command(unobserved=True)
def test(_run, _config, exp_id=-1, ckpt=None, strict=True, eval_after_train=False):
    opt, logger, device = init_environment(ex, _run, _config, eval_after_train=eval_after_train)
    ds_test, data_loader, num_classes = datasets.load(opt, logger, "test")
    logger.info(f'     ==> {len(ds_test)} testing samples')

    model = load_model(opt, logger)
    if not opt.no_resume:
        model_ckpt = misc.find_snapshot(_run.run_dir.parent, exp_id, ckpt)
        logger.info(f"     ==> Try to load checkpoint from {model_ckpt}")
        model.load_weights(model_ckpt, logger, strict=strict)
        logger.info(f"     ==> Checkpoint loaded.")

    tester = Evaluator(opt, logger, device, model, None, "EVAL")

    logger.info("Start testing.")
    loss, mean_iou, binary_iou, _, _ = tester.start_eval_loop(data_loader, num_classes)

    return f"Loss: {loss:.4f}, mIoU: {mean_iou * 100:.2f}, bIoU: {binary_iou * 100:.2f}"


@ex.command(unobserved=True)
def predict(_run, _config, exp_id=-1, ckpt=None, strict=True):
    opt, logger, device = init_environment(ex, _run, _config)

    model = load_model(opt, logger)
    if not opt.no_resume:
        model_ckpt = misc.find_snapshot(_run.run_dir.parent, exp_id, ckpt)
        logger.info(f"     ==> Try to load checkpoint from {model_ckpt}")
        model.load_weights(model_ckpt, logger, strict)
        logger.info(f"     ==> Checkpoint loaded.")
    model = model.to(device)
    loss_obj = get_loss_obj(opt, logger, loss='ce')

    sup_rgb, sup_msk, qry_rgb, qry_msk, qry_ori = datasets.load_p(opt, device)
    classes = torch.LongTensor([opt.p.cls]).cuda()

    logger.info("Start predicting.")

    model.eval()
    ret_values = []
    for i in range(qry_rgb.shape[0]):
        print('Processing:', i + 1)
        qry_rgb_i = qry_rgb[i:i + 1]
        qry_msk_i = qry_msk[i:i + 1] if qry_msk is not None else None
        qry_ori_i = qry_ori[i]

        output = model(qry_rgb_i, sup_rgb, sup_msk, out_shape=qry_ori_i.shape[-3:-1])
        pred = output['out'].argmax(dim=1).detach().cpu().numpy()

        if qry_msk_i is not None:
            loss = loss_obj(output['out'], qry_msk_i).item()
            ref = qry_msk_i.cpu().numpy()
            tp = int((np.logical_and(pred == 1, ref != 255) * np.logical_and(ref == 1, ref != 255)).sum())
            fp = int((np.logical_and(pred == 1, ref != 255) * np.logical_and(ref != 1, ref != 255)).sum())
            fn = int((np.logical_and(pred != 1, ref != 255) * np.logical_and(ref == 1, ref != 255)).sum())
            mean_iou = tp / (tp + fp + fn)
            binary_iou = 0
            ret_values.append(f"Loss: {loss:.4f}, mIoU: {mean_iou * 100:.2f}, bIoU: {binary_iou * 100:.2f}")
        else:
            ret_values.append(None)

        # Save to file
        if opt.p.out:
            pred = pred[0].astype(np.uint8) * 255
            if opt.p.overlap:
                out = qry_ori_i.copy()
                out[pred == 255] = out[pred == 255] * 0.5 + np.array([255, 0, 0]) * 0.5
            else:
                out = pred

            out_dir = Path(opt.p.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = Path(opt.p.qry or opt.p.qry_rgb[i]).stem + '_pred.png'
            out_path = out_dir / out_name
            Image.fromarray(out).save(out_path)

        # Release memory
        del output
        torch.cuda.empty_cache()

    if ret_values[0] is not None:
        return '\n'.join(ret_values)


if __name__ == '__main__':
    ex.run_commandline()



