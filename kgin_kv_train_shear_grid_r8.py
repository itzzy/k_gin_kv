import argparse
import numpy as np
import yaml
import torch
import os
from utils import wandb_setup, dict2obj
from trainer_shear_grid_r8 import TrainerKInterpolator

#nohup python kgin_kv_train_vista_r8_test.py --config config_kgin_kv_vista_r8_test.yaml > log_0216_test_3.txt 2>&1 &
'''
wandb: ERROR Run initialization has timed out after 90.0 sec. 
wandb: ERROR Please refer to the documentation for additional information: https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-
Traceback (most recent call last):
  File "train.py", line 35, in <module>
    if not config['general']['debug']: wandb_setup(config)
  File "/data0/zhiyong/code/github/itzzy_git/k-gin_kv/utils/train_related.py", line 150, in wandb_setup
    run = wandb.init(project='KInterpolator', entity=args['general']['wandb_entity'], group=group, config=args)
  File "/home/zhiyongzhang/anaconda3/envs/k_gin/lib/python3.8/site-packages/wandb/sdk/wandb_init.py", line 1255, in init
    wandb._sentry.reraise(e)
  File "/home/zhiyongzhang/anaconda3/envs/k_gin/lib/python3.8/site-packages/wandb/analytics/sentry.py", line 155, in reraise
    raise exc.with_traceback(sys.exc_info()[2])
  File "/home/zhiyongzhang/anaconda3/envs/k_gin/lib/python3.8/site-packages/wandb/sdk/wandb_init.py", line 1241, in init
    return wi.init()
  File "/home/zhiyongzhang/anaconda3/envs/k_gin/lib/python3.8/site-packages/wandb/sdk/wandb_init.py", line 838, in init
    raise error
wandb.errors.CommError: Run initialization has timed out after 90.0 sec. 
Please refer to the documentation for additional information: https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-
报了上面的错，设置为离线模式
'''
# import wandb
# wandb.init(mode="offline")

# nohup python k_gin_train_r10.py --config config_r_10.yaml > log_0107_test.txt 2>&1 &
parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, help='config file (.yml) containing the hyper-parameters for inference.')
parser.add_argument('--debug', action='store_true', help='if true, model will not be logged and saved')
parser.add_argument('--seed', type=int, help='seed of torch and numpy', default=1)
parser.add_argument('--val_frequency', type=int, help='training data and weights will be saved in this frequency of epoch')

if __name__ == '__main__':

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        print(f'Using {args.config} as config file')
        config = yaml.load(f, Loader=yaml.FullLoader)

    for arg in vars(args):
        if getattr(args, arg):
            if arg in config['general']:
                print(f'Overriding {arg} from argparse')
            config['general'][arg] = getattr(args, arg)

    if not config['general']['debug']: wandb_setup(config)
    config = dict2obj(config)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{config.general.gpus}'

    config.general.debug = args.debug
    trainer = TrainerKInterpolator(config)
    trainer.run()
