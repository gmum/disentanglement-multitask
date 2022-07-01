import logging
import os

import torch


class ModelSaver:
    def __init__(self, args, num_batches, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_save_iter = args.ckpt_save_iter if args.ckpt_save_iter else num_batches
        if args.all_iter:
            self.ckpt_save_iter = args.all_iter
    

    def save_checkpoint(self, iter, net_dict, optim_dict, ckptname='last'):
        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        model_states = dict()
        optim_states = dict()

        # neural models
        for key, value in net_dict.items():
            if isinstance(value, dict):
                list_state_dicts = {}
                for sub_key, net in value.items():
                    list_state_dicts.update({sub_key: net.state_dict()})
                model_states.update({key: list_state_dicts})
            else:
                model_states.update({key: value.state_dict()})

        # optimizers' states
        for key, value in optim_dict.items():
            if isinstance(value, dict):
                list_state_dicts = {}
                for sub_key, net in value.items():
                    list_state_dicts.update({sub_key: net.state_dict()})
                optim_states.update({key: list_state_dicts})
            else:
                optim_states.update({key: value.state_dict()})

        # wrap up everything in a dict
        states = {'iter': iter + 1,  # to avoid saving right after loading
                  'model_states': model_states,
                  'optim_states': optim_states}

        # make sure KeyboardInterrupt exceptions don't mess up the model saving process
        while True:
            try:
                with open(filepath, 'wb+') as f:
                    torch.save(states, f)
                break
            except KeyboardInterrupt:
                pass
        logging.info("saved checkpoint '{}' @ iter:{}".format(os.path.join(os.getcwd(), filepath), iter))
