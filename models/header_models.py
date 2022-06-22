
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch import nn
import numpy as np

from architectures import encoders as encoders
from architectures.headers.simple_header import DatasetHeader, Header
from common import constants as c
from models.base.base_header_model import BaseHeaderModel


class HeaderModel(nn.Module):
    def __init__(self, encoder, header):
        super().__init__()

        self.encoder = encoder
        self.header = header

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.header(z)


class MultiHeaderModel(nn.Module):
    def __init__(self, encoder, headers):
        super().__init__()

        self.encoder = encoder
        self.headers = []
        for h in headers:
            self.headers.append(h)
        self.headers = nn.ModuleList(self.headers)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        out = ()

        for h in self.headers:
            out = out + (h(z),)
        return out
        
class CelebAMT(BaseHeaderModel):
    def __init__(self, args):
        super().__init__(args)

        # encoder and decoder
        encoder_model = getattr(encoders, args.encoder[0])
        self.encoder = encoder_model(self.z_dim, self.num_channels, self.image_size)
        headers = []
        for i in range(c.CELEB_NUM_CLASSES):
            header_class = Header(self.z_dim, 2, args.header_dim).to(self.device)
            headers.append(header_class)

        # model and optimizer
        self.model = MultiHeaderModel(encoder, headers).to(self.device)

        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.nets = [self.model]
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

    def loss_fn(self, **kwargs):
        y_true = kwargs['y_true']

        losses = {}
        for class_id in range(c.CELEB_NUM_CLASSES):
            y_pred = kwargs['y_pred{}'.format(class_id)]
            loss = torch.mean(F.cross_entropy(y_pred, y_true[:, c.CELEB_CLASSES[class_id]].long()))
            losses['loss{}'.format(class_id)] = loss

        return losses

    def accuracies(self, **kwargs):

        y_true = kwargs['y_true']

        accs = {}
        for class_id in range(c.CELEB_NUM_CLASSES):
            y_pred = kwargs['y_pred{}'.format(class_id)]
            acc = torch.mean((1.0 * (torch.argmax(y_pred, axis=1) == y_true[:, c.CELEB_CLASSES[
                                                                                   class_id]].long())).float()).cpu().detach().numpy()
            accs['acc{}'.format(class_id)] = acc

        return accs

    def train(self):
        state = np.random.RandomState(self.seed)
        self.accumulator.reinit()
        for epoch in range(self.max_epoch):
            self.net_mode(train=True)
            for x_true1, y_true1 in self.train_loader:
                x_true1 = x_true1.to(self.device)
                y_true1 = y_true1.to(self.device)
                y_preds = self.model(x_true1)

                preds_dict = {'y_pred{}'.format(class_id): y_preds[class_id] for class_id in range(c.CELEB_NUM_CLASSES)}
                preds_dict.update({'y_true': y_true1})
                loss_dict = self.loss_fn(**preds_dict)
                i = state.randint(c.CELEB_NUM_CLASSES)
                loss_total = loss_dict['loss{}'.format(i)]
                accs = self.accuracies(**preds_dict)
                loss_dict.update({'loss_total': loss_total, 'task': i})
                loss_dict.update(accs)

                for key in loss_dict:
                    self.accumulator.cumulate(key, loss_dict[key], n=x_true1.shape[0])

                self.optim_G.zero_grad()
                loss_total.backward(retain_graph=True)
                self.optim_G.step()

            averages = self.accumulator.get_average()
            self.accumulator.reinit()
            self.metric_logger.compute_metrics(self, epoch, split="train", **averages)
            self.test(self.valid_loader, iter=epoch, split="valid", save=False)
            self.save_checkpoint(epoch, ckptname='last')
            
        self.save_metric_results(self.train_output_dir, "train_results.npy")

    def test(self, data_loader, iter=None, split="test", save=True, save_preds=False):
        assert split=="test" or split=="valid", "split must be either test or valid"
        self.accumulator.reinit()
        self.net_mode(train=False)
        preds = {}
        true_vals = {}
        for x_true1, y_true1 in tqdm.tqdm(data_loader):
            x_true1 = x_true1.to(self.device)
            y_true1 = y_true1.to(self.device)
            y_preds = self.model(x_true1)

            preds_dict = {'y_pred{}'.format(class_id): y_preds[class_id] for class_id in range(c.CELEB_NUM_CLASSES)}
            preds_dict.update({'y_true': y_true1})
            loss_dict = self.loss_fn(**preds_dict)
            accs = self.accuracies(**preds_dict)
            loss_dict.update(accs)

            for key in loss_dict:
                self.accumulator.cumulate(key, loss_dict[key], n=x_true1.shape[0]) 
                
            if save_preds:
                for class_id in range(c.CELEB_NUM_CLASSES):
                    name1 = 'y_pred{}'.format(class_id)
                    name2 = 'y_true{}'.format(class_id)
                    pred = preds_dict['y_pred{}'.format(class_id)].cpu().detach().numpy()
                    true_val =  y_true1[:, class_id].cpu().detach().numpy()
                    if name1 not in preds:
                        preds[name1] = []
                    if name2 not in true_vals:
                        true_vals[name2] = []
                    preds[name1].append(pred)
                    true_vals[name2].append(true_val)

        averages = self.accumulator.get_average()
        self.accumulator.reinit()
        self.metric_logger.compute_metrics(self, iter, split, **averages)
        if save:
            self.save_metric_results(self.test_output_dir, "{}_results.npy".format(split))
        if save_preds:
            for class_id in range(c.CELEB_NUM_CLASSES):  
                name1 = 'y_pred{}'.format(class_id)
                name2 = 'y_true{}'.format(class_id)
                preds[name1] = np.concatenate(preds[name1])
                true_vals[name2] = np.concatenate(true_vals[name2])
            return preds, true_vals            

class DspritesMT(BaseHeaderModel):
    def __init__(self, args):
        super().__init__(args)
        self.n_task_headers = args.n_task_headers 
        encoder_model = getattr(encoders, args.encoder[0])
        self.encoder = encoder_model(self.z_dim, self.num_channels, self.image_size)
        headers = []
        for i in range(args.n_task_headers):
            if args.header_type == "DatasetHeader":
                headers.append(DatasetHeader(self.z_dim, 1, 300).to(self.device))
            else:
                headers.append(Header(self.z_dim, 1, args.header_dim).to(self.device))

        # model and optimizer
        self.model = MultiHeaderModel(self.encoder, headers).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        self.single_task_batch = args.single_task_batch
        self.losses_weights = args.losses_weights

        # Check if weights define a proper prob distribution
        if self.single_task_batch:
            assert sum(self.losses_weights) == 1.

        # nets
        self.nets = [self.model]
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

    def loss_fn(self, **kwargs):

        y_true = kwargs['y_true']
        losses = []
        for i in range(self.n_task_headers):
            y_pred = kwargs['y_pred{}'.format(i)]
            losses.append(F.mse_loss(y_pred.view(-1), y_true[:, i], reduction='mean'))
        return losses

    def train(self):
        state = np.random.RandomState(self.seed)
        self.accumulator.reinit()
        for epoch in range(self.max_epoch):
            self.net_mode(train=True)
            for x_true1, y_true1 in tqdm.tqdm(self.train_loader):
                x_true1 = x_true1.to(self.device)
                y_true1 = y_true1.to(self.device)
                y_preds = self.model(x_true1)

                preds_dict = {'y_pred{}'.format(class_id): y_preds[class_id] for class_id in range(self.n_task_headers)}
                preds_dict.update({'y_true': y_true1})
                losses_tuple = self.loss_fn(**preds_dict)

                if self.single_task_batch:
                    i = state.choice(self.n_task_headers, p=self.losses_weights)
                    loss_total = losses_tuple[i]
                else:
                    loss_total = sum(loss * w for loss, w in zip(losses_tuple, self.losses_weights))

                loss_dict = {'loss{}'.format(i): losses_tuple[i] for i in range(self.n_task_headers)}
                if self.single_task_batch:
                    loss_dict['task'] = i

                for key in loss_dict:
                    self.accumulator.cumulate(key, loss_dict[key], n=x_true1.shape[0])

                self.optim_G.zero_grad()
                loss_total.backward(retain_graph=False)
                self.optim_G.step()

            averages = self.accumulator.get_average()
            self.accumulator.reinit()
            self.metric_logger.compute_metrics(self, epoch, split="train", **averages)
            self.test(self.valid_loader, iter=epoch, split="valid", save=False)
            self.save_checkpoint(epoch, ckptname='last')

        self.save_metric_results(self.train_output_dir, "train_results.npy")

    def test(self, data_loader, iter=None, split="test", save=True, save_preds=False):
        assert split=="test" or split=="valid", "split must be either test or valid"
        self.accumulator.reinit()
        self.net_mode(train=False)
        preds = {}
        codes = []
        true_vals = {}
        for x_true1, y_true1 in tqdm.tqdm(data_loader):
            x_true1 = x_true1.to(self.device)
            y_true1 = y_true1.to(self.device)
            y_preds = self.model(x_true1)
            encoded = self.model.encode(x_true1)
            codes.append(encoded.cpu().detach().numpy())

            preds_dict = {'y_pred{}'.format(class_id): y_preds[class_id] for class_id in range(self.n_task_headers)}
            preds_dict.update({'y_true': y_true1})
            losses_tuple = self.loss_fn(**preds_dict)
            
            loss_dict = {'loss{}'.format(i): losses_tuple[i] for i in range(self.n_task_headers)}

            
            for key in loss_dict:
                self.accumulator.cumulate(key, loss_dict[key], n=x_true1.shape[0])
                
            if save_preds:
                for class_id in range(self.n_task_headers):
                    name1 = 'y_pred{}'.format(class_id)
                    name2 = 'y_true{}'.format(class_id)
                    pred = preds_dict['y_pred{}'.format(class_id)].cpu().detach().numpy()
                    true_val =  y_true1[:, class_id].cpu().detach().numpy()
                    if name1 not in preds:
                        preds[name1] = []
                    if name2 not in true_vals:
                        true_vals[name2] = []
                    preds[name1].append(pred)
                    true_vals[name2].append(true_val)
                                       
                        
        averages = self.accumulator.get_average()
        self.accumulator.reinit()
        self.metric_logger.compute_metrics(self, iter, split, **averages)
        if save:
            self.save_metric_results(self.test_output_dir, "{}_results.npy".format(split))
        if save_preds:
            np.save("{}/{}_encodings.npy".format(self.test_output_dir, split),codes)
            for class_id in range(self.n_task_headers):  
                name1 = 'y_pred{}'.format(class_id)
                name2 = 'y_true{}'.format(class_id)
                preds[name1] = np.concatenate(preds[name1])
                true_vals[name2] = np.concatenate(true_vals[name2])
            return preds, true_vals
        else:
            return None, None

        
class FullRegression(BaseHeaderModel):
    def __init__(self, args):
        super().__init__(args)
        self.n_task_headers = args.n_task_headers 
        encoder_model = getattr(encoders, args.encoder[0])
        self.encoder = encoder_model(self.z_dim, self.num_channels, self.image_size)
        header=None
       
        if args.header_type == "DatasetHeader":
            header=DatasetHeader(self.z_dim, args.n_task_headers, 300).to(self.device)
        else:
            header=Header(self.z_dim, args.n_task_headers, args.header_dim).to(self.device)

        # model and optimizer
        self.model = HeaderModel(self.encoder, header).to(self.device)
        self.optim_G = optim.Adam(self.model.parameters(), lr=self.lr_G, betas=(self.beta1, self.beta2))

        # nets
        self.nets = [self.model]
        self.net_dict = {
            'G': self.model
        }
        self.optim_dict = {
            'optim_G': self.optim_G,
        }

    def loss_fn(self, **kwargs):

        y_true = kwargs['y_true']
        losses = []
        for i in range(self.n_task_headers):
            y_pred = kwargs['y_preds'][:,i]
            losses.append(F.mse_loss(y_pred, y_true[:, i], reduction='mean'))
            
        y_pred = kwargs['y_preds']
        losses.append(F.mse_loss(y_pred, y_true, reduction='mean'))
        return losses

    def train(self):
        state = np.random.RandomState(self.seed)
        self.accumulator.reinit()
        for epoch in range(self.max_epoch):
            self.net_mode(train=True)
            for x_true1, y_true1 in tqdm.tqdm(self.train_loader):
                x_true1 = x_true1.to(self.device)
                y_true1 = y_true1.to(self.device)
                y_preds = self.model(x_true1)

                preds_dict = {'y_preds': y_preds}
                preds_dict.update({'y_true': y_true1})
                losses_tuple = self.loss_fn(**preds_dict)

                loss_total = losses_tuple[-1]

                loss_dict = {'loss{}'.format(i): losses_tuple[i] for i in range(self.n_task_headers)}
                loss_dict["loss_total"] = loss_total
                
                for key in loss_dict:
                    self.accumulator.cumulate(key, loss_dict[key], n=x_true1.shape[0])

                self.optim_G.zero_grad()
                loss_total.backward(retain_graph=False)
                self.optim_G.step()

            averages = self.accumulator.get_average()
            self.accumulator.reinit()
            self.metric_logger.compute_metrics(self, epoch, split="train", **averages)
            self.test(self.valid_loader, iter=epoch, split="valid", save=False)
            self.save_checkpoint(epoch, ckptname='last')

        self.save_metric_results(self.train_output_dir, "train_results.npy")

    def test(self, data_loader, iter=None, split="test", save=True, save_preds=False):
        assert split=="test" or split=="valid", "split must be either test or valid"
        self.accumulator.reinit()
        self.net_mode(train=False)
        preds = {}
        codes = []
        true_vals = {}
        for x_true1, y_true1 in tqdm.tqdm(data_loader):
            x_true1 = x_true1.to(self.device)
            y_true1 = y_true1.to(self.device)
            y_preds = self.model(x_true1)
            encoded = self.model.encode(x_true1)
            codes.append(encoded.cpu().detach().numpy())

            preds_dict = {'y_preds': y_preds}
            preds_dict.update({'y_true': y_true1})
            losses_tuple = self.loss_fn(**preds_dict)
            
            loss_total = losses_tuple[-1]
            
            loss_dict = {'loss{}'.format(i): losses_tuple[i] for i in range(self.n_task_headers)}
            loss_dict["loss_total"] = loss_total
            
            for key in loss_dict:
                self.accumulator.cumulate(key, loss_dict[key], n=x_true1.shape[0])
                
            if save_preds:
                for class_id in range(self.n_task_headers):
                    name1 = 'y_pred{}'.format(class_id)
                    name2 = 'y_true{}'.format(class_id)
                    pred = preds_dict['y_preds'][:, class_id].cpu().detach().numpy()
                    true_val =  y_true1[:, class_id].cpu().detach().numpy()
                    if name1 not in preds:
                        preds[name1] = []
                    if name2 not in true_vals:
                        true_vals[name2] = []
                    preds[name1].append(pred)
                    true_vals[name2].append(true_val)
                                       
                        
        averages = self.accumulator.get_average()
        self.accumulator.reinit()
        self.metric_logger.compute_metrics(self, iter, split, **averages)
        if save:
            self.save_metric_results(self.test_output_dir, "{}_results.npy".format(split))
        if save_preds:
            np.save("{}/{}_encodings.npy".format(self.test_output_dir, split),codes)
            for class_id in range(self.n_task_headers):  
                name1 = 'y_pred{}'.format(class_id)
                name2 = 'y_true{}'.format(class_id)
                preds[name1] = np.concatenate(preds[name1])
                true_vals[name2] = np.concatenate(true_vals[name2])
            return preds, true_vals
        else:
            return None, None