#
import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from progress.bar import Bar
from sklearn.cluster import KMeans
from src.SEDR_model import SEDR


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn


def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask

    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


class SEDR_Train:
    def __init__(self, node_X, graph_dict, params):
        self.params = params
        self.device = params.device
        self.epochs = params.epochs
        self.node_X = torch.FloatTensor(node_X.copy()).to(self.device)
        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)
        self.norm_value = graph_dict["norm_value"]
        if params.using_mask is True:
            self.adj_mask = graph_dict["adj_mask"].to(self.device)
        else:
            self.adj_mask = None

        self.model = SEDR(self.params.cell_feat_dim, self.params).to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                          lr=self.params.gcn_lr, weight_decay=self.params.gcn_decay)

    def train_without_dec(self):
        self.model.train()
        bar = Bar('GNN model train without DEC: ', max=self.epochs)
        bar.check_tty = False
        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, _, feat_x, _ = self.model(self.node_X, self.adj_norm)

            loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,
                                logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value, mask=self.adj_label)
            loss_rec = reconstruction_loss(de_feat, self.node_X)
            loss = self.params.feat_w * loss_rec + self.params.gcn_w * loss_gcn
            loss.backward()
            self.optimizer.step()

            end_time = time.time()
            batch_time = end_time - start_time
            bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch + 1, self.epochs,
                                        batch_time=batch_time * (self.epochs - epoch) / 60, loss=loss.item())
            bar.next()
        bar.finish()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z = self.model(self.node_X, self.adj_norm)
        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()
        return latent_z, q, feat_x, gnn_z

    def train_with_dec(self):
        # initialize cluster parameter
        self.train_without_dec()
        kmeans = KMeans(n_clusters=self.params.dec_cluster_n, n_init=self.params.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()

        bar = Bar('Training Graph Net with DEC loss: ', max=self.epochs)
        bar.check_tty = False
        for epoch_id in range(self.epochs):
            # DEC clustering update
            if epoch_id % self.params.dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < self.params.dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', self.params.dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # training model
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, out_q, _, _ = self.model(self.node_X, self.adj_norm)
            loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,
                                logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value, mask=self.adj_label)
            loss_rec = reconstruction_loss(de_feat, self.node_X)
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss = self.params.gcn_w * loss_gcn + self.params.dec_kl_w * loss_kl + self.params.feat_w * loss_rec
            loss.backward()
            self.optimizer.step()

            bar_str = '{} / {} | Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch_id + 1, self.epochs, loss=loss.item())
            bar.next()
        bar.finish()


