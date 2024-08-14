import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dropblock import DropBlock2D
from torch.hub import download_url_to_file
from constants import pretrained_weights, model_urls
from core.losses import get as get_loss
from utils_.misc import interpb, interpn
from networks import vit_utils, vit
import ot

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat


        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        adj图邻接矩阵，维度[N,N]非零即一
        h.shape: (N, in_features), self.W.shape:(in_features,out_features)
        Wh.shape: (N, out_features)
        '''
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads == 1, "currently only implement num_heads==1"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.drop_prob = 0.1

    def forward(self, q, k, v, supp_mask=None, OT=None,supp_img=None,qry_names=None ):

        # supp_mask:[B,S,N]  --->part: [B,S,G,N] [B,1,G,N]
        Q = q.shape[1]  #B,c,5
        B, S, N, C = k.size()  # [B,S,C,N]
        q = q.view(B, -1, C)  # [B,S*G,C]      #[B,G,C]
        k = k.view(B, -1, C)  # [B,S*N,C]      #[B,S*N,C]
        v = v.view(B, -1, C)  # [B,S*N,C]      #[B,S*N,C]
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, S*G, S*N]
        # for j in range(attn.shape[0]):
        #     attn_j = attn[j]
        #     U,S,V=torch.svd(attn_j)
        #     print(U.shape,S,V.shape)
        if supp_mask is not None:
            supp_mask = supp_mask.view(B, -1)  # [B,S*N]
            supp_mask = (~supp_mask).unsqueeze(1).float()  # [B,1,S*N]
            supp_mask = supp_mask * -10000.0
            attn = attn + supp_mask  # [B,S*G,S*N]

        attn = attn.softmax(dim=-1)  # B,G,N
        attn = self.attn_drop(attn)  # [B,S*G,S*N]
        if OT:
            attn_list= []
            for i in range(attn.shape[0]):
                attn_i = attn[i]  # G,N
                attn_fg = torch.masked_select(attn_i, attn_i > 0).view(attn_i.shape[0], -1)  # G,K
                cost = (1-attn_fg).detach().cpu()

                # print(cost)
                r, c = attn_fg.size()
                attn_fg_new=ot.sinkhorn(np.ones(r) / r, np.ones(c) / c,np.array(cost),0.5)
                # attn_fg_new=ot.sinkhorn(np.ones(r) / r, np.ones(c) / c,np.array(cost),0.05)
                # print(attn_fg_new)
                # attn_fg_new,_ = self.compute_optimal_transport(np.array(cost), np.ones(r) / r, np.ones(c) / c, 0.05)
                attn_fg_new = torch.Tensor(attn_fg_new).cuda()
                attn_i_new = torch.zeros_like(attn_i)
                attn_i_new[attn_i>0] = attn_fg_new.view(-1)
                # attn_i = torch.where(attn_i>0,attn_fg_new,attn_i)
                # attn_list.append(attn_i)    # wrong
                attn_list.append(attn_i_new)  #ture
            attn = torch.stack(attn_list,dim=0)
            # if supp_img is not None:
            #     for i in range(attn.shape[1]):
            #         refine_map = attn[:,i]
            #         refine_map = refine_map.view(B, int(math.sqrt(N)),
            #                                      int(math.sqrt(N)))  # [B,h,w]
            #         q_map = refine_map.cpu()
            #         x_test = supp_img[:, 0].cpu()
            #         amap = cv2.cvtColor(q_map[0].squeeze(0).detach().numpy(), cv2.COLOR_RGB2BGR)
            #         new_map = cv2.resize(amap, x_test.shape[-2:])
            #         normed_mask = new_map / np.max(new_map)
            #         normed_mask = np.uint8(255 * normed_mask)
            #         normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
            #         img = np.transpose(x_test[0].detach().numpy(), (1, 2, 0))
            #         normed_mask = cv2.addWeighted(img, 1, normed_mask, 0.1, 0, dtype=cv2.CV_32F)
            #
            #         path = './vis/vis_4/'
            #         if not os.path.exists(path): os.makedirs(path)
            #         io.imsave(path + qry_names[0][0] + '_%d' % i + '.jpg',
            #                   cv2.cvtColor(normed_mask, cv2.COLOR_BGR2RGB))
        else:
            pass

        x = (attn @ v)  # [B,S*N,C]
        x = self.proj(x)
        x = self.proj_drop(x)  # [B,S*G,C]
        return x

class Mask_Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads == 1, "currently only implement num_heads==1"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.drop_prob = 0.1

    def forward(self, q, k, v, supp_mask=None):

        Q = q.shape[1]  #B,c,5
        B, S, N, C = k.size()  # [B,S,C,N]
        q = q.view(B, -1, C)  # [B,S*G,C]      #[B,G,C]
        k = k.view(B, -1, C)  # [B,S*N,C]      #[B,S*N,C]
        v = v.view(B, -1, C)  # [B,S*N,C]      #[B,S*N,C]
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B,5,N
        # for j in range(attn.shape[0]):
        #     attn_j = attn[j]
        #     U,S,V=torch.svd(attn_j)
        #     print(U.shape,S,V.shape)
        if supp_mask is not None:   # B,5,N
            supp_mask = (~supp_mask).float()  # [B,5,N]
            supp_mask = supp_mask * -10000.0
            attn = attn + supp_mask  # [B,S*G,S*N]

        attn = attn.softmax(dim=-1)  # B,G,N
        attn = self.attn_drop(attn)  # [B,S*G,S*N]
        x = (attn @ v)  # [B,S*N,C]
        x = self.proj(x)
        x = self.proj_drop(x)  # [B,S*G,C]
        return x

class Residual(nn.Module):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = interpb(x, (h * self.up, w * self.up))
        x = x_up + self.layers(x)
        return x


class agmtr(nn.Module):
    def __init__(self, opt, logger):
        super(agmtr, self).__init__()
        self.opt = opt
        self.logger = logger
        self.shot = opt.shot
        self.drop_dim = opt.drop_dim
        self.drop_rate = opt.drop_rate
        self.drop2d_kwargs = {'drop_prob': opt.drop_rate, 'block_size': opt.block_size}

        # Check existence.
        pretrained = self.get_or_download_pretrained(opt.backbone, opt.tqdm)

        # Main model
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', vit.vit_model(opt.backbone,
                                                      opt.height,
                                                      pretrained=pretrained,
                                                      num_classes=0,
                                                      opt=opt,
                                                      logger=logger))
        ]))

        embed_dim = vit.vit_factory[opt.backbone]['embed_dim']
        self.fg_sampler = np.random.RandomState(1289)
        # self.CLIP_model, preprocess = clip.load('ViT-B/16')
        self.purifier = self.build_upsampler(embed_dim)  # 下采样
        self.__class__.__name__ = f"FPTrans/{opt.backbone}"
        self.filter = opt.filter
        self.afha = nn.Parameter(torch.FloatTensor([1.0]))
        self.beta = nn.Parameter(torch.FloatTensor([0.4])) #0.5

        # Define pair-wise loss
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.LayerNorm=norm_layer(embed_dim)
        self.LayerNorm_GAT=norm_layer(embed_dim)
        self.LayerNorm_mlp=norm_layer(embed_dim)
        self.LayerNorm1=norm_layer(embed_dim)
        self.agg_depth = opt.agg_depth
        self.LayerNorm1_1=nn.Sequential(*[norm_layer(embed_dim) for i in range(self.agg_depth) ])
        # self.LayerNorm1_1=norm_layer(embed_dim)
        self.LayerNorm1_2=norm_layer(embed_dim)
        self.LayerNorm_mlp1=norm_layer(embed_dim)
        self.LayerNorm_mlp1_1=nn.Sequential(*[norm_layer(embed_dim) for i in range(self.agg_depth) ])
        # self.LayerNorm_mlp1_1=norm_layer(embed_dim)
        self.LayerNorm_mlp1_2=norm_layer(embed_dim)
        self.LayerNorm2=norm_layer(embed_dim)
        self.LayerNorm3=norm_layer(embed_dim)
        self.LayerNorm_mlp2=norm_layer(embed_dim)
        self.GraphAttentionLayer = GraphAttentionLayer(embed_dim,embed_dim,0.,0.1)
        self.Attention = Attention(embed_dim)
        self.CrossAttention =CrossAttention(embed_dim, attn_drop=0., proj_drop=0.1)
        self.CrossAttention1 =CrossAttention(embed_dim, attn_drop=0., proj_drop=0.1)
        self.Mask_Attention =Mask_Attention(embed_dim, attn_drop=0., proj_drop=0.1)

        self.Mask_Attention1 =nn.Sequential(*[Mask_Attention(embed_dim, attn_drop=0., proj_drop=0.1) for i in range(self.agg_depth) ])
        # self.Mask_Attention1 =Mask_Attention(embed_dim, attn_drop=0., proj_drop=0.1)
        self.Mask_Attention2 =Mask_Attention(embed_dim, attn_drop=0., proj_drop=0.1)
        self.MLP= vit_utils.Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0.0)
        self.MLP1= vit_utils.Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0.0)
        self.MLP3= vit_utils.Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0.0)

        self.MLP1_1 =nn.Sequential(*[vit_utils.Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0.0) for i in range(self.agg_depth) ])
        self.MLP1_2= vit_utils.Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0.0)
        self.MLP2= vit_utils.Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 4), act_layer=nn.GELU, drop=0.0)
        self.pairwise_loss = get_loss(opt, logger, loss='pairwise')  # 成对损失
        self.cls = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(embed_dim, 2, kernel_size=1)
        )
        self.part_tokens = nn.Parameter(torch.zeros(1, opt.fg_num, embed_dim))
        nn.init.normal_(self.part_tokens.permute(2, 0, 1), std=opt.pt_std)
        # Background sampler
        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} created")

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        ))

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        feature_q = feature_q.squeeze(1)  # bs,c,h,w
        fg_proto = fg_proto.permute(0, 2, 1).unsqueeze(2)  # bs,c,1,1
        bg_proto = bg_proto.view(bg_proto.shape[0], bg_proto.shape[1], -1, 1, 1)  # bs,K*s,c,1,1
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)  # bs,1,h,w
        similarity_bg = []
        for i in range(bg_proto.shape[1]):
            similarity_bg.append(F.cosine_similarity(feature_q, bg_proto[:, i], dim=1).unsqueeze(1))
        similarity_bg = torch.cat((similarity_bg), dim=1)
        out = torch.cat((similarity_fg[:, None, ...], similarity_bg), dim=1) * 10.0
        return out

    def forward(self, x, s_x, s_y, y=None, class_name=None,classes=None,qry_names=None, out_shape=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            [B, C, H, W], query image
        s_x: torch.Tensor
            [B, S, C, H, W], support image
        s_y: torch.Tensor
            [B, S, H, W], support mask
        y: torch.Tensor
            [B, 1, H, W], query mask, used for calculating the pair-wise loss
        out_shape: list
            The shape of the output predictions. If not provided, it is default
            to the last two dimensions of `y`. If `y` is also not provided, it is
            default to the [opt.height, opt.width].

        Returns
        -------
        output: dict
            'out': torch.Tensor
                logits that predicted by feature proxies
            'out_prompt': torch.Tensor
                logits that predicted by prompt proxies
            'loss_pair': float
                pair-wise loss
        """
        s_x, unlabeled_x = s_x.split([self.opt.shot,self.opt.unlabeled],dim=1)
        s_y, unlabeled_y = s_y.split([self.opt.shot,self.opt.unlabeled],dim=1)
        B, S, C, H, W = s_x.size()  # S represents the number of support images
        img_cat = torch.cat((s_x, x.view(B, 1, C, H, W)), dim=1).view(B * (S + 1), C, H, W)  # 将支持图像与查询图像concat到bs纬度

        # # Forward
        unlabeled_cat = unlabeled_x.reshape(-1,C,H,W)
        # img_cat = (img_cat, s_y, text_token)
        backbone_out = self.encoder(img_cat)  # B*(S+1),C,h0,w0/// B,C,1,1

        features = self.purifier(backbone_out['out'])  # [B(S+1), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S + 1, c, h, w)  # [B, S+1, c, h, w]
        sup_fts, qry_fts = features.split([S, 1], dim=1)  # [B, S, c, h, w] / [B, 1, c, h, w]
        sup_mask = interpn(s_y.reshape(B * S, 1, H, W), (h, w))  # [BS, 1, h, w]

        ### AAD
        unlabeled_out = self.encoder(unlabeled_cat)  # B*un,C,h0,w0
        unlabeled_features = self.purifier(unlabeled_out['out'])  # [B*un, c, h, w]
        unlabeled_y = interpn(unlabeled_y.reshape(-1, 1, H, W), (h, w))  # [B*un, 1, h, w]
        unlabeled_proto = self.unlabeled_proto(unlabeled_features,unlabeled_y)  # Bs, un*100, c

        ## GAT
        # unlabeled_sim = F.cosine_similarity(unlabeled_proto.unsqueeze(1),unlabeled_proto.unsqueeze(2),-1)   # Bs,un*100,un*100
        # unlabeled_adj = torch.zeros_like(unlabeled_sim)
        # # unlabeled_adj[unlabeled_sim>0] =1
        # unlabeled_adj[unlabeled_sim>0.5] =1
        # unlabeled_proto_list = []
        # for i in range(unlabeled_proto.shape[0]):
        #     unlabeled_proto_i = unlabeled_proto[i]
        #     unlabeled_adj_i = unlabeled_adj[i]
        #     unlabeled_proto_i=self.GraphAttentionLayer(self.LayerNorm_GAT(unlabeled_proto_i),unlabeled_adj_i)
        #     unlabeled_proto_list.append(unlabeled_proto_i)
        # unlabeled_proto = torch.stack(unlabeled_proto_list,dim=0) # Bs, un*100, c

        ### ALE
        sup_fg_proto,sup_bg_proto=self.sup_proto(sup_fts,sup_mask)  #B,c
        part_sup_proto = sup_fg_proto.view(B,-1,c) + self.part_tokens.expand(B,self.opt.fg_num,c)   #B,5,c
        part_sup_proto= part_sup_proto+self.afha*self.CrossAttention(self.LayerNorm(part_sup_proto),self.LayerNorm(sup_fts.view(B,S,-1,c)).clone(),self.LayerNorm(sup_fts.view(B,S,-1,c)).clone(),(sup_mask.view(B,S,-1)==1),OT=True)
        part_sup_proto= part_sup_proto+self.MLP(self.LayerNorm_mlp(part_sup_proto)) #B,5,c

        part_proto = torch.cat([sup_bg_proto.view(B,-1,c),part_sup_proto],dim=1)    # B,6,c
        query_sim = F.cosine_similarity(part_proto.unsqueeze(2),unlabeled_proto.unsqueeze(1),-1)*20 # B,6,un*100
        query_sim = query_sim.softmax(dim=-1)
        part_proto = part_proto + self.beta* (query_sim @ unlabeled_proto)

        ###SAD
        part_pred, pred_score = self.part_classifier(sup_fts, qry_fts,part_proto, sup_mask) # B,h,w; B,c,s
        aux_mask_tmp =[]
        aux_mask_tmp.append(pred_score)
        part_mask = []
        for i in range(self.opt.fg_num+1):
            part_mask_i = (part_pred==i).float()
            part_mask.append(part_mask_i)
        part_mask = torch.stack(part_mask,dim=1)    # B,5,h,w
        part_proto = part_proto+self.Mask_Attention(self.LayerNorm1(part_proto),self.LayerNorm1(qry_fts.view(B,1,-1,c)).clone(),self.LayerNorm1(qry_fts.view(B,1,-1,c)).clone(),(part_mask==1).view(B,self.opt.fg_num+1,-1))
        part_proto = (part_proto+self.MLP1(self.LayerNorm_mlp1(part_proto)))  # B,6,c

        for j in range(self.agg_depth):
            part_pred, pred_score1 = self.part_classifier_refine(qry_fts, part_proto)  # B,h,w; B,c,s
            part_mask = []
            for i in range(self.opt.fg_num + 1):
                part_mask_i = (part_pred == i).float()
                part_mask.append(part_mask_i)
            part_mask = torch.stack(part_mask, dim=1)  # B,5,h,w
            part_proto = part_proto + self.Mask_Attention1[j](self.LayerNorm1_1[j](part_proto),
                                                           self.LayerNorm1_1[j](qry_fts.view(B, 1, -1, c)).clone(),
                                                           self.LayerNorm1_1[j](qry_fts.view(B, 1, -1, c)).clone(),
                                                           (part_mask == 1).view(B, self.opt.fg_num + 1, -1))
            part_proto = (part_proto + self.MLP1_1[j](self.LayerNorm_mlp1_1[j](part_proto)))  # B,6,c
            aux_mask_tmp.append(pred_score1)
        part_proto = part_proto.unsqueeze(1)

        #
        qry_fts= qry_fts.view(B,-1,c) + self.CrossAttention1(self.LayerNorm2(qry_fts.view(B,1,-1,c)),self.LayerNorm2(part_proto),self.LayerNorm2(part_proto))
        qry_fts = qry_fts + self.MLP2(self.LayerNorm_mlp2(qry_fts))    # B,10,c

        qry_fts = qry_fts.view(B,c,h,w)
        pred= self.classifier(sup_fts, qry_fts, part_proto.squeeze(1), sup_mask)  # [B, 2, h, w]  [B,s,2,h,w]

        # Output
        if not out_shape:
            out_shape = y.shape[-2:] if y is not None else (H, W)
        out = interpb(pred, out_shape)  # [BQ, 2, *, *]
        aux_mask_list= []
        for aux_mask in aux_mask_tmp:
            aux_out = interpb(aux_mask, out_shape)  # [BQ, 2, *, *]
            aux_mask_list.append(aux_out)

        output = dict(out=out,aux_out=aux_mask_list)

        return output


    def part_classifier_refine(self, qry_fts, part_proto):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]
        text_token:
            [B,C]   [B*(S+1)*G,C ]
        background_token:
            [B*G,C]
        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, _, c, h, w = qry_fts.shape

        proto = part_proto.view(B,c,-1)  # B,C,6S

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)  # B,C,H,W
        pro_distances = []
        for i in range(proto.shape[-1]):
            pro_p = proto[:, :, i]
            pro_d = F.cosine_similarity(
                qry_fts, pro_p[..., None, None], dim=1) * 20  # [B, h, w]
            pro_distances.append(pro_d)
        pro_distances = torch.stack(pro_distances, dim=0) # 6S,B,h,w
        part_mask = pro_distances.max(0)[1] #B,h,w
        part_score = pro_distances.max(0)[0] #B,h,w
        part_score = (part_score>=self.filter).float()
        part_mask = part_mask*part_score
        pred_score = torch.cat((pro_distances[0].unsqueeze(1), pro_distances[1:, ].max(0)[0].unsqueeze(1)),
                               dim=1)  # B,2,h,w

        return part_mask, pred_score


    def part_proto_generation(self, qry_fts, part_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]
        text_token:
            [B,C]   [B*(S+1)*G,C ]
        background_token:
            [B*G,C]
        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, _, c, h, w = qry_fts.shape
        # sup_tokens,qry_tokens = text_token.view(B,S+1,-1,c).split([S,1],dim=1) #[B,S,G,c] [B,1,G,c]
        # merge_tokens = self.afha * qry_tokens + (1-self.afha) * sup_tokens.mean(1,keepdim=True) # [B,1,G,c]
        # FG proxies
        part_qry_proto=[]
        for i in range(self.opt.fg_num+1):
            part_mask_i = (part_mask == i).view(-1, 1, h * w)  # [B, 1, hw]
            fg_vecs = torch.sum(qry_fts.reshape(-1, c, h * w) * part_mask_i, dim=-1) / (part_mask_i.sum(dim=-1) + 1e-5)  # [BS, c]
            part_qry_proto.append(fg_vecs)
        part_qry_proto = torch.stack(part_qry_proto,dim=-2)
        return part_qry_proto



    def part_classifier(self, sup_fts, qry_fts, part_proto,sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]
        text_token:
            [B,C]   [B*(S+1)*G,C ]
        background_token:
            [B*G,C]
        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """

        B, S, c, h, w = sup_fts.shape
        # FG proxies

        proto = part_proto.view(B,c,-1)  # B,C,6S

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)  # B,C,H,W
        pro_distances = []
        for i in range(proto.shape[-1]):
            pro_p = proto[:, :, i]
            pro_d = F.cosine_similarity(
                qry_fts, pro_p[..., None, None], dim=1) * 20  # [B, h, w]
            pro_distances.append(pro_d)
        pro_distances = torch.stack(pro_distances, dim=0) # 6S,B,h,w
        part_mask = pro_distances.max(0)[1] #B,h,w
        part_score = pro_distances.max(0)[0] #B,h,w
        part_score = (part_score>=self.filter).float()
        part_mask = part_mask*part_score
        pred_score = torch.cat((pro_distances[0].unsqueeze(1),pro_distances[1:,].max(0)[0].unsqueeze(1)),dim=1)   #B,2,h,w

        return part_mask,pred_score
    def sup_proto(self, sup_fts,sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]
        text_token:
            [B,C]   [B*(S+1)*G,C ]
        background_token:
            [B*G,C]
        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, S, c, h, w = sup_fts.shape

        sup_fg = (sup_mask == 1).view(-1, 1, h * w)  # [BS, 1, hw]
        sup_bg = (sup_mask == 0).view(-1, 1, h * w)  # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)  # [BS, c]
        bg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_bg, dim=-1) / (sup_bg.sum(dim=-1) + 1e-5)  # [BS, c]
        # Merge multiple shots
        # qry_fts = qry_fts.reshape(-1, c, h, w)  #B,C,H,W
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1,keepdim=True)  # [B,c,1]

        bg_proto = bg_vecs.view(B, S, c).mean(dim=1,keepdim=True).permute(0,2,1)  # [B, c, S]
        return fg_proto,bg_proto
    def unlabeled_proto(self, unlabeled_features,unlabeled_y):

        B, c, h, w = unlabeled_features.shape
        unlabeled_proto = []
        for i in range(100):
            index = (unlabeled_y == i).view(-1, 1, h * w)
            proto_i = torch.sum(unlabeled_features.reshape(-1, c, h * w) * index, dim=-1) / (index.sum(dim=-1) + 1e-5)  # [BS, c]
            unlabeled_proto.append(proto_i)
        unlabeled_proto = torch.stack(unlabeled_proto,dim=1)
        unlabeled_proto = unlabeled_proto.view(B//self.opt.unlabeled,-1,c)
        return unlabeled_proto


    def classifier(self, sup_fts, qry_fts, part_proto, sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]
        text_token:
            [B,C]   [B*(S+1)*G,C ]
        background_token:
            [B*G,C]
        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        part_bg_proto, part_fg_proto = part_proto.split([1,self.opt.fg_num],dim=1)  #B,1,c;B,5,c
        B, S, c, h, w = sup_fts.shape
        # sup_tokens,qry_tokens = text_token.view(B,S+1,-1,c).split([S,1],dim=1) #[B,S,G,c] [B,1,G,c]
        # merge_tokens = self.afha * qry_tokens + (1-self.afha) * sup_tokens.mean(1,keepdim=True) # [B,1,G,c]
        # FG proxies
        sup_fg = (sup_mask == 1).view(-1, 1, h * w)  # [BS, 1, hw]
        sup_bg = (sup_mask == 0).view(-1, 1, h * w)  # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)  # [BS, c]
        bg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_bg, dim=-1) / (sup_bg.sum(dim=-1) + 1e-5)  # [BS, c]
        # Merge multiple shots
        # qry_fts = qry_fts.reshape(-1, c, h, w)  #B,C,H,W
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1,keepdim=True)  # [B,c,1]

        # fg_proto = torch.cat((fg_proto,part_proto),dim=1).permute(0,2,1)    #[B, c, 2]
        fg_proto = torch.cat((fg_proto,part_fg_proto),dim=1).permute(0,2,1)    #[B, c, 2]
        bg_proto = bg_vecs.view(B, S, c).mean(dim=1,keepdim=True).permute(0,2,1)  # [B, c, S]
        bg_proto = torch.cat((bg_proto,part_bg_proto.view(B,c,-1)),dim=2)
        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)  # B,C,H,W
        pred,bg_distance,_ = self.compute_multi_similarity(fg_proto, bg_proto, qry_fts)



        return pred


    @staticmethod
    def compute_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity
        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar  # [B, h, w]
        if len(bg_proto.shape) == 3:  # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], dim=1) * dist_scalar  # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = torch.stack(bg_distances, dim=0).max(0)[0]
        else:  # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar  # [B, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)  # [B, 2, h, w]

        return pred

    @staticmethod
    def compute_multi_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        if len(fg_proto.shape) ==3:
            fg_distances=[]
            fg_dis=[]
            for i in range(fg_proto.shape[-1]):
                fg_p = fg_proto[:,:,i]
                fp_d = F.cosine_similarity(
                    qry_fts, fg_p[..., None, None], dim=1) * dist_scalar  # [B, h, w]
                fg_distances.append(fp_d)
                fg_dis.append(fp_d)
            fg_distances = torch.stack(fg_distances,dim=0).max(0)[0]
            fg_dis = torch.stack(fg_dis,dim=1)
        else:
            fg_distances = F.cosine_similarity(
                qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar  # [B, h, w]

        if len(bg_proto.shape) == 3:  # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], dim=1) * dist_scalar  # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = torch.stack(bg_distances, dim=0).max(0)[0]
        else:  # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar  # [B, h, w]
        pred = torch.stack((bg_distance, fg_distances), dim=1)  # [B, 2, h, w]
        dis = torch.cat((bg_distance.unsqueeze(1), fg_dis), dim=1)  # [B, 2, h, w]

        return pred,bg_distance,dis


    def load_weights(self, ckpt_path, logger, strict=True):
        """

        Parameters
        ----------
        ckpt_path: Path
            path to the checkpoint
        logger
        strict: bool
            strict mode or not

        """
        weights = torch.load(str(ckpt_path), map_location='cpu')
        if "model_state" in weights:
            weights = weights["model_state"]
        if "state_dict" in weights:
            weights = weights["state_dict"]
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
        # Update with original_encoder
        weights.update({k: v for k, v in self.state_dict().items() if 'original_encoder' in k})

        self.load_state_dict(weights, strict=strict)
        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} initialized from {ckpt_path}")

    @staticmethod
    def get_or_download_pretrained(backbone, progress):
        if backbone not in pretrained_weights:
            raise ValueError(f'Not supported backbone {backbone}. '
                             f'Available backbones: {list(pretrained_weights.keys())}')

        cached_file = Path(pretrained_weights[backbone])
        if cached_file.exists():
            return cached_file

        # Try to download
        url = model_urls[backbone]
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, str(cached_file), progress=progress)
        return cached_file

    def get_params_list(self):
        params = []
        for var in self.parameters():
            if var.requires_grad:
                params.append(var)
        return [{'params': params}]
