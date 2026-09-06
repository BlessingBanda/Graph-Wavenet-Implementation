import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


#my additions for the dynamic graph generation
class batched_nconv(nn.Module):
    """Diffusion-conv operator for a per-sample (B, N, N) adjacency,
    instead of the shared (N, N) matrix the original nconv assumes."""
    def __init__(self):
        super(batched_nconv, self).__init__()

    def forward(self, x, A):
        # x: (B, C, N, L)   A: (B, N, N)
        x = torch.einsum('bcvl,bvw->bcwl', (x, A))
        return x.contiguous()


class batched_gcn(nn.Module):
    """Graph conv layer that consumes a single dynamic (B,N,N) support,
    mirroring the existing `gcn` class's structure (order-K diffusion + mlp)."""
    def __init__(self, c_in, c_out, dropout, order=2):
        super(batched_gcn, self).__init__()
        self.batched_nconv = batched_nconv()
        c_in_total = (order + 1) * c_in
        self.mlp = linear(c_in_total, c_out)   # reuse existing `linear` class
        self.dropout = dropout
        self.order = order

    def forward(self, x, DA):
        out = [x]
        x1 = self.batched_nconv(x, DA)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.batched_nconv(x1, DA)
            out.append(x2)
            x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DynamicGraphGenerator(nn.Module):
    def __init__(self, num_nodes, hidden_dim, emb_dim, dropout, support_len, device, order=2):
        super(DynamicGraphGenerator, self).__init__()
        in_dim_gconv = 2 + hidden_dim
        self.static_gconv = gcn(in_dim_gconv, emb_dim, dropout, support_len=support_len, order=order)
        self.E_src = nn.Parameter(torch.randn(num_nodes, emb_dim).to(device), requires_grad=True).to(device)
        self.E_tgt = nn.Parameter(torch.randn(num_nodes, emb_dim).to(device), requires_grad=True).to(device)

    def forward(self, X_raw, H_layer1, supports):
        V_agg = X_raw[:, 0:1, :, :].mean(dim=-1)
        T_agg = X_raw[:, 1:2, :, :].mean(dim=-1)
        H_proxy = H_layer1.mean(dim=-1)
        I = torch.cat([V_agg, T_agg, H_proxy], dim=1).unsqueeze(-1)
        DF = self.static_gconv(I, supports)
        DF = DF.squeeze(-1).permute(0, 2, 1)
        E_src_dyn = DF * self.E_src.unsqueeze(0)
        E_tgt_dyn = DF * self.E_tgt.unsqueeze(0)
        DA = F.softmax(F.relu(torch.bmm(E_src_dyn, E_tgt_dyn.transpose(1, 2))), dim=-1)
        return DA


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, dynamicadj=False, dyn_emb_dim=10):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
    
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
                
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        # --- NEW: dynamic module setup, unconditional on gcn_bool/addaptadj ---
        self.dynamicadj = dynamicadj
        if self.dynamicadj:
            self.dyn_graph_gen = DynamicGraphGenerator(
                num_nodes=num_nodes,
                hidden_dim=dilation_channels,
                emb_dim=dyn_emb_dim,
                dropout=dropout,
                support_len=self.supports_len if supports is not None else 0,
                device=device
            )
            self.batched_gconv = nn.ModuleList()

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                if self.dynamicadj:
                    self.batched_gconv.append(batched_gcn(dilation_channels, residual_channels, dropout))


        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input

        x_raw = x  # --- NEW: keep the (padded) raw 2-channel input around for the dynamic graph generator ---

        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        dynamic_DA = None  # --- NEW: will hold the (B,N,N) dynamic adjacency once generated ---

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            gconv_input = x  # --- NEW: dilation_channels-wide tensor, shared by gconv, batched_gconv, and the hypernet ---

            # --- NEW: generate the dynamic graph once, right after layer 0's TCN output ---
            if self.dynamicadj and i == 0:
                dynamic_DA = self.dyn_graph_gen(x_raw, gconv_input, self.supports)

            # parametrized skip connection

            s = gconv_input  # --- CHANGED: was `s = x`; now explicitly uses gconv_input for clarity ---
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](gconv_input, new_supports)  # --- CHANGED: was `x`, now `gconv_input` ---
                else:
                    x = self.gconv[i](gconv_input,self.supports)  # --- CHANGED: was `x`, now `gconv_input` ---
            else:
                x = self.residual_convs[i](gconv_input)  # --- CHANGED: was `x`, now `gconv_input` ---

            # --- NEW: add the dynamic graph's contribution, using the SAME pre-gconv input as gconv/residual above ---
            if self.dynamicadj and dynamic_DA is not None:
                x = x + self.batched_gconv[i](gconv_input, dynamic_DA)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x





