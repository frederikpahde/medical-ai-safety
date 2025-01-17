import torch
import torch.nn as nn
from typing import Iterable
import math
from enum import Enum
from zennit.layer import Sum
from zennit.torchvision import VGGCanonizer

NormType = Enum('NormType', 'Batch BatchZero')

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = [ps] if not isinstance(ps,Iterable) else ps
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=False) if act=="relu" else nn.ELU(inplace=False)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.AdaptiveAvgPool1d(1), nn.Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    return nn.Sequential(*layers)

def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None, layer_norm=False):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in) if layer_norm is False else nn.LayerNorm(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad():
        if getattr(m, 'bias', None) is not None: m.bias.fill_(0.)
    return m

def _get_norm(prefix, nf, zero=False, **kwargs):
    "Norm layer with `nf` features initialized depending on `norm_type`."
    bn = getattr(nn, f"{prefix}1d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def BatchNorm(nf, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, zero=norm_type==NormType.BatchZero, **kwargs)

class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=True, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=nn.ReLU, init=nn.init.kaiming_normal_, xtra=None, **kwargs):
        if padding is None: padding = ((ks-1)//2)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None: bias = not(bn)
        conv_func = nn.Conv1d
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs), init)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

class MHSA1d(nn.Module):
    def __init__(self, n_dims, length=14, heads=4):
        super(MHSA1d, self).__init__()
        self.heads = heads

        self.query = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv1d(n_dims, n_dims, kernel_size=1)

        self.rel = nn.Parameter(torch.randn([1, heads, n_dims // heads, length]), requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, length = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1) #128,4,16,16
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1) #128,4,16,16
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, length)

        return out
    
class ResBlock(nn.Module):
    "Resnet block from `ni` to `nh` with `stride`"
    def __init__(self, expansion, ni, nf, stride=1, kernel_size=3, groups=1, nh1=None, nh2=None, dw=False, g2=1,
                 norm_type=NormType.Batch, act_cls=nn.ReLU, pool=nn.AvgPool1d, pool_first=True, heads=4, mhsa=False, input_size=None, **kwargs):
        super().__init__()
        assert(mhsa is False or expansion>1)
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, **kwargs)
        if(expansion ==1):
            layers  = [ConvLayer(ni,  nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),ConvLayer(nh2,  nf, kernel_size, groups=g2, **k1)]
        else:
            layers = [ConvLayer(ni,  nh1, 1, **k0)]
            if(mhsa==False):
                layers.append(ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0))
            else:
                assert(nh1==nh2)
                layers.append(MHSA1d(nh1, length=int(input_size), heads=heads))
                if stride == 2:
                    layers.append(nn.AvgPool1d(2, 2))
            layers.append(ConvLayer(nh2,  nf, 1, groups=g2, **k1))
         
        self.convs = nn.Sequential(*layers)
        convpath = [self.convs]
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(2, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=False) if act_cls is nn.ReLU else act_cls()
        self.sum = Sum()

    def forward(self, x): 
        x1 = self.convpath(x)
        x2 = self.idpath(x)
        # return self.act(x1 + x2)
        return self.act(self.sum(torch.stack([x1,x2], dim=-1)))
    
class XResNet1d(nn.Sequential):
    def __init__(self, block, expansion, layers, input_channels=3, num_classes=1000, stem_szs=(32,32,64), input_size=1000, heads=4, mhsa=False, kernel_size=5,kernel_size_stem=5,
                 widen=1.0, act_cls=nn.ReLU, lin_ftrs_head=None, ps_head=0.5, bn_head=True, act_head="relu", concat_pooling=False, **kwargs):
        self.block = block
        self.expansion = expansion
        self.act_cls = act_cls

        stem_szs = [input_channels, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=kernel_size_stem, stride=2 if i==0 else 1, act_cls=act_cls)
                for i in range(3)]
        stem.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        if(input_size is not None):
            self.input_size = math.floor((input_size-1)/2+1) #strided conv and MaxPool1d
            self.input_size = math.floor((self.input_size-1)/2+1)

        #block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [int(o*widen) for o in [64,64,64,64] +[32]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        #ATTENTION- this uses the S1 Botnet variant with stride 1 in the final block
        blocks = [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                   stride=1 if i==0 else (1 if i==len(layers)-1 and mhsa else 2), kernel_size=kernel_size, heads=heads, mhsa=mhsa if i==len(layers)-1 else False, **kwargs)
                  for i,l in enumerate(layers)]
        
        head = create_head1d(block_szs[-1]*expansion, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)

        super().__init__(
            *stem,
            *blocks,
            head,
        )
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, heads=4, mhsa=False, **kwargs):
        input_size0 = self.input_size
        input_size1 = math.floor((self.input_size-1)/stride+1) if self.input_size is not None else None
        self.input_size = input_size1
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      kernel_size=kernel_size, act_cls=self.act_cls, heads=heads, mhsa=mhsa, input_size=input_size0 if i==0 else input_size1, **kwargs)
              for i in range(blocks)])

    def get_layer_groups(self):
        return (self[3],self[-1])

    def get_output_layer(self):
        return self[-1][-1]

    def set_output_layer(self,x):
        self[-1][-1]=x

def _xresnet1d(expansion, layers, **kwargs):
    return XResNet1d(ResBlock, expansion, layers, **kwargs)

def xresnet1d18 (**kwargs): return _xresnet1d(1, [2, 2,  2, 2], **kwargs)
def xresnet1d34 (**kwargs): return _xresnet1d(1, [3, 4,  6, 3], **kwargs)
def xresnet1d50 (**kwargs): return _xresnet1d(4, [3, 4,  6, 3], **kwargs)
def xresnet1d101(**kwargs): return _xresnet1d(4, [3, 4, 23, 3], **kwargs)
def xresnet1d152(**kwargs): return _xresnet1d(4, [3, 8, 36, 3], **kwargs)
def xresnet1d18_deep  (**kwargs): return _xresnet1d(1, [2,2,2,2,1,1], **kwargs)
def xresnet1d34_deep  (**kwargs): return _xresnet1d(1, [3,4,6,3,1,1], **kwargs)
def xresnet1d50_deep  (**kwargs): return _xresnet1d(4, [3,4,6,3,1,1], **kwargs)
def xresnet1d18_deeper(**kwargs): return _xresnet1d(1, [2,2,1,1,1,1,1,1], **kwargs)
def xresnet1d34_deeper(**kwargs): return _xresnet1d(1, [3,4,6,3,1,1,1,1], **kwargs)
def xresnet1d50_deeper(**kwargs): return _xresnet1d(4, [3,4,6,3,1,1,1,1], **kwargs)

def xbotnet1d50 (**kwargs): return _xresnet1d(4, [3, 4,  6, 3], mhsa=True, **kwargs)
def xbotnet1d101(**kwargs): return _xresnet1d(4, [3, 4, 23, 3], mhsa=True, **kwargs)
def xbotnet1d152(**kwargs): return _xresnet1d(4, [3, 8, 36, 3], mhsa=True, **kwargs)

class ECGResNet(nn.Module):

    def __init__(self, base_model, out_dim, widen=1.0, big_input=False, concat_pooling=True):
        super(ECGResNet, self).__init__()
        self.resnet_dict = {
            "xresnet1d50": xresnet1d50(widen=widen, concat_pooling=concat_pooling),
            "xresnet1d101": xresnet1d101(widen=widen, concat_pooling=concat_pooling)}

        resnet = self._get_basemodel(base_model)
    
        ## insert identities
        resnet[4] = nn.Sequential(*list(resnet[4]), nn.Identity())
        resnet[5] = nn.Sequential(*list(resnet[5]), nn.Identity())
        resnet[6] = nn.Sequential(*list(resnet[6]), nn.Identity())
        resnet[7] = nn.Sequential(*list(resnet[7]), nn.Identity())
        self.identity_global_pool = nn.Identity()
        
        list_of_modules = list(resnet.children())
        
        self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0])
        num_ftrs = resnet[-1][-1].in_features
        
        if big_input:
            resnet[0][0] = nn.Conv1d(12, 32, kernel_size=25, stride=10, padding=10)
        else:
            resnet[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm1d(512 if concat_pooling else 256)
        self.drop = nn.Dropout(p=0.5)
        self.l1 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = self.identity_global_pool(h)
        h = h.squeeze(-1)
        x = self.bn(h)
        x = self.drop(x)
        x = self.l1(x)
        return x
    
def get_xresnet1d50(ckpt_path=None, pretrained=True, n_class: int = 23, device: str="cuda") -> torch.nn.Module:
    m = ECGResNet("xresnet1d50", n_class)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        m.load_state_dict(checkpoint)

    return m

def get_xresnet1d50_canonizer():
    return [VGGCanonizer()]

if __name__ == "__main__":
    m = ECGResNet("xresnet1d50", 23)
    x = torch.rand(5, 12, 1000)
    pred = m(x)
    print("Done")