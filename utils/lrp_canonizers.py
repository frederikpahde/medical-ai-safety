import torch
from torchvision.models.efficientnet import MBConv, FusedMBConv
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, BasicBlock as ResNetBasicBlock
from torchvision.models.vision_transformer import EncoderBlock, Encoder
from torchvision.ops.misc import SqueezeExcitation
from zennit import canonizers as canonizers
from zennit import layer as zlayer
from zennit.canonizers import CompositeCanonizer, SequentialMergeBatchNorm, AttributeCanonizer, MergeBatchNorm
from zennit.layer import Sum
from timm.layers.squeeze_excite import SEModule
from timm.layers import BatchNormAct2d
from timm.models.resnet import Bottleneck as ResNetBottleneckTimm, BasicBlock as ResNetBasicBlockTimm
from timm.models.rexnet import LinearBottleneck as RexNetLinearBottleneckTimm
from torch.nn.modules.activation import ReLU
from torch.nn import AdaptiveAvgPool2d, Sequential
from torchvision.models import DenseNet
from zennit.core import collect_leaves
from zennit.types import ConvolutionTranspose

class SignalOnlyGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2):
        return x1 * x2

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), grad_output


class SECanonizer(canonizers.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SqueezeExcitation):
            print("Using SqueezeExcitation module")
            attributes = {
                'forward': cls.forward.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
    
        if isinstance(module, SEModule):
            print("Using SE module")
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
        return None

    @staticmethod
    def forward_timm(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        scale = self.gate(x_se)
        return self.fn_gate.apply(scale, x)
    
    @staticmethod
    def forward(self, input):
        scale = self._scale(input)
        return self.fn_gate.apply(scale, input)


class MBConvCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, MBConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        
        if isinstance(module, FusedMBConv):
            attributes = {
                'forward': cls.forward_fused.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)

            # result += input
            result = torch.stack([input, result], dim=-1)
            result = self.canonizer_sum(result)
        return result
    
    @staticmethod
    def forward_fused(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)

            # result += input
            result = torch.stack([input, result], dim=-1)
            result = self.canonizer_sum(result)
        return result




class NewAttention(torch.nn.MultiheadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inp):
        result, _ = super().forward(inp, inp, inp, need_weights=False)
        return result


class EncoderBlockCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, EncoderBlock):

            new_attention = NewAttention(module.self_attention.embed_dim,
                                         module.self_attention.num_heads,
                                         module.self_attention.dropout,
                                         batch_first=True)
            for name, param in module.self_attention.named_parameters():
                if "." in name:
                    getattr(new_attention, name.split(".")[0]).register_parameter(name.split(".")[1], param)
                else:
                    new_attention.register_parameter(name, param)
            attributes = {
                'forward': cls.forward.__get__(module),
                'new_attention': new_attention,
                'sum': zlayer.Sum(),
            }
            return attributes
        return None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.new_attention(x)
        x = self.dropout(x)
        x = self.sum(torch.stack([x, input], dim=-1))

        y = self.ln_2(x)
        y = self.mlp(y)
        return self.sum(torch.stack([x, y], dim=-1))


class EncoderCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, Encoder):
            attributes = {
                'forward': cls.forward.__get__(module),
                'sum': zlayer.Sum(),
            }
            return attributes
        return None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = self.sum(torch.stack([input, self.pos_embedding.expand_as(input)], dim=-1))
        return self.ln(self.layers(self.dropout(input)))


class VITCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            canonizers.SequentialMergeBatchNorm(),
            EncoderCanonizer(),
            EncoderBlockCanonizer(),
        ))


class ResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if isinstance(module, ResNetBottleneckTimm):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if isinstance(module, RexNetLinearBottleneckTimm):
            attributes = {
                'forward': cls.forward_timm_rexnet.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out
    
    @staticmethod
    def forward_timm(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        
        # x += shortcut
        x = torch.stack([shortcut, x], dim=-1)
        x = self.canonizer_sum(x)
        x = self.act3(x)

        return x
    
    @staticmethod
    def forward_timm_rexnet(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.conv_exp is not None:
            x = self.conv_exp(x)
        x = self.conv_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act_dw(x)
        x = self.conv_pwl(x)
        if self.use_shortcut:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x_stacked = torch.stack([shortcut, x[:, 0:self.in_channels]], dim=-1)
            x = torch.cat([self.canonizer_sum(x_stacked), x[:, self.in_channels:]], dim=1)
            # x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
        return x


class ResNetBasicBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        
        if isinstance(module, ResNetBasicBlockTimm):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward_timm(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        
        x = torch.stack([shortcut, x], dim=-1)
        x = self.canonizer_sum(x)
        # x += shortcut
        x = self.act2(x)
        return x
    
    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        if hasattr(self, 'last_conv'):
            out = self.last_conv(out)
            out = out + 0

        out = self.relu(out)

        return out

class BatchNormActCanonizer(AttributeCanonizer):
    '''Canonizer specifically for SE layers.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, BatchNormAct2d):
            new_bn = torch.nn.BatchNorm2d(module.num_features).eval()
            new_bn.bias.data = module.bias.data
            new_bn.weight.data = module.weight.data
            new_bn.running_mean = module.running_mean
            new_bn.running_var = module.running_var
            attributes = {
                'forward': cls.forward.__get__(module),
                'bn': new_bn
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified SE forward for SENetworks.'''
        x = self.bn(x)
        x = self.drop(x)
        x = self.act(x)
        return x

class CorrectCompositeCanonizer(CompositeCanonizer):
    # Zennit canonizer returns handles in the order they are applied.
    # We reverse the list so we can detach correctly after attach two different canonizers for the same parameter
    def apply(self, root_module):
        ret = super(CorrectCompositeCanonizer, self).apply(root_module)
        ret.reverse()
        return ret
    
class _TransitionNew(torch.nn.Sequential):
    def __init__(self, bn, conv) -> None:
        super().__init__()
        self.norm = bn
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = conv
        self.identity = torch.nn.Identity()
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

class DenseNetAdaptiveAvgPoolCanonizer(AttributeCanonizer):
    '''Canonizer specifically for AdaptiveAvgPooling2d layers at the end of torchvision.model densenet models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, DenseNet):
            attributes = {
                'forward': cls.forward.__get__(module),
                # 'last_conv': torch.nn.Identity(),
                # 'final_relu': ReLU(inplace=True),
                # 'pool': AdaptiveAvgPool2d((1, 1))
            }
            return attributes
        return None

    def copy(self):
        '''Copy this Canonizer.

        Returns
        -------
        obj:`Canonizer`
            A copy of this Canonizer.
        '''
        return DenseNetAdaptiveAvgPoolCanonizer()

    def register(self, module, attributes):

        module.features.transition1 = _TransitionNew(module.features.transition1.norm, module.features.transition1.conv)
        module.features.transition2 = _TransitionNew(module.features.transition2.norm, module.features.transition2.conv)
        module.features.transition3 = _TransitionNew(module.features.transition3.norm, module.features.transition3.conv)

        module.features.add_module('last_conv', torch.nn.Identity())
        module.features.add_module('final_relu', ReLU(inplace=True))
        module.features.add_module('adaptive_avg_pool2d', AdaptiveAvgPool2d((1, 1)))
        super(DenseNetAdaptiveAvgPoolCanonizer, self).register(module, attributes)

    def remove(self):
        '''Remove the overloaded attributes. Note that functions are descriptors, and therefore not direct attributes
        of instance, which is why deleting instance attributes with the same name reverts them to the original
        function.
        '''
        self.module.features = Sequential(*list(self.module.features.children())[:-2])
        for key in self.attribute_keys:
            delattr(self.module, key)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
class CorrectSequentialMergeBatchNorm(SequentialMergeBatchNorm):
    # Zennit does not set bn.epsilon to 0, resulting in an incorrect canonization. We solve this issue.
    def __init__(self):
        super(CorrectSequentialMergeBatchNorm, self).__init__()

    def apply(self, root_module):
        '''Finds a batch norm following right after a linear layer, and creates a copy of this instance to merge
        them by fusing the batch norm parameters into the linear layer and reducing the batch norm to the identity.

        Parameters
        ----------
        root_module: obj:`torch.nn.Module`
            A module of which the leaves will be searched and if a batch norm is found right after a linear layer, will
            be merged.

        Returns
        -------
        instances: list
            A list of instances of this class which modified the appropriate leaves.
        '''
        instances = []
        last_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(last_leaf, self.linear_type) and isinstance(leaf, self.batch_norm_type):
                if last_leaf.weight.shape[0] == leaf.weight.shape[0]:
                    instance = self.copy()
                    instance.register((last_leaf,), leaf)
                    instances.append(instance)
            last_leaf = leaf

        return instances

    def merge_batch_norm(self, modules, batch_norm):
        self.batch_norm_eps = batch_norm.eps
        super(CorrectSequentialMergeBatchNorm, self).merge_batch_norm(modules, batch_norm)
        batch_norm.eps = 0.

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        super(CorrectSequentialMergeBatchNorm, self).remove()
        self.batch_norm.eps = self.batch_norm_eps

# Canonizer to canonize BN->Linear or BN->Conv modules.
class SequentialMergeBatchNormtoRight(MergeBatchNorm):
    @staticmethod
    def convhook(module, x, y):
        # Add the feature map bias to the output of canonized conv layer with padding
        x = x[0]
        bias_kernel = module.canonization_params["bias_kernel"]
        pad1, pad2 = module.padding
        # ASSUMING module.kernel_size IS ALWAYS STRICTLY GREATER THAN module.padding
        # Upscale bias kernel to full feature map size
        if pad1 > 0:
            left_margin = bias_kernel[:, :, 0:pad1, :]
            right_margin = bias_kernel[:, :, pad1 + 1:, :]
            middle = bias_kernel[:, :, pad1:pad1 + 1, :].expand(1, bias_kernel.shape[1],
                                                                x.shape[2] - module.weight.shape[2] + 1,
                                                                bias_kernel.shape[-1])
            bias_kernel = torch.cat((left_margin, middle, right_margin), dim=2)

        if pad2 > 0:
            left_margin = bias_kernel[:, :, :, 0:pad2]
            right_margin = bias_kernel[:, :, :, pad2 + 1:]
            middle = bias_kernel[:, :, :, pad2:pad2 + 1].expand(1, bias_kernel.shape[1], bias_kernel.shape[-2],
                                                                x.shape[3] - module.weight.shape[3] + 1)
            bias_kernel = torch.cat((left_margin, middle, right_margin), dim=3)

        # account for stride by dropping some of the tensor
        if module.stride[0] > 1 or module.stride[1] > 1:
            indices1 = [i for i in range(0, bias_kernel.shape[2]) if i % module.stride[0] == 0]
            indices2 = [i for i in range(0, bias_kernel.shape[3]) if i % module.stride[1] == 0]
            bias_kernel = bias_kernel[:, :, indices1, :]
            bias_kernel = bias_kernel[:, :, :, indices2]
        ynew = y + bias_kernel
        return ynew

    def __init__(self):
        super().__init__()
        self.handles = []

    def apply(self, root_module):
        instances = []
        last_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(last_leaf, self.batch_norm_type) and isinstance(leaf, self.linear_type):
                instance = self.copy()
                instance.register((leaf,), last_leaf)
                instances.append(instance)
            last_leaf = leaf

        return instances

    def register(self, linears, batch_norm):
        '''Store the parameters of the linear modules and the batch norm module and apply the merge.

        Parameters
        ----------
        linear: list of obj:`torch.nn.Module`
            List of linear layer with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes
            `running_mean`, `running_var`, `weight`, `bias` and `eps`
        '''
        self.linears = linears
        self.batch_norm = batch_norm
        self.linear_params = [(linear.weight.data, getattr(linear.bias, 'data', None)) for linear in linears]

        self.batch_norm_params = {
            key: getattr(self.batch_norm, key).data for key in ('weight', 'bias', 'running_mean', 'running_var')
        }
        returned_handles = self.merge_batch_norm(self.linears, self.batch_norm)
        self.handles = returned_handles

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        super(SequentialMergeBatchNormtoRight, self).remove()
        self.batch_norm.eps = self.batch_norm_eps
        for h in self.handles:
            h.remove()
        for module in self.linears:
            if isinstance(module, torch.nn.Conv2d):
                if module.padding != (0, 0):
                    delattr(module, "canonization_params")

    def merge_batch_norm(self, modules, batch_norm):
        self.batch_norm_eps = batch_norm.eps
        return_handles = []
        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)  # Weight of the batch norm layer when seen as a linear layer
        shift = batch_norm.bias - batch_norm.running_mean * scale  # bias of the batch norm layer when seen as a linear layer

        for module in modules:
            original_weight = module.weight.data
            if module.bias is None:
                module.bias = torch.nn.Parameter(
                    torch.zeros(module.out_channels, device=original_weight.device, dtype=original_weight.dtype)
                )
            original_bias = module.bias.data

            if isinstance(module, ConvolutionTranspose):
                index = (slice(None), *((None,) * (original_weight.ndim - 1)))
            else:
                index = (None, slice(None), *((None,) * (original_weight.ndim - 2)))

            # merge batch_norm into linear layer to the right
            module.weight.data = (original_weight * scale[index])

            # module.bias.data = original_bias
            if isinstance(module, torch.nn.Conv2d):
                if module.padding == (0, 0):
                    module.bias.data = (original_weight * shift[index]).sum(dim=[1, 2, 3]) + original_bias
                else:
                    bias_kernel = shift[index].expand(*(shift[index].shape[0:-2] + original_weight.shape[-2:]))
                    temp_module = torch.nn.Conv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                                  kernel_size=module.kernel_size, padding=module.padding,padding_mode=module.padding_mode, bias=False)
                    temp_module.weight.data = original_weight
                    bias_kernel = temp_module(bias_kernel).detach()

                    module.canonization_params = {}
                    module.canonization_params["bias_kernel"] = bias_kernel
                    return_handles.append(module.register_forward_hook(SequentialMergeBatchNormtoRight.convhook))
            elif isinstance(module, torch.nn.Linear):
                module.bias.data = (original_weight * shift).sum(dim=1) + original_bias

        # change batch_norm parameters to produce identity
        batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
        batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
        batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
        batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)
        batch_norm.eps = 0.
        return return_handles
    
class ThreshReLUMergeBatchNorm(SequentialMergeBatchNormtoRight):
    # Hook functions for ReLU_thresh
    @staticmethod
    def prehook(module, x):
        module.canonization_params["original_x"] = x[0].clone()

    @staticmethod
    def fwdhook(module, x, y):
        x = module.canonization_params["original_x"]
        index = (None, slice(None), *((None,) * (module.canonization_params['weights'].ndim + 1)))
        y = module.canonization_params['weights'][index] * x + module.canonization_params['biases'][index]
        baseline_vals = -1. * (module.canonization_params['biases'] / module.canonization_params['weights'])[index]
        return torch.where(y > 0, x, baseline_vals)

    def __init__(self):
        super().__init__()
        self.relu = None
    @torch.no_grad()    # Need to force no_grad, as this will otherwise cause issues with CRP visiulization, trying to backward thorugh the graph multiple times
    def apply(self, root_module):
        instances = []
        oldest_leaf = None
        old_leaf = None
        mid_leaf = None
        for leaf in collect_leaves(root_module):
            if isinstance(old_leaf, self.batch_norm_type) and isinstance(mid_leaf, ReLU) and isinstance(leaf,
                                                                                                        self.linear_type):
                instance = self.copy()
                instance.register((leaf,), old_leaf, mid_leaf)
                instances.append(instance)
            elif isinstance(oldest_leaf, self.batch_norm_type) and isinstance(old_leaf, ReLU) and isinstance(mid_leaf,
                                                                                                             AdaptiveAvgPool2d) and isinstance(
                leaf, self.linear_type):
                instance = self.copy()
                instance.register((leaf,), oldest_leaf, old_leaf)
                instances.append(instance)
            oldest_leaf = old_leaf
            old_leaf = mid_leaf
            mid_leaf = leaf

        return instances

    def register(self, linears, batch_norm, relu):
        '''Store the parameters of the linear modules and the batch norm module and apply the merge.

        Parameters
        ----------
        linear: list of obj:`torch.nn.Module`
            List of linear layer with mandatory attributes `weight` and `bias`.
        batch_norm: obj:`torch.nn.Module`
            Batch Normalization module with mandatory attributes
            `running_mean`, `running_var`, `weight`, `bias` and `eps`
        '''
        self.relu = relu

        denominator = (batch_norm.running_var + batch_norm.eps) ** .5
        scale = (batch_norm.weight / denominator)  # Weight of the batch norm layer when seen as a linear layer
        shift = batch_norm.bias - batch_norm.running_mean * scale  # bias of the batch norm layer when seen as a linear layer
        self.relu.canonization_params = {}
        self.relu.canonization_params['weights'] = scale
        self.relu.canonization_params['biases'] = shift

        super().register(linears,batch_norm)
        self.handles.append(self.relu.register_forward_pre_hook(ThreshReLUMergeBatchNorm.prehook))
        self.handles.append(self.relu.register_forward_hook(ThreshReLUMergeBatchNorm.fwdhook))

    def remove(self):
        '''Undo the merge by reverting the parameters of both the linear and the batch norm modules to the state before
        the merge.
        '''
        super().remove()
        delattr(self.relu, "canonization_params")
        
class ResNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ResNetBottleneckCanonizer(),
            ResNetBasicBlockCanonizer(),
        ))

class EfficientNetBNCanonizer(CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            MBConvCanonizer(),
            canonizers.SequentialMergeBatchNorm(),
        ))

class RexNetCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            ResNetBottleneckCanonizer(),
            BatchNormActCanonizer(),
            SequentialMergeBatchNorm(),
        ))

class SequentialThreshCanonizer(CorrectCompositeCanonizer):
    def __init__(self):
        super().__init__((
            DenseNetAdaptiveAvgPoolCanonizer(),
            CorrectSequentialMergeBatchNorm(),
            ThreshReLUMergeBatchNorm(),
        ))
