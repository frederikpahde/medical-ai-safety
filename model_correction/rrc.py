import torch

from model_correction.base_correction_method import Freeze
from model_correction.clarc import Clarc


class RRClarc(Clarc):
    """
    Classifier with Right Reasons loss for latent concept unlearning.
    """

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.lamb = self.config["lamb"]
        self.aggregation = self.config.get("compute", "l2_mean")
        self.gradient_target = self.config.get("criterion", "all_logits_random")
        self.intermediate = torch.tensor(0.0)
        self.layer_name = config["layer_name"]

    def clarc_hook(self, m, i, o):
        self.intermediate = o
        return o.clone()

    def criterion_fn(self, y_hat, y):
        if self.gradient_target == 'max_logit':
            return y_hat.max(1)[0]
        elif self.gradient_target == 'target_logit':
            target_class = self.config.get("target_class", y)
            return y_hat[range(len(y)), target_class]
        elif self.gradient_target == 'all_logits':
            return (y_hat).sum(1)
        elif self.gradient_target == 'all_logits_random':
            return (y_hat * torch.sign(0.5 - torch.rand_like(y_hat))).sum(1)
        elif self.gradient_target == 'logprobs':
            return (y_hat.softmax(1) + 1e-5).log().mean(1)
        else:
            raise NotImplementedError(f"Criterion {self.gradient_target} not implemented")

    def loss_compute(self, gradient):
        is_2dim = gradient.dim() == 2
        gradient = gradient[..., None, None] if is_2dim else gradient
        if "swin_former" in self.config["model_name"]:
            gradient = gradient.transpose(1,3).transpose(2,3)
        cav = self.cav.to(gradient)
        if "mean" in self.aggregation and gradient.dim() != 2:
            axes = 2 if gradient.dim() == 3 else (2,3)
            gradient = gradient.mean(axes, keepdim=True).expand_as(gradient)

        if self.mode == "cavs_full":
            g_flat = gradient.flatten(1)
        else:
            axes_new = (1, 0, 2, 3) if gradient.dim() == 4 else (1, 0, 2)
            g_flat = gradient.permute(axes_new).flatten(start_dim=1).permute(1, 0)

        if "cosine" in self.aggregation:
            return torch.nn.functional.cosine_similarity(g_flat, cav).abs().mean(0)
        elif "l2" in self.aggregation:
            return ((g_flat * cav).sum(1) ** 2).mean(0)  
        elif "l1" in self.aggregation:
            return (g_flat * cav).sum(1).abs().mean(0)  
        else:
            raise NotImplementedError

    def default_step(self, x, y, stage):
        with torch.enable_grad():
            x.requires_grad = True
            y_hat = self(x)
            yc_hat = self.criterion_fn(y_hat, y)
            grad = torch.autograd.grad(outputs=yc_hat,
                                       inputs=self.intermediate,
                                       create_graph=True,
                                       retain_graph=True,
                                       grad_outputs=torch.ones_like(yc_hat))[0]
            aux_loss = self.loss_compute(grad)

        loss = self.loss(y_hat, y) + self.lamb * aux_loss
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             f"{stage}_auxloss": aux_loss},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_callbacks(self):
        return [
            Freeze(self.layer_name)
        ]
