
from argparse import ArgumentParser
import os
import torch
from matplotlib import pyplot as plt
import numpy as np
from datasets import load_dataset
from models import get_canonizer, get_fn_model_loader
import tqdm
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat
from crp.image import imgify, zimage
from torchvision.utils import make_grid
import torchvision.transforms as T
from utils.helper import load_config
from utils.render import COLORS, overlay_heatmap, overlay_heatmaps, vis_opaque_img_border

def get_parser():

    parser = ArgumentParser()
    parser.add_argument("--sample_ids", 
                        default="1990,2015,2047,2120", type=str) 

    parser.add_argument("--max_mode", default="relevance", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--n_concepts", default=3, type=int)
    parser.add_argument("--n_refimgs", default=6, type=int)
    parser.add_argument("--config_file", 
                        default="config_files/revealing/hyper_kvasir_attacked/local/resnet50d_identity_2.yaml")
    parser.add_argument('--savedir', default='plot_files/crp_plots/')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.config_file)
    sample_ids = [int(i) for i in args.sample_ids.split(",")]
    run_crp(config, sample_ids, args.n_concepts, args.n_refimgs, args.max_mode, args.split, args.savedir)

def run_crp(config, sample_ids, n_concepts, n_refimgs, max_mode, split, savedir):

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name']
    model_name = config['model_name']
    ckpt_path = config['ckpt_path']

    dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])

    model = get_fn_model_loader(model_name=model_name)(n_class=len(dataset.classes), ckpt_path=ckpt_path).to(device).eval()

    layer_name = config["layer_name"]
    layer_names = [layer_name]

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_map = {layer: cc for layer in layer_names}
    print(f"using layer {layer_name}")
    attribution = CondAttribution(model)

    fv_name = f"{config['dir_precomputed_data']}/crp_files/{dataset_name}_{split}_{model_name}"
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.normalize_fn,
                              path=fv_name, cache=None)


    for sample_id in tqdm.tqdm(sample_ids):

        data_p, y = fv.get_data_sample(sample_id, preprocessing=True)
        pred = model(data_p).detach().argmax().cpu()
        target = pred
        attr = attribution(data_p.clone().requires_grad_(),
                        [{"y": target}],
                        composite,
                        record_layer=layer_names)
        

        channel_rels = cc.attribute(attr.relevances[layer_name], abs_norm=True)
        topk = torch.topk(channel_rels[0], n_concepts)
        topk_ind = topk.indices.detach().cpu().numpy()#[1:2]
        topk_rel = topk.values.detach().cpu().numpy()#[1:2]
        attribution.model.zero_grad()

        conditions = [{"y": target, 
                        layer_name: c_id
                        } for c_id in topk_ind]

        data_p, _ = fv.get_data_sample(sample_id, preprocessing=True)
        attr_cond = attribution(data_p.clone().requires_grad_(), conditions, composite)
        attribution.model.zero_grad()
        cond_heatmap, _, _, _ = attr_cond
        torch.cuda.empty_cache()

        ref_imgs = fv.get_max_reference(topk_ind, layer_name, max_mode, (0, n_refimgs), composite=composite, rf=True,
                                plot_fn=vis_opaque_img_border)
        
        crp_plot_savedir = f"{savedir}/{dataset_name}/{config['config_name']}"
        plot_glocal_explanation(attr, fv, cond_heatmap, sample_id, n_concepts, n_refimgs, 
                                topk_ind, topk_rel, ref_imgs, max_mode, target, pred, y, crp_plot_savedir, 
                                plot_overlay=False)

        # plot_glocal_explanation(attr, fv, cond_heatmap, sample_id, n_concepts, n_refimgs, 
        #                         topk_ind, topk_rel, ref_imgs, max_mode, target, pred, y, crp_plot_savedir, 
        #                         plot_overlay=True)

def plot_glocal_explanation(attr, fv, cond_heatmap, sample_id, n_concepts, n_refimgs, 
                            topk_ind, topk_rel, ref_imgs, mode, target, pred, y, savedir, plot_overlay=False):
    base_size = 2
    
    ## Remove
    n_concepts = cond_heatmap.shape[0]
    ####
    NUM_HMS = 2
    cols = 4 if plot_overlay else 3
    width_ratios = [1] * (cols-1) + [n_refimgs / 4]
    fig, axs = plt.subplots(n_concepts, cols, gridspec_kw={'width_ratios': width_ratios},
                            figsize=(base_size * cols * 1.1, base_size * n_concepts), dpi=200, squeeze=False)
    level = 2
    resize = T.Resize((150, 150))

    for r, row_axs in enumerate(axs):

        for c, ax in enumerate(row_axs):
            if c == 0:
                if r == 0:
                    ax.set_title(f"input ({fv.dataset.classes[pred]}/gt: {fv.dataset.classes[y]})")
                    img =fv.get_data_sample(sample_id, preprocessing=False)[0][0].detach().cpu()
                    ax.imshow(img.numpy().transpose(1,2,0))
                elif r == 1:
                    ax.set_title("heatmap")
                    amax = attr.heatmap.abs().max()
                    img = imgify(attr.heatmap / amax, cmap="bwr", vmin=-1, vmax=1, symmetric=True, level=level)
                    ax.imshow(img)
                elif r == 2 and plot_overlay:
                    ax.set_title("image + concept heatmaps")
                    img = fv.get_data_sample(sample_id, preprocessing=False)[0][0].detach().cpu()
                    cond_hms = [cond_heatmap[id_hm].detach().cpu() for id_hm in range(NUM_HMS)]
                    img = overlay_heatmaps(img, cond_hms, COLORS[:NUM_HMS])
                    ax.imshow(img.numpy().transpose(1,2,0))
                else:
                    ax.axis("off")

            if c == 1:
                if r == 0:
                    ax.set_title("cond. heatmap")
                ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=True))
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):2.1f}%")

            elif (plot_overlay and c==2):
                if r == 0:
                    ax.set_title("input + cond. heatmap")
                img_ = fv.get_data_sample(sample_id, preprocessing=False)[0][0].detach().cpu()
                hm_ = cond_heatmap[r]
                img_overlay = overlay_heatmap(img_.detach().cpu(), hm_.detach().cpu(), COLORS[r])
                
                ax.imshow(img_overlay)
            elif (plot_overlay and c==3) or (not plot_overlay and c == 2):
                if r == 0 and c == 2:
                    ax.set_title("concept")
                grid = make_grid(
                    [resize(torch.from_numpy(np.asarray(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs / 2),
                    padding=0)
                grid = np.array(zimage.imgify(grid.detach().cpu()))
                img = imgify(ref_imgs[topk_ind[r]][c - 2], padding=True)
                ax.imshow(grid)
                ax.yaxis.set_label_position("right")

            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(f"{savedir}", exist_ok=True)
    fname = f"{savedir}/sample_{sample_id}_wrt_{target}_crp_{mode}"
    fname += "_overlay" if plot_overlay else ""
    print(f"store to {fname}")
    plt.savefig(f"{fname}.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
