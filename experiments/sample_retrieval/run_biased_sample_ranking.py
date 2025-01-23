
from argparse import ArgumentParser
import os
import copy
import torch
import numpy as np

from datasets import load_dataset
from experiments.sample_retrieval.plot_distribution_figure import create_data_annotation_plot
from models import get_fn_model_loader, get_canonizer
from zennit.composites import EpsilonPlusFlat
import tqdm
from sklearn import metrics
from crp.attribution import CondAttribution
from utils.cav_utils import get_cav_from_model
from utils.helper import get_features, get_features_and_relevances, load_config
import pandas as pd 
from scipy.stats import kendalltau
import wandb
from sklearn.decomposition import NMF

EVAL_NMF = False
EVAL_NEURONS = False

def get_parser():

    parser = ArgumentParser()
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--artifacts_file", type=str, default="data/artifact_samples/artifact_samples_isic.json")
    parser.add_argument("--artifact", type=str, default="band_aid")
    parser.add_argument("--fraction", default=1, type=float)
    parser.add_argument("--no_wandb", default=True, type=bool)
    parser.add_argument("--config_file", 
                        default="config_files/revealing/chexpert/local/vgg16_binaryTarget-Cardiomegaly_pm_features.22.yaml")
    parser.add_argument('--savedir', default='plot_files/data_annotation/')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(f"Run CAV-based artifact ranking for {args.config_file}")

    config = load_config(args.config_file)

    if args.no_wandb:
        config["wandb_api_key"] = None

    if args.artifacts_file is not None:
        config["artifacts_file"] = args.artifacts_file

    run_cav_artifact_ranking(config, args.artifact, args.fraction, args.batch_size, args.savedir)

def get_ranking_metrics(y, pred):
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    auc = metrics.auc(fpr, tpr)
    ap = metrics.average_precision_score(y, pred)
    kendall_tau = kendalltau(pred, y)
    return auc, ap, kendall_tau

def run_cav_artifact_ranking(config, artifact, fraction, batch_size, savedir):

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['config_name'], project=config['wandb_project_name'], resume=True)

    ## Load Data
    dataset = load_dataset(config, normalize_data=True)

    ## CAV Scope
    config["cav_scope"] = None

    # Load Model
    model = get_fn_model_loader(config["model_name"])(n_class=len(dataset.classes), ckpt_path=config["ckpt_path"], device=device).to(device)
    model = model.eval()
    attribution = CondAttribution(model)

    canonizers = get_canonizer(config["model_name"])
    composite = EpsilonPlusFlat(canonizers)
    N_COMPONENTS = 64

    # config["direction_mode"] = "signal"
    cav_a = get_cav_from_model(model, dataset, config, artifact, mode="cavs_max").float().to(device)
    cav_r = get_cav_from_model(model, dataset, config, artifact, mode="rel").float().to(device)

    split_set = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test
        }
    
    best_n_validation_auc, best_nmf_validation_auc = None, None
    best_n_validation_ap, best_nmf_validation_ap = None, None
    best_n_validation_kendall_tau, best_nmf_validation_kendall_tau = None, None
    best_n_validation_auc_rel, best_nmf_validation_auc_rel = None, None
    best_n_validation_ap_rel, best_nmf_validation_ap_rel = None, None
    best_n_validation_kendall_tau_rel, best_nmf_validation_kendall_tau_rel = None, None

    # train: 
    #   - find unsupervised directions
    # val:
    #   - pick best neuron
    #   - pick best unsupervised direction 
    # test
    #   - report all metrics

    for split in [
        # "train",
        # "val",
        # "test",
        "all"    
        ]:
        if split == "test" and EVAL_NEURONS:
            assert best_n_validation_auc is not None, "Run 'val' first to pick best neuron"
        if split == "test" and EVAL_NMF:
            assert best_nmf_validation_auc is not None, "Run 'val' first to pick best neuron"

        dataset_split = dataset if split == "all" else dataset.get_subset_by_idxs(split_set[split])

        np.random.seed(42)
        all_sample_ids = np.arange(len(dataset_split))
        idxs_art = sorted(dataset_split.sample_ids_by_artifact[artifact])
        clean_sample_ids = [i for i in all_sample_ids if i not in idxs_art]
        idxs_clean = sorted(np.random.choice(clean_sample_ids, size=int(len(clean_sample_ids)*fraction), replace=False))
        
        idxs_relevant = sorted(idxs_clean + idxs_art)
        print(f"Using {len(idxs_relevant)} samples")

        num_batches = (len(idxs_relevant) // batch_size)
        num_batches += 0 if len(idxs_relevant) % batch_size == 0 else 1

        ## Collect latent activations projected onto CAV
        fname_cav_scores = f"{config['dir_precomputed_data']}/cav_scores/{config['dataset_name']}_{split}_{config['model_name']}/{config['layer_name']}_{artifact}_{fraction}.pth"
        x_proj_cav_all = []
        rels_proj_cav_all = []
        x_latent_all = None
        rels_latent_all = None
        for idx_batch in tqdm.tqdm(range(num_batches)):
            x_batch = torch.stack([dataset_split[idx][0] for idx in idxs_relevant[idx_batch * batch_size:(idx_batch+1) * batch_size]])
            x_latent, rels_latent = get_features_and_relevances(x_batch.to(device), config, attribution)
            x_proj_cav = (x_latent@cav_a[None, :].T).cpu()   
            rels_proj_cav = (rels_latent@cav_r[None, :].T).cpu()
            x_proj_cav_all.append(x_proj_cav)
            rels_proj_cav_all.append(rels_proj_cav)
            x_latent_all = x_latent.cpu() if x_latent_all is None else torch.concat([x_latent_all, x_latent.cpu()])
            rels_latent_all = rels_latent.cpu() if rels_latent_all is None else torch.concat([rels_latent_all, rels_latent.cpu()])

        x_latent_all = x_latent_all.clamp(min=0)
        rels_latent_all = rels_latent_all.clamp(min=0)
        x_proj_cav_all = torch.cat(x_proj_cav_all)
        rels_proj_cav_all = torch.cat(rels_proj_cav_all)
        
        
        os.makedirs(os.path.dirname(fname_cav_scores), exist_ok=True)
        torch.save(x_proj_cav_all.reshape(-1), fname_cav_scores)

        if EVAL_NMF:
            ## fit NMF
            fname_nmf_directions = f"{config['dir_precomputed_data']}/unsupervised_cavs/{config['dataset_name']}_train_{config['model_name']}/{config['layer_name']}_{artifact}_{fraction}.pth"
            os.makedirs(os.path.dirname(fname_nmf_directions), exist_ok=True)
            if split in ("all", "train"):
                print("Fitting NMF")
                nmf_model = NMF(n_components=N_COMPONENTS, random_state=0).fit(x_latent_all.numpy())
                nmf_directions = torch.from_numpy(nmf_model.components_)
                torch.save(nmf_directions, fname_nmf_directions)
            else:
                print("Loading NMF")
                nmf_directions = torch.load(fname_nmf_directions)

        artifact_labels = [1 if idx in dataset_split.sample_ids_by_artifact[artifact] else 0 for idx in idxs_relevant]

        ## Summarize data in pd dataframe for seaborn
        data_pd = pd.DataFrame({
            "value": x_proj_cav_all.squeeze(),
            "value_rels": rels_proj_cav_all.squeeze(),
            "artifact_label": artifact_labels,
            "idx": idxs_relevant
        }).set_index("idx")

        ## Identify samples to visualize
        idxs_clean_sorted = data_pd[data_pd.artifact_label == 0].sort_values("value").index.values
        idxs_art_sorted = data_pd[data_pd.artifact_label == 1].sort_values("value").index.values

        idx_clean_1, idx_clean_50, idx_clean_99 = idxs_clean_sorted[len(idxs_clean_sorted) // 500], idxs_clean_sorted[len(idxs_clean_sorted) // 2], idxs_clean_sorted[-len(idxs_clean_sorted)//500]
        idxs_interesting_clean = [idx_clean_1, idx_clean_50, idx_clean_99]
        if len(idxs_art_sorted) > 0:
            idx_art_1, idx_art_50, idx_art_99 = idxs_art_sorted[len(idxs_art_sorted) // 500], idxs_art_sorted[len(idxs_art_sorted) // 2], idxs_art_sorted[-len(idxs_art_sorted) // 500]
            idxs_interesting_art = [idx_art_1, idx_art_50, idx_art_99]
        else:
            idxs_interesting_art = []

        ## Compute metrics (CAV)
        scores_clean = data_pd[data_pd['artifact_label'] == 0].value.values
        scores_rel_clean = data_pd[data_pd['artifact_label'] == 0].value_rels.values
        scores_art = data_pd[data_pd['artifact_label'] == 1].value.values
        scores_rel_art = data_pd[data_pd['artifact_label'] == 1].value_rels.values

        pred = np.concatenate([scores_art, scores_clean])
        pred_rel = np.concatenate([scores_rel_art, scores_rel_clean])
        y = np.concatenate([np.ones_like(scores_art), np.zeros_like(scores_clean)])
        print(f"Pos: {len(scores_art)}, Neg: {len(scores_clean)}")

        auc, ap, kendall_tau = get_ranking_metrics(y, pred)
        auc_rel, ap_rel, kendall_tau_rel = get_ranking_metrics(y, pred_rel)

        if EVAL_NEURONS:
            ## Compute metrics per neuron
            x_latent_clean = x_latent_all[data_pd['artifact_label'] == 0]
            rels_latent_clean = rels_latent_all[data_pd['artifact_label'] == 0]
            x_latent_art = x_latent_all[data_pd['artifact_label'] == 1]
            rels_latent_art = rels_latent_all[data_pd['artifact_label'] == 1]
            
            neuron_aps, neuron_aucs, neuron_kendall_taus, neuron_kendall_tau_pvalues = [], [], [], []
            neuron_aps_rel, neuron_aucs_rel, neuron_kendall_taus_rel, neuron_kendall_tau_pvalues_rel = [], [], [], []
            
            for cid in range(x_latent_all.shape[1]):
                n_pred = torch.concat([x_latent_art[:,cid], x_latent_clean[:,cid]])
                n_pred_rel = torch.concat([rels_latent_art[:,cid], rels_latent_clean[:,cid]])
                
                n_auc, n_ap, n_kendall_tau = get_ranking_metrics(y, n_pred)
                n_auc_rel, n_ap_rel, n_kendall_tau_rel = get_ranking_metrics(y, n_pred_rel)

                neuron_aps.append(n_ap)
                neuron_aps_rel.append(n_ap_rel)
                neuron_aucs.append(n_auc)
                neuron_aucs_rel.append(n_auc_rel)
                neuron_kendall_taus.append(n_kendall_tau.statistic)
                neuron_kendall_taus_rel.append(n_kendall_tau_rel.statistic)
                neuron_kendall_tau_pvalues.append(n_kendall_tau.pvalue)
                neuron_kendall_tau_pvalues_rel.append(n_kendall_tau_rel.pvalue)

        if EVAL_NMF:
            nmf_dirs_aps, nmf_dirs_aucs, nmf_dirs_kendall_taus, nmf_dirs_kendall_tau_pvalues = [], [], [], []
            nmf_dirs_aps_rel, nmf_dirs_aucs_rel, nmf_dirs_kendall_taus_rel, nmf_dirs_kendall_tau_pvalues_rel = [], [], [], []
            
            ## Compute metrics per direction
            for dir_id in range(len(nmf_directions)):
                scores_direction = torch.concat([
                    x_latent_art@nmf_directions[dir_id], 
                    x_latent_clean@nmf_directions[dir_id]
                    ])
                scores_direction_rel = torch.concat([
                    rels_latent_art@nmf_directions[dir_id], 
                    rels_latent_clean@nmf_directions[dir_id]
                    ])
                nmf_dir_auc, nmf_dir_ap, nmf_dir_kendall_tau = get_ranking_metrics(y, scores_direction)
                nmf_dir_auc_rel, nmf_dir_ap_rel, nmf_dir_kendall_tau_rel = get_ranking_metrics(y, scores_direction_rel)

                nmf_dirs_aps.append(nmf_dir_ap)
                nmf_dirs_aps_rel.append(nmf_dir_ap_rel)
                nmf_dirs_aucs.append(nmf_dir_auc)
                nmf_dirs_aucs_rel.append(nmf_dir_auc_rel)
                nmf_dirs_kendall_taus.append(nmf_dir_kendall_tau.statistic)
                nmf_dirs_kendall_taus_rel.append(nmf_dir_kendall_tau_rel.statistic)
                nmf_dirs_kendall_tau_pvalues.append(nmf_dir_kendall_tau.pvalue)
                nmf_dirs_kendall_tau_pvalues_rel.append(nmf_dir_kendall_tau_rel.pvalue)

        if split in ["train", "val", "all"]:
            # pick best neuron / NMF direction for train/val set
            if EVAL_NEURONS:
                best_n_validation_auc = np.argmax(neuron_aucs)
                best_n_validation_ap = np.argmax(neuron_aps)
                best_n_validation_kendall_tau = np.argmax(neuron_kendall_taus)
                best_n_validation_auc_rel = np.argmax(neuron_aucs_rel)
                best_n_validation_ap_rel = np.argmax(neuron_aps_rel)
                best_n_validation_kendall_tau_rel = np.argmax(neuron_kendall_taus_rel)

            if EVAL_NMF:
                best_nmf_validation_auc = np.argmax(nmf_dirs_aucs)
                best_nmf_validation_ap = np.argmax(nmf_dirs_aps)
                best_nmf_validation_kendall_tau = np.argmax(nmf_dirs_kendall_taus)
                best_nmf_validation_auc_rel = np.argmax(nmf_dirs_aucs_rel)
                best_nmf_validation_ap_rel = np.argmax(nmf_dirs_aps_rel)
                best_nmf_validation_kendall_tau_rel = np.argmax(nmf_dirs_kendall_taus_rel)


        metrics_all = {
            f"{split}_{artifact}_cav_auc": auc,
            f"{split}_{artifact}_cav_rel_auc": auc_rel,
            f"{split}_{artifact}_cav_ap": ap,
            f"{split}_{artifact}_cav_rel_ap": ap_rel,
            f"{split}_{artifact}_cav_kendall_tau": kendall_tau.statistic,
            f"{split}_{artifact}_cav_rel_kendall_tau": kendall_tau_rel.statistic,
            f"{split}_{artifact}_cav_kendall_tau_pvalue": kendall_tau.pvalue,
            f"{split}_{artifact}_cav_rel_kendall_tau_pvalue": kendall_tau_rel.pvalue
            }
        
        if EVAL_NEURONS:
            metrics_all = {
                **metrics_all,
                f"{split}_{artifact}_best_n_ap": neuron_aps[best_n_validation_ap],
                f"{split}_{artifact}_best_n_rel_ap": neuron_aps_rel[best_n_validation_ap_rel],
                f"{split}_{artifact}_best_n_auc": neuron_aucs[best_n_validation_auc],
                f"{split}_{artifact}_selected_n_auc": best_n_validation_auc,
                f"{split}_{artifact}_selected_n_ap": best_n_validation_ap,
                f"{split}_{artifact}_best_n_rel_auc": neuron_aucs_rel[best_n_validation_auc_rel],
                f"{split}_{artifact}_best_n_kendall_tau": neuron_kendall_taus[best_n_validation_kendall_tau],
                f"{split}_{artifact}_best_n_rel_kendall_tau": neuron_kendall_taus_rel[best_n_validation_kendall_tau_rel],
                f"{split}_{artifact}_best_n_kendall_tau_pvalue": neuron_kendall_tau_pvalues[best_n_validation_kendall_tau],
                f"{split}_{artifact}_best_n_rel_kendall_tau_pvalue": neuron_kendall_tau_pvalues_rel[best_n_validation_kendall_tau_rel],
            }
        if EVAL_NMF:
            metrics_all = {
                **metrics_all,
                f"{split}_{artifact}_best_nmf_ap": nmf_dirs_aps[best_nmf_validation_ap],
                f"{split}_{artifact}_best_nmf_rel_ap": nmf_dirs_aps_rel[best_nmf_validation_ap_rel],
                f"{split}_{artifact}_best_nmf_auc": nmf_dirs_aucs[best_nmf_validation_auc],
                f"{split}_{artifact}_best_nmf_rel_auc": nmf_dirs_aucs_rel[best_nmf_validation_auc_rel],
                f"{split}_{artifact}_best_nmf_kendall_tau": nmf_dirs_kendall_taus[best_nmf_validation_kendall_tau],
                f"{split}_{artifact}_best_nmf_rel_kendall_tau": nmf_dirs_kendall_taus_rel[best_nmf_validation_kendall_tau_rel],
                f"{split}_{artifact}_best_nmf_kendall_tau_pvalue": nmf_dirs_kendall_tau_pvalues[best_nmf_validation_kendall_tau],
                f"{split}_{artifact}_best_nmf_rel_kendall_tau_pvalue": nmf_dirs_kendall_tau_pvalues_rel[best_nmf_validation_kendall_tau_rel],
            }

        print(metrics_all)
        if config.get('wandb_api_key', None):
            
            wandb.log({**metrics_all, **config})
            print("Logging metrics!")

        if "vit" in config["model_name"]:
            img_size=dataset_split[0][0].shape[1]
            localizations = torch.ones(6,img_size,img_size)
        else:
            localizations_clean = get_localizations(idxs_interesting_clean, cav_a.cpu(), dataset_split, attribution, composite, config, device)
            localizations_art = get_localizations(idxs_interesting_art, cav_a.cpu(), dataset_split, attribution, composite, config, device)
            localizations = torch.cat([localizations_clean, localizations_art])

        savename = f"{savedir}/distribution/{config['dataset_name']}/{config['model_name']}/{artifact}_{config['layer_name']}_{split}.pdf"

        create_data_annotation_plot(data_pd, dataset_split, idxs_interesting_clean, idxs_interesting_art, 
                                localizations, plot_connections=True, savename=savename)
        
        create_data_annotation_plot(data_pd, dataset_split, idxs_interesting_clean, idxs_interesting_art, 
                                localizations, plot_connections=False, savename=savename[:-4] + f"_no_connections.pdf")
        
        # ## Create precision-recall plots
        # savedir_precision_recall = f"{savedir}/precision_recall/{config['dataset_name']}/{config['model_name']}"
        # savename_precision_recall_cav = f"{savedir_precision_recall}/cav_{artifact}_{config['layer_name']}_{split}.png"
        # savename_precision_recall_n = f"{savedir_precision_recall}/best_n_{artifact}_{config['layer_name']}_{split}.png"
        # os.makedirs(savedir_precision_recall, exist_ok=True)

        # display_cav = PrecisionRecallDisplay.from_predictions(y, pred)
        # display_cav.figure_.savefig(savename_precision_recall_cav, bbox_inches="tight", dpi=300)

        # best_n = best_n_validation_ap
        # display_best_n = PrecisionRecallDisplay.from_predictions(y, torch.concat([x_latent_art[:,best_n], x_latent_clean[:,best_n]]))
        # display_best_n.figure_.savefig(savename_precision_recall_n, bbox_inches="tight", dpi=300)

        if split == "train":
            # reset
            best_n_validation_auc, best_nmf_validation_auc = None, None
            best_n_validation_ap, best_nmf_validation_ap = None, None
            best_n_validation_kendall_tau, best_nmf_validation_kendall_tau = None, None
            best_n_validation_auc_rel, best_nmf_validation_auc_rel = None, None
            best_n_validation_ap_rel, best_nmf_validation_ap_rel = None, None
            best_n_validation_kendall_tau_rel, best_nmf_validation_kendall_tau_rel = None, None


def get_localizations(idxs, cav, dataset_split, attribution, composite, config, device):
    if len(idxs) == 0:
        return torch.tensor([])
    
    config1 = copy.deepcopy(config)
    config1["cav_mode"] = "cavs_full"

    x = torch.stack([dataset_split[idx][0] for idx in idxs])
    act = get_features(x.to(device), config1, attribution).detach().cpu()
    init_rel = (act.clamp(min=0) * cav[..., None, None]).to(device)
    attr = attribution(x.to(device).requires_grad_(), [{}], composite, start_layer=config["layer_name"], init_rel=init_rel)
    hms = attr.heatmap.detach().cpu().clamp(min=0)
    return hms

if __name__ == "__main__":
    main()
