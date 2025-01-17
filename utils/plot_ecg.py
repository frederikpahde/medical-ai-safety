import numpy as np
import torch
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import utils.ecg_utils as ecg_utils

default_channel_axis = {
        'I':[1,0], 
        'II':[3,0], 
        'III':[5,0], 
        'AVR':[2,0], 
        'AVL':[0,0], 
        'AVF':[4,0], 
        'V1':[0,1], 
        'V2':[1,1], 
        'V3':[2,1], 
        'V4':[3,1],
        'V5':[4,1], 
        'V6':[5,1]
    }

def _heatmap_modifier_ecg(self, data, on_device=None):
    heatmap = data.grad.detach()
    heatmap = heatmap.to(on_device) if on_device else heatmap
    return heatmap

def plot_reference_img(d, a, rf=True, color="k", lw=2, offset=2, qmax=.99, qmin=0.2, 
                       crop_th=0.001, mult=1, width_ratio=3):
    fig, ax = plt.subplots(1, 1, figsize=(width_ratio*mult,1*mult))

    if a is not None:
        a_filter = torch.from_numpy(gaussian_filter1d(a, sigma=5))
        amax = np.quantile(abs(a_filter.flatten()),q=qmax)
        amin = np.quantile(abs(a_filter.flatten()),q=qmin)
        filtered_heat = a_filter / a_filter.clamp(min=0).max()
        if rf:
            imin, imax = get_crop_range(filtered_heat, crop_th)
            xvalues = np.array(range(imin, imax))
            d = d[imin:imax]
            a_filter = a_filter[imin:imax]
        else:
            xvalues = np.array(range(len(d)))
    else:
        a_filter = None
        xvalues = np.array(range(len(d)))
        amin, amax = None, None

    plot_beat_with_hm(d, a_filter, xvalues, ax, amin, amax, color, lw, offset)
    plt.tight_layout();
    plt.subplots_adjust(wspace=0, hspace=0);
    img = to_image(ax).copy()
    plt.close()
    return img

def to_image(ax):
    ax.figure.canvas.draw()
    image = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
    return image

def get_crop_range(heatmap, crop_th):
    idxs = torch.where(heatmap > crop_th)[0]
    if len(idxs) == 0:
        return 0, -1
    imin, imax = int(idxs.min()), int(idxs.max())
    if imin == imax:
        return 0, -1
    return imin, imax

def remove_axis(ax):
    ax.set_yticks([]); ax.set_xticks([]); 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def plot_beat_with_hm(d, a, xvalues, ax, amin, amax, color, lw, offset, x_offset=25):
    # Plot Beat
    ax.plot(xvalues, d, c=color, lw=lw, zorder=9, alpha=1);

    # Plot HM
    if a is not None:
        norm = Normalize(vmin=amin, vmax=amax)
        cmap = plt.cm.Reds
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap = ListedColormap(my_cmap)

        xy = np.vstack([xvalues, d]).T
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])
        coll = LineCollection(segments, cmap=my_cmap, norm=norm, linewidths=lw, zorder=10)
        coll.set_array(a);
        ax.add_collection(coll);

    x_offset = 25
    ax.set_ylim(-offset, offset)
    ax.set_xlim(-x_offset, 1000+x_offset)
    ax.set_yticks([])
    ax.set_xticks([])

def plot_ecg_concept(data, cid, lead_id, rf=True, qmax=.99, qmin=0.2, lw=2, offset=2, color="k", 
                     vis_th=0.05, crop_th=0.001, img_scale=1):
    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5*img_scale, 2*img_scale))
    for i in range(len(data[cid][0])):
        d = data[cid][0][i][lead_id]

        a = data[cid][1][i][lead_id]
        a_filter = torch.from_numpy(gaussian_filter1d(a, sigma=5))
        amax = np.quantile(abs(a_filter.flatten()),q=qmax)
        amin = np.quantile(abs(a_filter.flatten()),q=qmin)
        filtered_heat = a_filter / a_filter.clamp(min=0).max()
        # vis_mask = filtered_heat > vis_th

        ax = axs[i // ncols, i % ncols]

        if rf:
            imin, imax = get_crop_range(filtered_heat, crop_th)
            if imax <= imin:
                xvalues = np.array(range(len(d)))
            else:
                xvalues = np.array(range(imin, imax))
                d = d[imin:imax]
                a_filter = a_filter[imin:imax]
        else:
            xvalues = np.array(range(len(d)))

        # print(xvalues)
        # if len(xvalues) == 0:
        #     print("PROBLEM")
        plot_beat_with_hm(d, a_filter, xvalues, ax, amin, amax, color, lw, offset)
        
    plt.tight_layout();
    plt.subplots_adjust(wspace=0, hspace=0);
    return to_image(ax)

def vis_attributions_full(x, attr, offset=2, qlower=.05, qupper=.95, 
                          grid=True, ax=None, with_attributions=True):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plot_beat(
        x,  
        ax, 
        "",
        offset=offset,
        color='k', alpha=1, lw=2, border=True, annotate=True)
    if with_attributions:
        plot_heat(
            x,
            attr,
            ax,
            'laber',
            offset=offset, annotate=False)

    if grid:
        grid_color='gray'
        grid_alpha=.15
        # mV
        for mv in np.arange(-6*offset,offset//2,.1):
            ax.axhline(mv, c=grid_color, lw=1, alpha=grid_alpha)
        for mv in np.arange(-6*offset,offset//2,.5):
            ax.axhline(mv, c=grid_color, lw=2, alpha=grid_alpha)
        # ms
        for ms in np.arange(0,160,2):
            ax.axvline(ms, c=grid_color, lw=1, alpha=grid_alpha)
        for ms in np.arange(0,160,10):
            ax.axvline(ms, c=grid_color, lw=2, alpha=grid_alpha)
                
        ax.axvline(30, c=grid_color,zorder=1,lw=2, alpha=.5)
        ax.axvline(110, c=grid_color,zorder=1,lw=2, alpha=.5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
def plot_beat(
    x,
    ax,
    label,
    lower=None,
    upper=None,
    color='k',
    alpha=1,
    annotate=True,
    offset=4,
    lw=4,
    border=True,
    grid=False,
    channel_axis=default_channel_axis
):
    
    
    
    length = x.shape[0]
    pad=.1
    for i, channel in enumerate(x.T):
        [iy,ix] = channel_axis[ecg_utils.leads[i].upper()]
        iy = -iy
        if ecg_utils.leads[i].upper() == 'AVR':
            ax.plot(np.array(range(len(channel)))+ix*length, -channel+iy*offset, c=color, lw=lw, zorder=9, label=label, alpha=alpha)
            if not lower is None:
                ax.fill_between(np.array(range(len(channel)))+ix*length, -upper.T[i]+iy*offset, -lower.T[i]+iy*offset, facecolor=color, alpha=0.25, zorder=8)
            if annotate:
                ax.text(ix*length+40*pad, iy*offset+offset/2-pad,  '-'+ecg_utils.leads[i], va='top', ha='left', fontweight='bold')#, bbox=dict(facecolor='.8', edgecolor='k', pad=pad, alpha=1., zorder=10))
        else:
            ax.plot(np.array(range(len(channel)))+ix*length, channel+iy*offset, c=color, lw=lw, zorder=9, alpha=alpha)
            if not lower is None:
                ax.fill_between(np.array(range(len(channel)))+ix*length, upper.T[i]+iy*offset, lower.T[i]+iy*offset, facecolor=color, alpha=0.25, zorder=8)
            if annotate:
                ax.text(ix*length+40*pad, iy*offset+offset/2-pad, ecg_utils.leads[i], va='top', ha='left', fontweight='bold')#, bbox=dict(facecolor='.8', edgecolor='k', pad=pad, alpha=1., zorder=10))

                
    if border:
        border_color='k'
        lwg = 2
        ax.axvline(0, c=border_color,zorder=5, lw=lwg)
        ax.axvline(length, c=border_color,zorder=5, lw=lwg)
        ax.axvline(2*length, c=border_color,zorder=5, lw=lwg)
        for iy in range(7):
            ax.axhline(-iy*offset+offset/2, c=border_color,zorder=5, lw=lwg)
    ax.set_xlim(-lwg/10.,2*length)
    ax.set_ylim(-5*offset-offset/2, offset/2)
    if grid:
        grid_color='gray'
        grid_alpha=.15
        # mV
        for mv in np.arange(-6*offset,offset//2,.1):
            ax.axhline(mv, c=grid_color, lw=1, alpha=grid_alpha)
        for mv in np.arange(-6*offset,offset//2,.5):
            ax.axhline(mv, c=grid_color, lw=2, alpha=grid_alpha)
        # ms
        for ms in np.arange(0,length,2):
            ax.axvline(ms, c=grid_color, lw=1, alpha=grid_alpha)
        for ms in np.arange(0,length,10):
            ax.axvline(ms, c=grid_color, lw=2, alpha=grid_alpha)
        ax.axvline(30, c=grid_color,zorder=1,lw=2, alpha=.5)
        ax.axvline(110, c=grid_color,zorder=1,lw=2, alpha=.5)

def plot_heat(
    x,
    a,
    ax,
    label,
    color='b',
    alpha=1,
    annotate=True,
    offset=4,
    lw =2,
    channel_axis=default_channel_axis,
    border=False, qmax=.99, qmin=0.2):
    
    
    cmap = plt.cm.Reds
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap = ListedColormap(my_cmap)
    
    channel_itos = ecg_utils.leads
    length = x.shape[0]
    dom = np.array(range(length))
    
    amax = np.quantile(abs(a.flatten()),q=qmax)
    amin = np.quantile(abs(a.flatten()),q=qmin)
    norm = Normalize(vmin=amin, vmax=amax)
    
    for i, channel in enumerate(x.T):
        [iy,ix] = channel_axis[ecg_utils.leads[i].upper()]
        iy = -iy
        if ecg_utils.leads[i].upper() == 'AVR':
            xy = np.vstack([dom+ix*length, -channel+iy*offset]).T
            xy = xy.reshape(-1, 1, 2)
            segments = np.hstack([xy[:-1], xy[1:]])
            coll = LineCollection(segments, cmap=my_cmap, norm=norm, linewidths=lw, zorder=10)
            coll.set_array(a[:,i])
            ax.add_collection(coll)
        else:
            xy = np.vstack([dom+ix*length, channel+iy*offset]).T
            xy = xy.reshape(-1, 1, 2)
            segments = np.hstack([xy[:-1], xy[1:]])
            coll = LineCollection(segments, cmap=my_cmap, norm=norm, linewidths=lw, zorder=10)
            coll.set_array(a[:,i])
            ax.add_collection(coll)
        if border:
            # BORDER
            border_color='k'
            lwg = 4
            ax.axvline(0, c=border_color,zorder=10, lw=lwg)
            ax.axvline(length, c=border_color,zorder=10, lw=lwg)
            ax.axvline(2*length, c=border_color,zorder=10, lw=lwg)
            ax.axhline(iy*offset+offset/2, c=border_color,zorder=7, lw=lwg)
            ax.axhline(iy*offset-offset/2, c=border_color,zorder=7, lw=lwg)
            ax.set_xlim(-lwg/10.,2*length + lwg/10.)
            ax.set_ylim(iy*offset-offset//2 -lwg/200., offset//2+ lwg/200.)