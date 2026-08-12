import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from pathlib import Path
from typing import Tuple,Union
from sklearn.manifold import TSNE

def run_tsne(features,labels,n_subsample=10000,perplexity=30.0,random_state=42):
    N=features.shape[0]
    if n_subsample < N:
        rng=np.random.default_rng(random_state)
        idx=rng.choice(N,size=n_subsample,replace=False)
    else:
        idx=np.arange(N)
    features_sub=features[idx].copy() # (n_subsample,D)
    labels_sub=labels[idx].copy() # (n_subsample,)

    norms=np.linalg.norm(features_sub,axis=1,keepdims=True)
    norms=np.where(norms==0.0,1.0,norms)
    features_sub /=norms

    tsne=TSNE(n_components=2,perplexity=perplexity,random_state=random_state,init="pca",learning_rate="auto",max_iter=1000,metric="cosine",n_jobs=-1)
    coords_2d=tsne.fit_transform(features_sub)   # (n_subsample,2)

    return coords_2d,labels_sub

def plot_tsne(coords,labels,title,save_path):
    save_path=Path(save_path)
    save_path.parent.mkdir(parents=True,exist_ok=True)
    unique_counts=np.sort(np.unique(labels))
    n_classes=len(unique_counts)

    cmap=cm.get_cmap("tab10",n_classes)
    bounds=np.arange(unique_counts.min(),unique_counts.max() + 2) - 0.5
    norm=BoundaryNorm(bounds,n_classes)

    fig,ax=plt.subplots(figsize=(9,7))
    sc=ax.scatter(coords[:,0],coords[:,1],c=labels,cmap=cmap,norm=norm,s=3,alpha=0.5,linewidths=0,rasterized=True)
    cbar=fig.colorbar(sc,ax=ax,ticks=unique_counts,pad=0.02)
    cbar.set_label("Object count",fontsize=11)
    cbar.ax.set_yticklabels([str(c) for c in unique_counts],fontsize=9)
    ax.set_title(title,fontsize=13,fontweight="bold",pad=12)
    ax.set_xlabel("t-SNE dimension 1",fontsize=10)
    ax.set_ylabel("t-SNE dimension 2",fontsize=10)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.set_aspect("equal",adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved to {save_path}")