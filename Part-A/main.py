import argparse
import contextlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_clip
import train_dino
from dataset import CLEVRCaptionDataset_Aa,CLEVRProbeDataset,SimpleTokenizer
from models.clip import CLIPModel,TextTransformer
from models.vit import VisionTransformer
from probe.linear_probe import LinearProbe,build_backbone,build_probe_transform,extract_features,train_probe,eval_probe
from probe.tsne_viz import run_tsne,plot_tsne
from probe.retrieval import configure_retrieval,run_retrieval,build_clip_image_transform

@contextlib.contextmanager
def _patch_argv(args_list):
    old=sys.argv[:]
    sys.argv=[""] + list(args_list)
    try:
        yield
    finally:
        sys.argv=old

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _make_loader(ds,batch_size,workers,shuffle=False):
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=workers,pin_memory=True)

def cmd_train_clip(remaining):
    with _patch_argv(remaining):
        train_clip.main()

def cmd_train_dino(remaining):
    with _patch_argv(remaining):
        train_dino.main()

# probe (Table 1)
_PROBE_MODELS=[("clip","clip_ckpt","tokenizer","CLIP"),
    ("dino_student","dino_ckpt",None,"DINO Student"),
    ("dino_teacher","dino_ckpt",None,"DINO Teacher")]

def _get_or_extract(model_kind,ckpt_attr,tok_attr,task,split,args,device,loader):
    feat_dir=Path(args.output_dir)/"probe"
    feat_dir.mkdir(parents=True,exist_ok=True)
    cache=feat_dir/f"{model_kind}_{task}_{split}.npz"
    if cache.exists() and not args.force_extract:
        print(f"    [cache] {cache.name}")
        d=np.load(cache)
        return d["cls"],d["gap"],d["y"]
    ckpt_path=getattr(args,ckpt_attr)
    tok_path=getattr(args,tok_attr) if tok_attr else None
    if not ckpt_path:
        raise FileNotFoundError("Checkpoint not provided!")
    print(f"    [extract] backbone={model_kind} task={task} split={split} ...")
    backbone=build_backbone(kind=model_kind,ckpt_path=ckpt_path,tokenizer_path=tok_path,max_len=args.max_len).to(device)
    backbone.eval()
    cls_f,gap_f,labels=extract_features(backbone,loader,device)
    np.savez_compressed(cache,cls=cls_f,gap=gap_f,y=labels)
    print(f"    [cache] saved -> {cache}")
    return cls_f,gap_f,labels

def cmd_probe(args):
    device=_device()
    transform=build_probe_transform(224)
    out_dir=Path(args.output_dir)/"probe"
    out_dir.mkdir(parents=True,exist_ok=True)
    table={}# table[model_kind][task][emb_name]={"train":score,"val":score}
    for model_kind,ckpt_attr,tok_attr,display_name in _PROBE_MODELS:
        if not getattr(args,ckpt_attr):
            print(f"\n[probe] Skipping {display_name}:"
                  f"--{ckpt_attr.replace('_','-')} not provided.")
            continue
        print(f"\n[probe]==={display_name}===")
        table.setdefault(model_kind,{})
        for task in ("count","colors"):
            print(f"  task={task}")
            train_ds=CLEVRProbeDataset(root=args.probe_data_root,task=task,split="train",transform=transform)
            val_ds=CLEVRProbeDataset(root=args.probe_data_root,task=task,split="val",transform=transform)
            train_loader=_make_loader(train_ds,args.batch_size,args.workers)
            val_loader=_make_loader(val_ds,args.batch_size,args.workers)

            tr_cls,tr_gap,tr_y=_get_or_extract(model_kind,ckpt_attr,tok_attr,task,"train",args,device,train_loader)
            va_cls,va_gap,va_y=_get_or_extract(model_kind,ckpt_attr,tok_attr,task,"val",args,device,val_loader)
            num_classes=train_ds.num_classes
            table[model_kind].setdefault(task,{})
            for emb_name,tr_x,va_x in (("cls",tr_cls,va_cls),("gap",tr_gap,va_gap),):
                probe=LinearProbe(input_dim=tr_x.shape[1],num_classes=num_classes).to(device)
                probe=train_probe(probe,tr_x,tr_y,task=task,epochs=args.probe_epochs,lr=args.probe_lr)
                tr_score=eval_probe(probe,tr_x,tr_y,task=task)
                va_score=eval_probe(probe,va_x,va_y,task=task)
                table[model_kind][task][emb_name]={"train":tr_score,"val":va_score}
                metric_label="acc" if task=="count" else "F1"
                print(
                    f"    [{emb_name}] train {metric_label}="
                    f"{tr_score:.4f}  val {metric_label}={va_score:.4f}"
                )
                torch.save(probe.state_dict(),out_dir/f"probe_{model_kind}_{task}_{emb_name}.pt")
    _print_table1(table)
    result_path=out_dir/"table1.json"
    with open(result_path,"w",encoding="utf-8") as fh:
        json.dump(table,fh,indent=2)
    print(f"\n[probe] Full results -> {result_path}")

def _print_table1(table):
    W=8
    header=(
        f"\n{'Model':<22}"
        f"  {'--- [CLS] Counting ---':^{2*W+1}}"
        f"  {'-- [CLS] Color Pred. --':^{2*W+1}}"
        f"  {'--- GAP Counting ---':^{2*W+1}}"
        f"  {'-- GAP Color Pred. --':^{2*W+1}}"
    )
    subrow=(
        f"{'':22}"
        + (f"  {'Train':>{W}} {'Val':>{W}}" * 4)
    )
    sep="=" * len(header)
    print()
    print(sep)
    print("TABLE 1 - Representation Analysis")
    print(sep)
    print(header)
    print(subrow)
    print("-" * len(header))
    for model_kind,_,_,display_name in _PROBE_MODELS:
        if model_kind not in table:
            continue
        row=f"{display_name:<22}"
        for emb in ("cls","gap"):
            for task in ("count","colors"):
                d=table.get(model_kind,{}).get(task,{}).get(emb,{})
                tr=d.get("train",float("nan"))
                va=d.get("val",float("nan"))
                row +=f"  {tr:>{W}.4f} {va:>{W}.4f}"
        print(row)
    print(sep)

def cmd_tsne(args):
    device=_device()
    out_dir=Path(args.output_dir)/"tsne"
    out_dir.mkdir(parents=True,exist_ok=True)
    transform=build_probe_transform(224)
    count_ds=CLEVRProbeDataset(root=args.probe_data_root,task="count",split="train",transform=transform)
    count_loader=_make_loader(count_ds,args.batch_size,args.workers)
    tsne_cfgs=[
        ("clip","clip_ckpt","tokenizer","CLIP Vision Encoder"),
        ("dino_student","dino_ckpt",None,"DINO Student"),
        ("dino_teacher","dino_ckpt",None,"DINO Teacher"),
    ]
    for model_kind,ckpt_attr,tok_attr,title in tsne_cfgs:
        ckpt_path=getattr(args,ckpt_attr)
        if not ckpt_path:
            print(f"[tsne] Skipping {title}:"
                  f"--{ckpt_attr.replace('_','-')} not provided.")
            continue
        probe_cache=(Path(args.output_dir)/"probe"/f"{model_kind}_count_train.npz")
        if probe_cache.exists() and not args.force_extract:
            print(f"[tsne] {title}:loading cached features from probe step")
            d  =np.load(probe_cache)
            cls_feats=d["cls"]
            labels=d["y"].astype(int)
        else:
            print(f"[tsne] {title}:extracting features ...")
            tok_path=getattr(args,tok_attr) if tok_attr else None
            backbone=build_backbone(kind=model_kind,ckpt_path=ckpt_path,tokenizer_path=tok_path,max_len=args.max_len).to(device)
            backbone.eval()
            cls_feats,_,labels=extract_features(backbone,count_loader,device)
            labels=labels.astype(int)
        print(
            f"[tsne] {title}:running t-SNE "
            f"(N={len(cls_feats)},subsample={args.n_subsample}) ..."
        )
        coords,lbls=run_tsne(cls_feats,labels,n_subsample=args.n_subsample)
        save_path=out_dir/f"tsne_{model_kind}.pdf"#i need to chang it to .png if needed
        plot_tsne(coords,lbls,title=title,save_path=str(save_path))

def cmd_retrieval(args):
    for flag,name in (("clip_ckpt","--clip-ckpt"),("tokenizer","--tokenizer"),("data_root","--data-root"),):
        if not getattr(args,flag):
            raise ValueError(f"{name} is required for the retrieval step.")
    device=_device()
    out_dir=Path(args.output_dir)/"retrieval"
    configure_retrieval(max_len=args.max_len,batch_size=args.batch_size,workers=args.workers,num_examples=args.num_examples,output_dir=str(out_dir))
    tokenizer=SimpleTokenizer.load(args.tokenizer)
    vit=VisionTransformer(img_size=224,patch_size=16,embed_dim=384,depth=12,num_heads=6,mlp_dim=1536)
    txt=TextTransformer(vocab_size=tokenizer.vocab_size,max_len=args.max_len,embed_dim=384,depth=6,num_heads=6,mlp_dim=1536,pad_id=SimpleTokenizer.PAD_ID)
    model=CLIPModel(vision_encoder=vit,text_encoder=txt,embed_dim=512)
    ckpt=torch.load(args.clip_ckpt,map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model=model.to(device).eval()
    val_ds=CLEVRCaptionDataset_Aa(root=args.probe_data_root,split="val",transform=build_clip_image_transform(224))
    print("[retrieval] Encoding validation set ...")
    metrics=run_retrieval(model,val_ds,tokenizer,device)

    print()
    print("[retrieval] ----------------------------------------")
    print(f"  Image -> Text   R@1={metrics['image_to_text_r1']:.4f}")
    print(f"  Image -> Text   R@3={metrics['image_to_text_r3']:.4f}")
    print(f"  Text  -> Image  R@1={metrics['text_to_image_r1']:.4f}")
    print(f"  Text  -> Image  R@3={metrics['text_to_image_r3']:.4f}")
    print(f"[retrieval] Saved to {out_dir}")

def cmd_run_all(args):
    # Run probe -> tsne -> retrieval
    print("\n" + "#" * 60)
    print("#  STEP 1/3  -  Linear Probing  (Table 1)")
    print("#" * 60)
    cmd_probe(args)

    print("\n" + "#" * 60)
    print("#  STEP 2/3  -  t-SNE Visualisation")
    print("#" * 60)
    cmd_tsne(args)

    print("\n" + "#" * 60)
    print("#  STEP 3/3  -  Cross-Modal Retrieval")
    print("#" * 60)
    cmd_retrieval(args)

    print("\n[run-all] All evaluation steps complete.")

def _add_eval_args(p):
    # Attach the shared eval arguments to a subparser.
    g=p.add_argument_group("data roots")
    g.add_argument("--probe-data-root",default=None,help="Root of Part Aa dataset  (required for probe/tsne)")
    g.add_argument("--data-root",default=None,help="Root of Part A dataset   (required for retrieval)")
    g=p.add_argument_group("checkpoints")
    g.add_argument("--clip-ckpt",default=None,help="Path to trained CLIP checkpoint (.pt)")
    g.add_argument("--dino-ckpt",default=None,help="Path to trained DINO checkpoint (.pt)")
    g.add_argument("--tokenizer",default=None,help="Path to saved tokenizer JSON")
    p.add_argument("--output-dir",default="./outputs",help="Base directory for all outputs")
    g=p.add_argument_group("data loading")
    g.add_argument("--batch-size",type=int,default=256)
    g.add_argument("--workers",type=int,default=8)
    g.add_argument("--max-len",type=int,default=40,help="Tokenizer context length (must match CLIP training)")
    g=p.add_argument_group("linear probe")
    g.add_argument("--probe-epochs",type=int,default=200,help="Epochs to train each linear probe")
    g.add_argument("--probe-lr",type=float,default=1e-2,help="AdamW learning rate for linear probes")
    g=p.add_argument_group("t-SNE")
    g.add_argument("--n-subsample",type=int,default=10000,help="Points to subsample before running t-SNE")
    g=p.add_argument_group("retrieval")
    g.add_argument("--num-examples",type=int,default=10,help="Number of example queries to save for the report")
    p.add_argument("--force-extract",action="store_true",help="Ignore cached .npz feature files and re-extract")

def build_parser():
    p=argparse.ArgumentParser(prog="main.py",description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    subs=p.add_subparsers(dest="cmd",metavar="SUBCOMMAND")
    subs.required=True
    subs.add_parser("train-clip",help="Train CLIP  (pass --help after for full option list)",add_help=False)
    subs.add_parser("train-dino",help="Train DINO  (pass --help after for full option list)",add_help=False)
    fmt=argparse.ArgumentDefaultsHelpFormatter
    for name,helpstr in (("probe","Linear-probe all models -> Table 1"),
        ("tsne","t-SNE scatter plots coloured by object count"),
        ("retrieval","CLIP cross-modal retrieval  R@1/R@3"),
        ("run-all","probe + tsne + retrieval in one shot")):
        _add_eval_args(subs.add_parser(name,help=helpstr,formatter_class=fmt))
    return p

_DISPATCH={"probe":cmd_probe,"tsne":cmd_tsne,"retrieval":cmd_retrieval,"run-all":cmd_run_all}

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("train-clip","train-dino"):
        fn=cmd_train_clip if sys.argv[1]=="train-clip" else cmd_train_dino
        fn(sys.argv[2:])
        return
    parser=build_parser()
    args=parser.parse_args()
    _DISPATCH[args.cmd](args)

if __name__=="__main__":
    main()