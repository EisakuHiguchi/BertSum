import models.data_loader as data_loader
from models.trainer import build_trainer
from pytorch_pretrained_bert import BertConfig
from models.model_builder import Summarizer
from others.logging import logger, init_logger

import torch
import os

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def preprocess(ex, is_test, use_interval):
    src = ex["src"]
    if("labels" in ex):
        labels = ex["labels"]
    else:
        labels = ex["src_sent_labels"]

    segs = ex["segs"]
    if(not use_interval):
        segs = [0]*len(segs)
    clss = ex["clss"]
    src_txt = ex["src_txt"]
    tgt_txt = ex["tgt_txt"]

    if(is_test):
        return src, labels, segs, clss, src_txt, tgt_txt
    else:
        return src, labels, segs, clss


def getArgs(
    encoder="classifier",
    model_path="../models/",
    result_path="../results/cnndm",
    temp_dir="../temp",
    bert_config_path="../bert_config_uncased_base.json",
    batch_size=1000,
    use_interval=True,
    ff_size=512,
    heads=4,
    inter_layers=2,
    rnn_size=512,
    param_init=0,
    param_init_glorot=True,
    dropout=0.1,
    save_checkpoint_steps=5,
    accum_count=2,
    world_size=1,
    report_every=1,
    recall_eval=False,
    visible_gpus="-1",
    gpu_ranks="0",
    log_file="../logs/cnndm.log",
    test_from="",
    report_rouge=False,
    block_trigram=True,
    source_data=""
):

    args = {}
    # choices=["classifier","transformer","rnn","baseline"])
    args["encoder"] = encoder
    args["model_path"] = model_path
    args["result_path"] = result_path
    args["temp_dir"] = temp_dir
    args["bert_config_path"] = bert_config_path

    args["batch_size"] = batch_size  # 1000

    args["use_interval"] = use_interval
    args["ff_size"] = ff_size  # 512
    args["heads"] = heads  # 4
    args["inter_layers"] = inter_layers
    args["rnn_size"] = rnn_size

    args["param_init"] = param_init
    args["param_init_glorot"] = param_init_glorot

    args["dropout"] = dropout

    args["save_checkpoint_steps"] = save_checkpoint_steps
    args["accum_count"] = accum_count
    args["world_size"] = world_size
    args["report_every"] = report_every

    args["recall_eval"] = recall_eval
    args["visible_gpus"] = visible_gpus
    args["gpu_ranks"] = gpu_ranks

    args["log_file"] = log_file
    args["test_from"] = test_from
    args["report_rouge"] = report_rouge
    args["block_trigram"] = block_trigram

    args["source_data"] = source_data

    class Args():
        def __init__(self, attrs):
            for k, v in attrs.items():
                setattr(self, k, v)

    args = Args(args)
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(",")]

    return args

def get_input_data(source_data, device, use_interval):
    model = torch.load(source_data)
    DATA = []
    for k in model[0].keys():
        DATA.append(model[0][k])
    DATA = preprocess(model[0], is_test=True, use_interval=use_interval)

    return data_loader.Batch(data=[DATA], device=device, is_test=True)

def get_trainer(args):
    device = "cpu" if args.visible_gpus == "-1" else "cuda"
    device_id = 0 if device == "cuda" else -1

    pt = args.test_from
    if (pt != ""):
        test_from = pt
    else:
        test_from = args.test_from

    print("Loading checkpoint from %s" % test_from)
    checkpoint = torch.load(
        test_from, map_location=lambda storage, loc: storage)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False,
                       bert_config=config)
    model.load_cp(checkpoint)
    model.eval()

    return build_trainer(args, device_id, model, None)

def generate_summary(args, step=100):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == "-1" else "cuda"

    source_data = args.source_data
    example = get_input_data(source_data, device, args.use_interval)
    trainer = get_trainer(args)

    return trainer.example_api(example=example, step=step, device=device)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-encoder", default="classifier", type=str,
                        choices=["classifier", "transformer", "rnn", "baseline"])
    #parser.add_argument("-mode", default="train", type=str,
    #                    choices=["train", "validate", "test"])
    #parser.add_argument("-bert_data_path", default="../bert_data/cnndm")
    parser.add_argument("-model_path", default="../models/")
    parser.add_argument("-result_path", default="../results/cnndm")
    parser.add_argument("-temp_dir", default="../temp")
    parser.add_argument("-bert_config_path",
                        default="../bert_config_uncased_base.json")

    parser.add_argument("-batch_size", default=1000, type=int)

    parser.add_argument("-use_interval", type=str2bool,
                        nargs="?", const=True, default=True)
    #parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=512, type=int)
    parser.add_argument("-heads", default=4, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)
    parser.add_argument("-rnn_size", default=512, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool,
                        nargs="?", const=True, default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    #parser.add_argument("-optim", default="adam", type=str)
    #parser.add_argument("-lr", default=1, type=float)
    #parser.add_argument("-beta1", default=0.9, type=float)
    #parser.add_argument("-beta2", default=0.999, type=float)
    #parser.add_argument("-decay_method", default="", type=str)
    #parser.add_argument("-warmup_steps", default=8000, type=int)
    #parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    #parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool,
                        nargs="?", const=True, default=False)

    parser.add_argument("-visible_gpus", default="-1", type=str)
    parser.add_argument("-gpu_ranks", default="0", type=str)
    parser.add_argument("-log_file", default="../logs/cnndm.log")
    #parser.add_argument("-dataset", default="")
    #parser.add_argument("-seed", default=666, type=int)

    #parser.add_argument("-test_all", type=str2bool,
    #                    nargs="?", const=True, default=False)
    parser.add_argument("-test_from", default="")
    #parser.add_argument("-train_from", default="")
    parser.add_argument("-report_rouge", type=str2bool,
                        nargs="?", const=True, default=False)
    parser.add_argument("-block_trigram", type=str2bool,
                        nargs="?", const=True, default=True)
    parser.add_argument("-source_data", default="")


    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(",")]
    
    result = generate_summary(args)
    print(result)






