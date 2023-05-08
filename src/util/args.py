import sys
import os

import argparse
from pyhocon import ConfigFactory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", type=str, default=None)
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited")
    parser.add_argument("--name", "-n", type=str, default='dtu_exp', help="experiment name")
    parser.add_argument("--dataset_format", "-F", type=str, default=None, help="Dataset format, dtu for tmp")
    parser.add_argument("--logs_path", type=str, default="logs", help="logs output directory")
    parser.add_argument("--checkpoints_path", type=str, default="checkpoints", help="checkpoints output directory")
    parser.add_argument("--visual_path", type=str, default="visuals", help="visualization output directory")
    parser.add_argument("--epochs", type=int, default=10000000, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=1.00, help="learning rate decay factor")
    parser.add_argument("--datadir", "-D", type=str, default=None, help="Dataset directory")
    parser.add_argument("--ray_batch_size", "-R", type=int, default=128, help="Ray batch size")
    parser.add_argument("--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')")
    parser.add_argument("--nviews","-V",type=str,default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')")
    parser.add_argument("--freeze_enc",action="store_true",default=None,help="Freeze encoder weights and only train MLP")
    parser.add_argument("--no_bbox_step",type=int,default=100000,help="Step to stop using bbox sampling")
    parser.add_argument("--fixed_test", action="store_true", default=None, help="Freeze encoder weights and only train MLP")

    args = parser.parse_args()

    os.makedirs(os.path.join(args.checkpoints_path, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.visual_path, args.name), exist_ok=True)

    conf = ConfigFactory.parse_file(args.conf)

    args.gpu_id = list(map(int, args.gpu_id.split()))

    print("EXPERIMENT NAME:", args.name)
    print("CONTINUE?", "yes" if args.resume else "no")
    print("* Config file:", args.conf)
    print("* Dataset format:", args.dataset_format)
    print("* Dataset location:", args.datadir)
    return args, conf
