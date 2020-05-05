from text_gan.data.squad1_ca_q import Squad1_CA_Q
from text_gan.data.squad1_ca_qc import SQuAD_CA_QC
from text_gan import cfg_from_file, cfg

import argparse
import multiprocessing
import logging

DATA = [
    "squadca-q",
    "squadca-qc"
]


def parse_args():
    parser = argparse.ArgumentParser(prog="Preprocess SQuAD 1 data")
    parser.add_argument(
        "--dataset", "-d", choices=DATA,
        required=True, dest="dataset",
        help="Select dataset to preprocess")
    parser.add_argument(
        "--cfg", dest="cfg", type=str, help="Config YAML filepath",
        required=False, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    logging.basicConfig(
        level=cfg.LOG_LVL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.dataset == "squadca-q":
        _ = Squad1_CA_Q(prepare=True)
    elif args.dataset == "squadca-qc":
        _ = SQuAD_CA_QC(prepare=True)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # option to support debugging
    main()
