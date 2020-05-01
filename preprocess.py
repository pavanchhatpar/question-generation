from text_gan.data.squad1_ca_q import Squad1_CA_Q
from text_gan import cfg_from_file, cfg

import argparse
import multiprocessing
import logging


def parse_args():
    parser = argparse.ArgumentParser(prog="Preprocess SQuAD 1 data")
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
    _ = Squad1_CA_Q(prepare=True)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # option to support debugging
    main()
