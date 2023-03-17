import argparse

import datasets

parser = argparse.ArgumentParser(description='Augment Huggingface datasets')
parser.add_argument('--dataset', type=str, help='HuggingFace dataset', required=True)