import argparse
import os

from rnng.trainer import Trainer


def make_parser(subparsers=None) -> argparse.ArgumentParser:
    description = 'Train RNNG on a given corpus.'
    if subparsers is None:
        parser = argparse.ArgumentParser(description=description)
    else:
        parser = subparsers.add_parser('train', description=description)

    parser.add_argument(
        '-t', '--train-corpus', required=False, default='../../ptb/train-gen.oracle',
        metavar='FILE', help='path to train corpus')
    parser.add_argument(
        '-d', '--dev-corpus', required=False, default='../../ptb/dev-gen.oracle',
        metavar='FILE', help='path to dev corpus')
    parser.add_argument(
        '-s', '--save-to', required=False, default='../../model/', metavar='DIR',
        help='directory to save the training artifacts')
    parser.add_argument(
        '--encoding', default='utf-8', help='file encoding to use (default: utf-8)')
    parser.add_argument(
        '--no-lower', action='store_false', dest='lower',
        help='whether not to lowercase the words')
    parser.add_argument(
        '--min-freq', type=int, default=2, metavar='NUMBER',
        help='minimum word frequency to be included in the vocabulary (default: 2)')
    parser.add_argument(
        '--word-embedding-size', type=int, default=32, metavar='NUMBER',
        help='dimension of word embeddings (default: 32)')
    parser.add_argument(
        '--nt-embedding-size', type=int, default=60, metavar='NUMBER',
        help='dimension of nonterminal embeddings (default: 12)')
    parser.add_argument(
        '--action-embedding-size', type=int, default=16, metavar='NUMBER',
        help='dimension of action embeddings (default: 16)')
    parser.add_argument(
        '--input-size', type=int, default=128, metavar='NUMBER',
        help='input dimension of the LSTM parser state encoders (default: 128)')
    parser.add_argument(
        '--hidden-size', type=int, default=128, metavar='NUMBER',
        help='hidden dimension of the LSTM parser state encoders (default: 128)')
    parser.add_argument(
        '--num-layers', type=int, default=2, metavar='NUMBER',
        help='number of layers of the LSTM parser state encoders and composers (default: 2)')
    parser.add_argument(
        '--dropout', type=float, default=0.5, metavar='NUMBER',
        help='dropout rate (default: 0.5)')
    parser.add_argument(
        '--learning-rate', type=float, default=0.001, metavar='NUMBER',
        help='learning rate (default: 0.001)')
    parser.add_argument(
        '--max-epochs', type=int, default=20, metavar='NUMBER',
        help='maximum number of epochs to train (default: 20)')
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='NUMBER',
        help='print logs every this number of iterations (default: 1000)')
    parser.add_argument(
        '--seed', type=int, default=25122017, help='random seed (default: 25122017)')
    parser.add_argument(
        '--device', type=int, default=-1, help='GPU device to use (default: -1 for CPU)')
    parser.add_argument(
        '--load-artifacts', type=bool, default=True, help='use saved model insteaf of bulding a new one'
    )
    parser.set_defaults(func=main)

    return parser


def main(args: argparse.Namespace) -> None:
    if args.device >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = 0
    kwargs = vars(args)
    kwargs.pop('func', None)
    train_corpus = kwargs.pop('train_corpus')
    save_to = kwargs.pop('save_to')
    trainer = Trainer(train_corpus, save_to, **kwargs)
    trainer.run()


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_usage()
