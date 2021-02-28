import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim

from src.models.mlp import MLP
from src.data import make_environments
from src.utils import *


def main(args):
    for k, v in sorted(vars(args).items()):
        print(f"\t{k}: {v}")

    final_train_accs = []
    final_test_accs  = []

    for restart in range(args.n_restarts):
        print("Restart", restart)

        # Load MNIST, make train/val splits and shuffle train set 
        mnist = datasets.MNIST("/mnt/local1/szchen/Datasets/MNIST/", train=True, download=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])
        # Shuffle
        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1].numpy())

        # build envs
        envs = [
            make_environments(mnist_train[0][::2], mnist_train[1][::2], 0.2, args.device),
            make_environments(mnist_train[0][1::2], mnist_train[1][1::2], 0.1, args.device),
            make_environments(mnist_val[0], mnist_val[1], 0.9, args.device)
        ]

        mlp = MLP(args.hidden_dim, args.grayscale).to(args.device)
        optimizer = optim.Adam(mlp.parameters(), lr=args.lr)

        pretty_print("step", "train null", "train acc", "train penalty", "test acc")

        for step in range(args.steps):
            for env in envs:
                logits = mlp(env["images"])
                env["nll"] = mean_nll(logits, env["labels"])
                env["acc"] = mean_accuracy(logits, env["labels"])
                env["penalty"] = penalty(logits, env["labels"], args.device)

            train_nll = torch.stack([envs[0]["nll"], envs[1]["nll"]]).mean()
            train_acc = torch.stack([envs[0]["acc"], envs[1]["acc"]]).mean()
            train_penalty = torch.stack([envs[0]["penalty"], envs[1]["penalty"]]).mean()

            weight_norm = torch.tensor(0.).to(args.device)
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += args.l2_regularizer_weight * weight_norm

            if not args.not_penalty:
                penalty_weight = (args.penalty_weight if step >= args.penalty_anneal_iters else 1.0)
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_acc = envs[2]["acc"]
            if step % 100 == 0:
                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    test_acc.detach().cpu().numpy()
                )

        final_train_accs.append(train_acc.detach().cpu().numpy())
        final_test_accs.append(test_acc.detach().cpu().numpy())
        print("Final train acc (mean/std acrossrestarts so far): ")
        print(np.mean(final_train_accs), np.std(final_train_accs))
        print("Final test acc (mean/std across restarts so far): ")
        print(np.mean(final_test_accs), np.std(final_test_accs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colored MNIST")        
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--l2-regularizer-weight", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--penalty-anneal-iters", type=int, default=100)
    parser.add_argument("--penalty-weight", type=float, default=10000.0)
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--not-penalty", action="store_true")
    args = parser.parse_args()

    main(args)