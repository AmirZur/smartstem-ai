"""Implementation of prototypical networks for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard
from transformers import AutoTokenizer, BertModel
import sklearn.metrics as metrics

import openstax_dataset

import util

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 1000


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, model, learning_rate, log_dir):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """
        self._device = DEVICE

        self._network = model.to(self._device)
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _predict(self, task_batch):
        predictions_batch = []
        for task in task_batch:
            support, labels_support, query, labels_query = task

            support = {k: v.to(self._device) for k, v in support.items()}
            query = {k: v.to(self._device) for k, v in query.items()}
            labels_support = labels_support.to(self._device)
            labels_query = labels_query.to(self._device)
            n = 2
            k = labels_support.shape[0] // n
            # (nk, dim)
            support_representations = self._network(
                **support
            )[1]

            # (nq, dim)
            query_representations = self._network(
                **query
            )[1]

            # (n, dim)
            prototypes = support_representations.view(n, k, -1).mean(dim=1)

            # (nq, n) 
            query_distances = torch.cdist(query_representations, prototypes) 
            query_logits = F.softmax(-query_distances, dim=1)

            predictions_batch.append(torch.argmax(query_logits, dim=-1))
        return torch.stack(predictions_batch)

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            support, labels_support, query, labels_query = task

            support = {k: v.to(self._device) for k, v in support.items()}
            query = {k: v.to(self._device) for k, v in query.items()}
            labels_support = labels_support.to(self._device)
            labels_query = labels_query.to(self._device)
            n = 2
            k = labels_support.shape[0] // n
            # (nk, dim)
            support_representations = self._network(
                **support
            )[1]

            # (nq, dim)
            query_representations = self._network(
                **query
            )[1]

            # (n, dim)
            prototypes = support_representations.view(n, k, -1).mean(dim=1)

            # (nq, n) 
            query_distances = torch.cdist(query_representations, prototypes) 
            query_logits = F.softmax(-query_distances, dim=1)

            support_distances = torch.cdist(support_representations, prototypes) 
            support_logits = F.softmax(-support_distances, dim=1)

            loss = F.cross_entropy(query_logits, labels_query)
            accuracy_support = util.score(support_logits, labels_support)
            accuracy_query = util.score(query_logits, labels_query)

            loss_batch.append(loss)
            accuracy_support_batch.append(accuracy_support)
            accuracy_query_batch.append(accuracy_query)
        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for val_task_batch in dataloader_val:
                        loss, accuracy_support, accuracy_query = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy_support:.3f}, '
                    f'query accuracy: {accuracy_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    accuracy_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )
    
    def test_on_course(self, dataloader_test):
        accuracies = []
        f1_scores = []
        for i, task_batch in enumerate(dataloader_test):
            if i >= 10:
                break
            predictions = self._predict(task_batch).squeeze(1).cpu().numpy()
            labels_query = np.array([task[-1][0] for task in task_batch], dtype=np.int64)
            print(predictions.dtype, labels_query.dtype)
            print(np.unique(labels_query), np.unique(predictions))

            f1_scores.append(metrics.f1_score(y_true=labels_query, y_pred=predictions))
            accuracies.append((predictions == labels_query).sum() / len(predictions))
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_95_confidence_interval_acc = 1.96 * std_accuracy / np.sqrt(len(accuracies))
        print(
            f'Accuracy over {len(accuracies)} test questions: '
            f'mean {mean_accuracy * 100:.3f}'
            f'95% confidence interval {mean_95_confidence_interval_acc * 100:.3f}'
        )
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_95_confidence_interval_f1 = 1.96 * std_f1 / np.sqrt(len(accuracies))
        print(
            f'F1 score over {len(accuracies)} test questions: '
            f'mean {mean_f1 * 100:.3f}'
            f'95% confidence interval {mean_95_confidence_interval_f1 * 100:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = 1 # checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/openstax.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.lr_{args.learning_rate}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    tokenizer = AutoTokenizer.from_pretrained(f'prajjwal1/bert-{args.model_size}')
    model = BertModel.from_pretrained(f'prajjwal1/bert-{args.model_size}')

    protonet = ProtoNet(model, args.learning_rate, log_dir)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = (args.num_train_iterations - args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}',
            f'device={protonet._device}'
        )
        dataloader_train = openstax_dataset.get_openstax_dataloader(
            split='train',
            batch_size=args.batch_size,
            num_support=args.num_support,
            num_query=args.num_query,
            num_tasks_per_epoch=num_training_tasks,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
            max_length=args.max_length,
            sample_by_learning_goal=args.sample_by_learning_goal
        )
        dataloader_val = openstax_dataset.get_openstax_dataloader(
            split='val',
            batch_size=args.batch_size,
            num_support=args.num_support,
            num_query=args.num_query,
            num_tasks_per_epoch=args.batch_size * 4,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
            max_length=args.max_length,
            sample_by_learning_goal=args.sample_by_learning_goal
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        if args.course_name is not None:
            print(f'Testing on tagging for course {args.course_name}')
            dataset_test = openstax_dataset.OpenstaxTestDataset(
                course_name=args.course_name,
                num_support=args.num_support,
                num_query=args.num_query,
                tokenizer=tokenizer,
                max_length=args.max_length
            )
            protonet.test_on_course(dataset_test)
        else:
            print(
                f'Testing on tasks with composition '
                f'num_support={args.num_support}, '
                f'num_query={args.num_query}'
            )
            dataloader_test = openstax_dataset.get_openstax_dataloader(
                split='test',
                batch_size=args.batch_size,
                num_support=args.num_support,
                num_query=args.num_query,
                num_tasks_per_epoch=NUM_TEST_TASKS,
                tokenizer=tokenizer,
                num_workers=args.num_workers,
                max_length=args.max_length,
                sample_by_learning_goal=args.sample_by_learning_goal
            )
            protonet.test(dataloader_test)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--max_length', type=int, default=128,
                        help='maximum tokenized sequence length')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers to use for data loading')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--course_name', type=str, default=None,
                        help='Course to test on (only applies if --test flag is true)')
    parser.add_argument('--model_size', type=str, default='small', 
                        help='Size of (bert) model to use.')
    parser.add_argument('--sample_by_learning_goal', default=False, action='store_true',
                        help='Set true to sample by learning goal instead of by question')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)
