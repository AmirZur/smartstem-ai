"""Implementation of prototypical networks for labeling questions with learning objectives."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard
from transformers import AutoTokenizer, BertModel
from models.protobert import BertModel as BertTAMModel
import sklearn.metrics as metrics

import openstax_dataset

from tqdm import tqdm

import util

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 1000


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(
        self, 
        model, 
        learning_rate, 
        log_dir,
        num_epochs=5,
        gradient_accumulation_steps=1,
        max_grad_norm=None,
        early_stopping=False,
        n_iter_no_change=3,
        tolerance=1e-5,
        device=DEVICE
    ):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """
        self._device = device

        self._network = model.to(self._device)
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_epoch = 0
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tolerance = tolerance
        self.no_improvement_count = 0
        self.best_loss = np.inf
        self.best_score = -np.inf

    def _predict(self, task_batch):
        with torch.no_grad():
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

                predictions_batch.append(query_logits[:, 1].cpu().numpy())
        return np.stack(predictions_batch)

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
        print(f'Starting training at epoch {self._start_train_epoch}.')
        self._network.to(self._device)

        self._optimizer.zero_grad()

        for epoch in range(self._start_train_epoch + 1, self._start_train_epoch + self.num_epochs + 1):
            epoch_loss = 0

            self._network.train()
            with tqdm(dataloader_train, desc=f'Epoch {epoch}') as progress_bar:
                for i_step, task_batch in enumerate(progress_bar):
                    loss, accuracy_support, accuracy_query = self._step(task_batch)

                    if self.gradient_accumulation_steps > 1 and self.loss.reduction == "mean":
                        loss /= self.gradient_accumulation_steps
                    
                    loss.backward()

                    epoch_loss += loss.item()

                    if i_step % self.gradient_accumulation_steps == 0 or i_step == len(dataloader_train):
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self._network.parameters(), self.max_grad_norm
                            )
                        self._optimizer.step()
                        self._optimizer.zero_grad()
                    
                    progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy_query.item())

                    if i_step % PRINT_INTERVAL == 0:
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
            
            epoch_loss = epoch_loss / len(progress_bar)

            # Validation
            val_loss, val_accuracy_support, val_accuracy_query = self.score(dataloader_val)            
            print(
                f'Validation: '
                f'training loss = {epoch_loss:.3f}, '
                f'validation loss = {val_loss:.3f}, '
                f'support accuracy = {val_accuracy_support:.3f}, '
                f'query accuracy = {val_accuracy_query:.3f}'
            )
            writer.add_scalar('loss/val', val_loss, i_step)
            writer.add_scalar(
                'val_accuracy/support',
                val_accuracy_support,
                i_step
            )
            writer.add_scalar(
                'val_accuracy/query',
                val_accuracy_query,
                i_step
            )

            if self.early_stopping:
                self._update_no_improvement_count_early_stopping(val_accuracy_query, epoch)
                if self.no_improvement_count > self.n_iter_no_change:
                    print(
                        f"Stopping after epoch {epoch}. Validation score did "
                        f"not improve by tol={self.tolerance} for more than {self.n_iter_no_change} epochs. "
                        f"Final error is {epoch_loss}"
                    )
                    break
            else:
                self._update_no_improvement_count_early_stopping(epoch_loss, epoch)
                if self.no_improvement_count > self.n_iter_no_change:
                    print(
                        f"Stopping after epoch {epoch}. Training loss did "
                        f"not improve by tol={self.tolerance} for more than {self.n_iter_no_change} epochs. "
                        f"Final error is {epoch_loss}"
                    )
                    break
    
    def _update_no_improvement_count_early_stopping(self, val_accuracy_query, epoch):
        if val_accuracy_query < (self.best_score + self.tolerance):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        if val_accuracy_query > self.best_score:
            self.best_score = val_accuracy_query
            self._save(epoch)

    def _update_no_improvement_count_early_stopping(self, loss, epoch):
        if loss > (self.best_loss - self.tolerance):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        if loss < self.best_loss:
            self.best_loss = loss
            self._save(epoch)

    def score(self, dataloader_val):
        self._network.eval()
        with torch.no_grad():
            losses, accuracies_support, accuracies_query = [], [], []
            for val_task_batch in tqdm(dataloader_val, desc='Validation'):
                loss, accuracy_support, accuracy_query = (
                    self._step(val_task_batch)
                )
                losses.append(loss.item())
                accuracies_support.append(accuracy_support)
                accuracies_query.append(accuracy_query)
            loss = np.mean(losses)
            accuracy_support = np.mean(accuracies_support)
            accuracy_query = np.mean(accuracies_query)
        return loss, accuracy_support, accuracy_query

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        self._network.eval()
        with torch.no_grad():
            # accuracies = []
            # for task_batch in tqdm(dataloader_test, desc='Testing'):
            #     accuracies.append(self._step(task_batch)[2])
            # mean = np.mean(accuracies)
            # std = np.std(accuracies)
            # mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
            # print(
            #     f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            #     f'mean {mean:.5f}, '
            #     f'95% confidence interval {mean_95_confidence_interval:.3f}'
            # )
            predictions_batch = []
            labels_batch = []
            for task_batch in tqdm(dataloader_test, desc='Testing'):
                predictions_batch.append(self._predict(task_batch).squeeze())
                labels_batch.append([task[-1].cpu().numpy() for task in task_batch])
                
            predictions = np.stack(predictions_batch).reshape(NUM_TEST_TASKS, -1)
            labels = np.stack(labels_batch).reshape(NUM_TEST_TASKS, -1)

            aucs = np.array([
                metrics.roc_auc_score(y_score=p, y_true=l, average='macro') for p, l in zip(predictions, labels) 
                if sum(l == 1) > 0 and sum(l == 0) > 0
            ])
            auc_cfi = aucs.std() * 1.96 / np.sqrt(len(aucs))

            accuracies = np.array([
                metrics.accuracy_score(y_true=l, y_pred=(p >= 0.5)) for p, l in zip(predictions, labels) 
                if sum(l == 1) > 0 and sum(l == 0) > 0
            ])
            accuracy_cfi = accuracies.std() * 1.96 / np.sqrt(len(accuracies))

            f1s = np.array([
                metrics.f1_score(y_pred=(p >= 0.5), y_true=l, average='macro') for p, l in zip(predictions, labels) 
                if sum(l == 1) > 0 and sum(l == 0) > 0
            ])
            f1_cfi = f1s.std() * 1.96 / np.sqrt(len(f1s))
            
            print(f'ROC-AUC: {aucs.mean():.3f} +- {auc_cfi:.3f}')
            print(f'Accuracy: {accuracies.mean():.3f} +- {accuracy_cfi:.3f}')
            print(f'F1: {f1s.mean():.3f} +- {f1_cfi:.3f}')
            
        return aucs.mean(), auc_cfi, accuracies.mean(), accuracy_cfi, f1s.mean(), f1_cfi
            
    def test_on_course(self, dataloader_test, num_questions=None, return_preds=False):
        """
        Evaluate prototypical network on held-out course.
        Unlike held-out test set, held-out course consists of l different
        tasks for each question, where l is the number of total learning objectives.
        A question is tagged with all learning goals to which the model assigns 
        a positive prediction for the (question, learning goal) pair.
        """
        with torch.no_grad():
            predictions_batch = []
            labels_batch = []
            for i, task_batch in enumerate(tqdm(dataloader_test, desc='Tagging Course')):
                if num_questions is not None and i >= num_questions:
                    break
                predictions = self._predict(task_batch).squeeze()
                labels_query = np.array([task[-1].item() for task in task_batch], dtype=np.int64)
                predictions_batch.append(predictions)
                labels_batch.append(labels_query)
            predictions = np.stack(predictions_batch)
            labels = np.stack(labels_batch)

            aucs = np.array([
                metrics.roc_auc_score(y_score=p, y_true=l, average='macro') for p, l in zip(predictions, labels) 
                if sum(l == 1) > 0 and sum(l == 0) > 0
            ])
            auc_cfi = aucs.std() * 1.96 / np.sqrt(len(aucs))

            accuracies = np.array([
                metrics.accuracy_score(y_true=l, y_pred=(p >= 0.5)) for p, l in zip(predictions, labels) 
                if sum(l == 1) > 0 and sum(l == 0) > 0
            ])
            accuracy_cfi = accuracies.std() * 1.96 / np.sqrt(len(accuracies))

            f1s = np.array([
                metrics.f1_score(y_pred=(p >= 0.5), y_true=l, average='macro') for p, l in zip(predictions, labels) 
                if sum(l == 1) > 0 and sum(l == 0) > 0
            ])
            f1_cfi = f1s.std() * 1.96 / np.sqrt(len(f1s))
            
            print(f'ROC-AUC: {aucs.mean():.3f} +- {auc_cfi:.3f}')
            print(f'Accuracy: {accuracies.mean():.3f} +- {accuracy_cfi:.3f}')
            print(f'F1: {f1s.mean():.3f} +- {f1_cfi:.3f}')
           
        if return_preds:
            return predictions
        return aucs.mean(), auc_cfi, accuracies.mean(), accuracy_cfi, f1s.mean(), f1_cfi

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

    def save_pretrained(self, name):
        # save network using Huggingface Transformers library
        self._network.save_pretrained(name)
    
    def load_pretrained(self, classname, name):
        # save network using Huggingface Transformers library
        self._network = classname.from_pretrained(name)


def main(args):
    # load log directory
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/openstax.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.lr_{args.learning_rate}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(util.get_model_name(args.model_size))
    if args.task_embedding_model_type is not None:
        model = BertTAMModel.from_pretrained(util.get_model_name(args.model_size), is_tam=True)
    else:
        model = BertModel.from_pretrained(util.get_model_name(args.model_size))

    # construct prototypical network trainer
    protonet = ProtoNet(
        model, 
        args.learning_rate, 
        log_dir,
        num_epochs=args.num_epochs
    )

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        # training run
        print(
            f'Training on tasks with composition '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = openstax_dataset.get_nway_kshot_dataloader(
            split='train',
            batch_size=args.batch_size,
            num_support=args.num_support,
            num_query=args.num_query,
            num_tasks_per_epoch=args.train_size,
            tokenizer=tokenizer,
            task_embedding_model_type=args.task_embedding_model_type,
            num_workers=args.num_workers,
            max_length=args.max_length,
            balanced=not args.imbalanced,
            sample_by_learning_goal=args.sample_by_learning_goal
        )
        dataloader_val = openstax_dataset.get_nway_kshot_dataloader(
            split='val',
            batch_size=args.batch_size,
            num_support=args.num_support,
            num_query=args.num_query,
            num_tasks_per_epoch=args.validation_size,
            tokenizer=tokenizer,
            task_embedding_model_type=args.task_embedding_model_type,
            num_workers=args.num_workers,
            max_length=args.max_length,
            balanced=not args.imbalanced,
            sample_by_learning_goal=args.sample_by_learning_goal
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        # test run (divided into testing on held-out course or held-out dataset)
        if args.course_name is not None:
            print(f'Testing on tagging for course {args.course_name}')
            dataset_test = openstax_dataset.CourseTestDataset(
                course_name=args.course_name,
                num_support=args.num_support,
                num_query=args.num_query,
                tokenizer=tokenizer,
                max_length=args.max_length
            )
            protonet.test_on_course(dataset_test, args.num_questions)
        else:
            print(
                f'Testing on tasks with composition '
                f'num_support={args.num_support}, '
                f'num_query={args.num_query}'
            )
            dataloader_test = openstax_dataset.get_nway_kshot_dataloader(
                split=args.split,
                batch_size=args.batch_size,
                num_support=args.num_support,
                num_query=args.num_query,
                num_tasks_per_epoch=NUM_TEST_TASKS,
                tokenizer=tokenizer,
                task_embedding_model_type=args.task_embedding_model_type,
                num_workers=args.num_workers,
                max_length=args.max_length,
                balanced=not args.imbalanced,
                sample_by_learning_goal=args.sample_by_learning_goal
            )
            protonet.test(dataloader_test)

    protonet.save_pretrained(args.log_dir + 'pretrained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--model_size', type=str, default='small', 
                        help='Size of (bert) model to use.')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--max_length', type=int, default=256,
                        help='maximum tokenized sequence length')
    parser.add_argument('--train_size', type=int, default=None,
                        help='size of training set (if none, uses all learning goals, otherwise samples)')
    parser.add_argument('--validation_size', type=int, default=None,
                        help='size of validation set (if none, uses all learning goals, otherwise samples)')
    parser.add_argument('--sample_by_learning_goal', default=False, action='store_true',
                        help='Set true to sample by learning goal instead of by question')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of epochs to train for')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers to use for data loading')
    parser.add_argument('--task_embedding_model_type', type=str, default=None,
                        help='if supplied, the SBERT model to use for task embeddings')
    parser.add_argument('--imbalanced', default=False, action='store_true', help='balance data or sample from distribution')
    # Testing and loading models
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--course_name', type=str, default=None,
                        help='Course to test on (only applies if --test flag is true)')
    parser.add_argument('--num_questions', type=int, default=None,
                        help='Number of questions to tag from test course.')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--split', type=str, default='test',
                        help='Choose data split for testing. Can be one of train, test, or val.')
    
    main_args = parser.parse_args()
    main(main_args)
