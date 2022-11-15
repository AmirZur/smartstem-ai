"""Dataloading for Omniglot."""
import itertools
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import dataset, sampler, dataloader

import util

NUM_TRAIN_CLASSES = 800
NUM_VAL_CLASSES = 188
NUM_TEST_CLASSES = 100
NUM_SAMPLES_PER_CLASS = 20

SEED = 42

class OpenstaxTestDataset(dataset.Dataset):
    def __init__(self, course_name, num_support, num_query, tokenizer, max_length=128) -> None:
        super().__init__()

        self.num_support = num_support
        self.num_query = num_query
        self.tokenizer = tokenizer
        self.max_length = max_length

        # load questions from Openstax Dataset
        # columns: question, learning_goal, course (all text)
        # multiple learning goals per question, multiple questions per course
        data = util.load_openstax_course(course_name)

        # group data by question
        # columns: question (str), learning_goal (list), course (list of single str)
        self.data_by_question = data.groupby('question').agg(list)

        # group data by learning goal
        # columns: question (list), learning_goal (str), course (list of single str)
        self.data_by_learning_goal = data.groupby('learning_goal').agg(list)

    def _tokenize(self, x):
        return self.tokenizer(
            x,
            return_tensors='pt',
            max_length=self.max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

    def __getitem__(self, question_index):
        question = self.data_by_question.iloc[question_index].name

        tasks = []
        for learning_goal in self.data_by_learning_goal.index:
            # select examples that match the sampled learning goal
            examples_1 = self.data_by_learning_goal.loc[learning_goal].question
            support_1 = np.random.default_rng(seed=SEED).choice(examples_1, self.num_support)

            # select examples that do not have this learning goal
            examples_0 = self.data_by_question.drop(examples_1).index
            support_0 = np.random.default_rng(seed=SEED).choice(examples_0, self.num_support)

            support = list(support_0) + list(support_1)
            query = [question]
            labels_support = ([0] * self.num_support) + ([1] * self.num_support)
            labels_query = [int(question in examples_1)]

            if self.tokenizer:
                support, query = self._tokenize(support), self._tokenize(query)
            labels_support, labels_query = torch.tensor(labels_support), torch.tensor(labels_query)
            tasks.append(
                (support, labels_support, query, labels_query)
            )

        return tasks

    def __len__(self) -> int:
        return self.data_by_question.shape[0]


class OpenstaxDataset(dataset.Dataset):
    _COURSES = [
        'Chemistry 2e', 
        'University Physics Volume 1', 
        'University Physics Volume 2', 
        'University Physics Volume 3'
    ]

    def __init__(self, num_support, num_query, tokenizer, max_length=128) -> None:
        super().__init__()

        self.num_support = num_support
        self.num_query = num_query
        self.tokenizer = tokenizer
        self.max_length = max_length

        # load questions from Openstax Dataset
        # columns: question, learning_goal, course (all text)
        # multiple learning goals per question, multiple questions per course
        data = pd.concat([
            util.load_openstax_course(course) for course in self._COURSES
        ])

        # group data by question
        # columns: question (str), learning_goal (list), course (list of single str)
        self.data_by_question = data.groupby('question').agg(list)

        # group data by learning goal
        # columns: question (list), learning_goal (str), course (list of single str)
        self.data_by_learning_goal = data.groupby('learning_goal').agg(list)

    def _tokenize(self, x):
        return self.tokenizer(
            x,
            return_tensors='pt',
            max_length=self.max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )

    def __getitem__(self, index_and_seed):
        index, seed = index_and_seed

        # get learning goal from index
        learning_goal = self.data_by_learning_goal.iloc[index]

        # select examples that match the sampled learning goal
        examples_1 = learning_goal.question
        support_1 = np.random.default_rng(seed=seed).choice(examples_1, self.num_support)
        query_1 = np.random.default_rng(seed=seed).choice(examples_1, self.num_query)

        # select examples that do not have this learning goal
        examples_0 = self.data_by_question.drop(learning_goal.question).index
        support_0 = np.random.default_rng(seed=seed).choice(examples_0, self.num_support)
        query_0 = np.random.default_rng(seed=seed).choice(examples_0, self.num_query)

        support = list(support_0) + list(support_1)
        query = list(query_0) + list(query_1)
        labels_support = ([0] * self.num_support) + ([1] * self.num_support)
        labels_query = ([0] * self.num_query) + ([1] * self.num_query)

        if self.tokenizer:
            support, query = self._tokenize(support), self._tokenize(query)
        labels_support, labels_query = torch.tensor(labels_support), torch.tensor(labels_query)

        return support, labels_support, query, labels_query

    def __len__(self) -> int:
        return self.data_by_learning_goal.shape[0]


class OmniglotDataset(dataset.Dataset):
    """Omniglot dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './omniglot_resized'
    _GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'

    def __init__(self, num_support, num_query):
        """Inits OmniglotDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()


        # if necessary, download the Omniglot dataset
        if not os.path.isdir(self._BASE_PATH):
            gdd.GoogleDriveDownloader.download_file_from_google_drive(
                file_id=self._GDD_FILE_ID,
                dest_path=f'{self._BASE_PATH}.zip',
                unzip=True
            )

        # get all character folders
        self._character_folders = glob.glob(
            os.path.join(self._BASE_PATH, '*/*/'))
        assert len(self._character_folders) == (
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )

        # shuffle characters
        np.random.default_rng(0).shuffle(self._character_folders)

        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._character_folders[class_idx], '*.png')
            )
            sampled_file_paths = np.random.default_rng().choice(
                all_file_paths,
                size=self._num_support + self._num_query,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query


class OpenstaxSampler(sampler.Sampler):
    def __init__(self, split_idx, num_tasks, num_seeds) -> None:
        super().__init__(None)
        self._num_tasks = num_tasks
        self._indices = np.array(list(itertools.product(split_idx, range(num_seeds))))
    
    def __iter__(self):
        return (
            self._indices[np.random.default_rng(seed=SEED).choice(self._indices.shape[0], replace=False)]
            for _ in range(self._num_tasks)
        )
    
    def __len__(self):
        return self._num_tasks


class OmniglotSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks

def identity(x):
    return x

def get_openstax_dataloader(
    split,
    batch_size,
    num_support,
    num_query,
    num_tasks_per_epoch,
    tokenizer,
    num_workers=8,
    max_length=128,
    num_folds=10
):
    if split == 'train':
        split_idxs = range(NUM_TRAIN_CLASSES)
    elif split == 'val':
        split_idxs = range(NUM_TRAIN_CLASSES, NUM_TRAIN_CLASSES + NUM_VAL_CLASSES)
    else:
        split_idxs = range(NUM_TRAIN_CLASSES + NUM_VAL_CLASSES, NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES)
    
    return dataloader.DataLoader(
        dataset=OpenstaxDataset(num_support, num_query, tokenizer, max_length=max_length),
        batch_size=batch_size,
        sampler=OpenstaxSampler(split_idxs, num_tasks_per_epoch, num_folds),
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


def get_omniglot_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch,
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if split == 'train':
        split_idxs = range(NUM_TRAIN_CLASSES)
    elif split == 'val':
        split_idxs = range(
            NUM_TRAIN_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES
        )
    elif split == 'test':
        split_idxs = range(
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=OmniglotDataset(num_support, num_query),
        batch_size=batch_size,
        sampler=OmniglotSampler(split_idxs, num_way, num_tasks_per_epoch),
        num_workers=8,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
