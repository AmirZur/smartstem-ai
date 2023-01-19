"""Dataloading for custom learning objective dataset."""
import numpy as np
import pandas as pd

import torch
from torch.utils.data import dataset, sampler, dataloader

from sentence_transformers import SentenceTransformer

import util

FRAC_TRAIN_CLASSES = 0.8
FRAC_VAL_CLASSES = 0.1
FRAC_TEST_CLASSES = 0.1

SEED = 42

class GPT3Dataset(dataset.Dataset):
    """
    Data wrapper for GPT-3 embeddings, for held-out test set.
    Paralel to nWaykShotDataset
    """

    _OPENSTAX_COURSES = [
        'Chemistry 2e', 
        'University Physics Volume 1', 
        'University Physics Volume 2', 
        'University Physics Volume 3'
    ]

    _PRINCIPLES_OF_CHEMISTRY_COURSE = 'Principles of Chemistry 3rd edition'
    _CHEM31A_COURSE = 'Chem 31A'

    def __init__(self, num_support, num_query, zero_shot=False, seed=SEED) -> None:
        super().__init__()

        self.num_support = num_support
        self.num_query = num_query
        self.seed = seed
        # flag whether we're using learning goal embeddings
        self.tam = False

        # load questions from Openstax Dataset
        # columns: question, learning_goal, course (all text)
        # multiple learning goals per question, multiple questions per course
        data = pd.concat([
            util.load_openstax_course(course) for course in self._OPENSTAX_COURSES
        ] + [util.load_principles_of_chemistry_course(self._PRINCIPLES_OF_CHEMISTRY_COURSE)])
        # data = util.load_chem31a_course()

        # group data by question
        # dictionary mapping from course name to dataframe of questions within this course
        # columns: question (str), learning_goal (list), course (list of single str)
        self.data_by_question = {
            course_name: data[data['course'] == course_name].groupby('question').agg(list)
            for course_name in data['course'].unique()
        }

        # group data by learning goal
        # columns: question (list), learning_goal (str), course (list of single str)
        self.data_by_learning_goal = data.groupby('learning_goal').agg(list)

        learning_goal_embeddings = torch.load('learning_goal_curie_embeddings.pt')
        self.learning_goal_to_embedding = dict(zip(
            self.data_by_learning_goal.index, learning_goal_embeddings
        ))

        # ignore learning goals that do not have enough training examples
        # NOTE: questions under these learning goals can still appear as NEGATIVE examples, just not positive examples
        self.data_by_learning_goal = self.data_by_learning_goal[
            self.data_by_learning_goal['question'].apply(len) >= self.num_support + self.num_query
        ]
        # shuffle order of learning goals for training!!
        self.data_by_learning_goal = self.data_by_learning_goal.sample(frac=1., random_state=self.seed)

        question_embeddings = torch.load('question_curie_embeddings.pt')
        self.question_to_embedding = dict(zip(
            [q for key in self.data_by_question for q in self.data_by_question[key].index], question_embeddings
        ))

        self.zero_shot = zero_shot

        # construct a random number generator
        self.rng = np.random.default_rng(seed=self.seed)  

    def get_data_by_learning_goal(self):
        return self.data_by_learning_goal

    def __getitem__(self, index):
        if self.zero_shot:
            learning_goal = self.data_by_learning_goal.iloc[index]
            query_1 = self.rng.choice(
                learning_goal.question, self.num_query, replace=False
            )
            examples_0 = self.data_by_question[learning_goal.course[0]].drop(learning_goal.question)
            query_0 = self.rng.choice(
                examples_0.index, 
                self.num_query,
                replace=False
            )
            learning_goals_0 = [self.rng.choice(lgs) for lgs in examples_0.loc[query_0].learning_goal.values]

            learning_goals_1 = [learning_goal.name] * self.num_query

            query = list(query_0) + list(query_1)
            learning_goals = learning_goals_0 + learning_goals_1
            labels = ([0] * self.num_query) + ([1] * self.num_query)

            query = torch.stack([self.question_to_embedding[q] for q in query])
            learning_goals = torch.stack([self.learning_goal_to_embedding[l] for l in learning_goals])
            labels = torch.tensor(labels)

            return learning_goals, query, labels

        # get learning goal from index
        learning_goal = self.data_by_learning_goal.iloc[index]

        # select examples that match the sampled learning goal
        examples_1 = learning_goal.question
        support_and_query_1 = self.rng.choice(
            examples_1, self.num_support + self.num_query, replace=False
        )
        support_1 = support_and_query_1[:self.num_support]
        query_1 = support_and_query_1[self.num_support:]

        # select examples that do not have this learning goal
        examples_0 = self.data_by_question[learning_goal.course[0]].drop(learning_goal.question).index
        support_and_query_0 = self.rng.choice(
            examples_0, self.num_support + self.num_query, replace=False
        )
        support_0 = support_and_query_0[:self.num_support]
        query_0 = support_and_query_0[self.num_support:]

        support = list(support_0) + list(support_1)
        query = list(query_0) + list(query_1)
        labels_support = ([0] * self.num_support) + ([1] * self.num_support)
        labels_query = ([0] * self.num_query) + ([1] * self.num_query)

        support = torch.stack([self.question_to_embedding[q] for q in support])
        query = torch.stack([self.question_to_embedding[q] for q in query])

        labels_support, labels_query = torch.tensor(labels_support), torch.tensor(labels_query)

        return support, labels_support, query, labels_query

    def __len__(self) -> int:
        return self.data_by_learning_goal.shape[0]


class GPT3TestDataset(dataset.Dataset):
    """
    Dataset wrapper for GPT-3 embeddings for held-out course.
    Parallel to CourseTestDataset
    """

    _OPENSTAX_COURSES = [
        'Chemistry 2e', 
        'University Physics Volume 1', 
        'University Physics Volume 2', 
        'University Physics Volume 3'
    ]

    _PRINCIPLES_OF_CHEMISTRY_COURSE = 'Principles of Chemistry 3rd edition'
    _CHEM31A_COURSE = 'Chem 31A'

    def __init__(self, course_name, num_support, num_query, zero_shot=False) -> None:
        super().__init__()

        self.num_support = num_support
        self.num_query = num_query

        # load questions from Openstax Dataset
        # columns: question, learning_goal, course (all text)
        # multiple learning goals per question, multiple questions per course
        if course_name in self._OPENSTAX_COURSES:
            data = util.load_openstax_course(course_name)
        elif course_name == self._PRINCIPLES_OF_CHEMISTRY_COURSE:
            data = util.load_principles_of_chemistry_course(course_name)
        elif course_name == self._CHEM31A_COURSE:
            data = util.load_chem31a_course()

        # group data by question
        # columns: question (str), learning_goal (list), course (list of single str)
        self.data_by_question = data.groupby('question').agg(list)

        question_embeddings = torch.load('question_curie_embeddings_chem31a.pt')
        self.question_to_embedding = dict(zip(
            self.data_by_question.index, question_embeddings
        ))

        # group data by learning goal
        # columns: question (list), learning_goal (str), course (list of single str)
        self.data_by_learning_goal = data.groupby('learning_goal').agg(list)

        learning_goal_embeddings = torch.load('learning_goal_curie_embeddings_chem31a.pt')
        self.learning_goal_to_embedding = dict(zip(
            self.data_by_learning_goal.index, learning_goal_embeddings
        ))

        self.data_by_learning_goal = self.data_by_learning_goal[
            self.data_by_learning_goal['question'].apply(len) > 1
        ]

        self.zero_shot = zero_shot

    def __getitem__(self, question_index):
        question = self.data_by_question.iloc[question_index].name

        tasks = []
        for i, learning_goal in enumerate(self.data_by_learning_goal.index):
            if self.zero_shot:
                query = [question]
                support_1 = [learning_goal]
                support_0 = [np.random.default_rng(seed=SEED).choice(
                    self.data_by_learning_goal.drop(learning_goal).index
                )]
                support = support_0 + support_1
                labels = [int(question in self.data_by_learning_goal.loc[learning_goal].question)]

                support = torch.stack([self.learning_goal_to_embedding[s] for s in support])
                query = torch.stack([self.question_to_embedding[q] for q in query])
                labels = torch.tensor(labels)

                tasks.append(
                    (support, query, labels)
                )
                continue

            # select examples that match the sampled learning goal
            examples_1 = self.data_by_learning_goal.loc[learning_goal].question
            examples_1_minus_q = [q for q in examples_1 if q != question]
            support_1 = np.random.default_rng(seed=SEED).choice(examples_1_minus_q, self.num_support)

            # select examples that do not have this learning goal
            examples_0 = self.data_by_question.drop(examples_1).index
            support_0 = np.random.default_rng(seed=SEED).choice(examples_0, self.num_support)

            support = list(support_0) + list(support_1)
            query = [question]
            labels_support = ([0] * self.num_support) + ([1] * self.num_support)
            labels_query = [int(question in examples_1)]

            support = torch.stack([self.question_to_embedding[s] for s in support])
            query = torch.stack([self.question_to_embedding[q] for q in query])
            
            labels_support, labels_query = torch.tensor(labels_support), torch.tensor(labels_query)
            tasks.append(
                (support, labels_support, query, labels_query)
            )

        return tasks

    def __len__(self) -> int:
        return self.data_by_question.shape[0]


class CourseTestDataset(dataset.Dataset):
    """
    Dataset for testing a model on tagging a full course, given the course name.
    In this setting, each question consists of a number of tasks equal to
    the number of learning objectives in the course. A question
    is labeled with all learning objectives to which the model assigns a positive
    prediction score.
    """

    _OPENSTAX_COURSES = [
        'Chemistry 2e', 
        'University Physics Volume 1', 
        'University Physics Volume 2', 
        'University Physics Volume 3'
    ]

    _PRINCIPLES_OF_CHEMISTRY_COURSE = 'Principles of Chemistry 3rd edition'
    _CHEM31A_COURSE = 'Chem 31A'

    def __init__(self, course_name, num_support, num_query, tokenizer, max_length=128, task_embedding_model=None) -> None:
        super().__init__()

        self.num_support = num_support
        self.num_query = num_query
        self.tokenizer = tokenizer
        self.max_length = max_length
        # flag whether we're using learning goal embeddings
        self.tam = False

        # load questions from Openstax Dataset
        # columns: question, learning_goal, course (all text)
        # multiple learning goals per question, multiple questions per course
        if course_name in self._OPENSTAX_COURSES:
            data = util.load_openstax_course(course_name)
        elif course_name == self._PRINCIPLES_OF_CHEMISTRY_COURSE:
            data = util.load_principles_of_chemistry_course(course_name)
        elif course_name == self._CHEM31A_COURSE:
            data = util.load_chem31a_course()

        # group data by question
        # columns: question (str), learning_goal (list), course (list of single str)
        self.data_by_question = data.groupby('question').agg(list)

        # group data by learning goal
        # columns: question (list), learning_goal (str), course (list of single str)
        self.data_by_learning_goal = data.groupby('learning_goal').agg(list)
        self.data_by_learning_goal = self.data_by_learning_goal[
            self.data_by_learning_goal['question'].apply(len) > 1
        ]

        # encode learning goals as task embeddings
        self.tam = False
        if task_embedding_model is not None:    
            with torch.no_grad():
                self.learning_goal_embeddings = task_embedding_model.encode(
                    self.data_by_learning_goal.index.values
                )
            self.tam = True

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
        for i, learning_goal in enumerate(self.data_by_learning_goal.index):
            # select examples that match the sampled learning goal
            examples_1 = self.data_by_learning_goal.loc[learning_goal].question
            examples_1_minus_q = [q for q in examples_1 if q != question]
            support_1 = np.random.default_rng(seed=SEED).choice(examples_1_minus_q, self.num_support)

            # select examples that do not have this learning goal
            examples_0 = self.data_by_question.drop(examples_1).index
            support_0 = np.random.default_rng(seed=SEED).choice(examples_0, self.num_support)

            support = list(support_0) + list(support_1)
            query = [question]
            labels_support = ([0] * self.num_support) + ([1] * self.num_support)
            labels_query = [int(question in examples_1)]

            if self.tokenizer:
                support, query = self._tokenize(support), self._tokenize(query)

            # add in task embeddings
            if self.tam:
                support.update({
                    'task_embeds': torch.tensor(self.learning_goal_embeddings[i]).unsqueeze(0).repeat(2 * self.num_support, 1).unsqueeze(1)
                })
                query.update({
                    'task_embeds': torch.tensor(self.learning_goal_embeddings[i]).unsqueeze(0).repeat(1, 1).unsqueeze(1)
                })
            
            labels_support, labels_query = torch.tensor(labels_support), torch.tensor(labels_query)
            tasks.append(
                (support, labels_support, query, labels_query)
            )

        return tasks

    def __len__(self) -> int:
        return self.data_by_question.shape[0]


class nWaykShotDataset(dataset.Dataset):
    """
    Dataset for n-way k-shot classification of questions with learning objectives.
    Used to train and evaluate prototypical network.
    """

    _OPENSTAX_COURSES = [
        'Chemistry 2e', 
        'University Physics Volume 1', 
        'University Physics Volume 2', 
        'University Physics Volume 3'
    ]

    _PRINCIPLES_OF_CHEMISTRY_COURSE = 'Principles of Chemistry 3rd edition'

    def __init__(
        self, 
        num_support, 
        num_query, 
        tokenizer, 
        task_embedding_model=None, 
        max_length=256,
        balanced=True, 
        seed=SEED
    ) -> None:
        super().__init__()

        self.num_support = num_support
        self.num_query = num_query
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.balanced = balanced
        # flag whether we're using learning goal embeddings
        self.tam = False

        # load questions from Openstax Dataset
        # columns: question, learning_goal, course (all text)
        # multiple learning goals per question, multiple questions per course
        data = pd.concat([
            util.load_openstax_course(course) for course in self._OPENSTAX_COURSES
        ] + [util.load_principles_of_chemistry_course(self._PRINCIPLES_OF_CHEMISTRY_COURSE)])

        # group data by question
        # dictionary mapping from course name to dataframe of questions within this course
        # columns: question (str), learning_goal (list), course (list of single str)
        self.data_by_question = {
            course_name: data[data['course'] == course_name].groupby('question').agg(list)
            for course_name in data['course'].unique()
        }

        # group data by learning goal
        # columns: question (list), learning_goal (str), course (list of single str)
        self.data_by_learning_goal = data.groupby('learning_goal').agg(list)
        # ignore learning goals that do not have enough training examples
        # NOTE: questions under these learning goals can still appear as NEGATIVE examples, just not positive examples
        self.data_by_learning_goal = self.data_by_learning_goal[
            self.data_by_learning_goal['question'].apply(len) >= self.num_support + self.num_query
        ]
        # shuffle order of learning goals for training!!
        self.data_by_learning_goal = self.data_by_learning_goal.sample(frac=1., random_state=self.seed)

        # encode learning goals as task embeddings
        if task_embedding_model is not None:    
            with torch.no_grad():
                self.learning_goal_embeddings = task_embedding_model.encode(
                    self.data_by_learning_goal.index.values
                )
            self.tam = True

        # construct a random number generator
        self.rng = np.random.default_rng(seed=self.seed)        


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

    def get_data_by_learning_goal(self):
        return self.data_by_learning_goal

    def __getitem__(self, index):
        if not self.balanced:
            # get learning goal from index
            learning_goal = self.data_by_learning_goal.iloc[index]

            # select examples that match the sampled learning goal
            examples_1 = learning_goal.question
            support_1 = self.rng.choice(
                examples_1, self.num_support, replace=False
            )

            # select examples that do not have this learning goal
            examples_0 = self.data_by_question[learning_goal.course[0]].drop(learning_goal.question).index
            support_0 = self.rng.choice(
                examples_0, self.num_support, replace=False
            )

            support = list(support_0) + list(support_1)
            query = list(self.rng.choice(
                self.data_by_question[learning_goal.course[0]].drop(support).index, self.num_query, replace=False
            ))

            labels_support = ([0] * self.num_support) + ([1] * self.num_support)
            labels_query = [int(q in learning_goal.question)for q in query]

            if self.tokenizer:
                support, query = self._tokenize(support), self._tokenize(query)

            # add in task embeddings
            if self.tam:
                support.update({
                    'task_embeds': torch.tensor(self.learning_goal_embeddings[index]).unsqueeze(0).repeat(2 * self.num_support, 1).unsqueeze(1)
                })
                query.update({
                    'task_embeds': torch.tensor(self.learning_goal_embeddings[index]).unsqueeze(0).repeat(2 * self.num_query, 1).unsqueeze(1)
                })

            labels_support, labels_query = torch.tensor(labels_support), torch.tensor(labels_query)

            return support, labels_support, query, labels_query
        else:
            # get learning goal from index
            learning_goal = self.data_by_learning_goal.iloc[index]

            # select examples that match the sampled learning goal
            examples_1 = learning_goal.question
            support_and_query_1 = self.rng.choice(
                examples_1, self.num_support + self.num_query, replace=False
            )
            support_1 = support_and_query_1[:self.num_support]
            query_1 = support_and_query_1[self.num_support:]

            # select examples that do not have this learning goal
            examples_0 = self.data_by_question[learning_goal.course[0]].drop(learning_goal.question).index
            support_and_query_0 = self.rng.choice(
                examples_0, self.num_support + self.num_query, replace=False
            )
            support_0 = support_and_query_0[:self.num_support]
            query_0 = support_and_query_0[self.num_support:]

            support = list(support_0) + list(support_1)
            query = list(query_0) + list(query_1)
            labels_support = ([0] * self.num_support) + ([1] * self.num_support)
            labels_query = ([0] * self.num_query) + ([1] * self.num_query)

            if self.tokenizer:
                support, query = self._tokenize(support), self._tokenize(query)

            # add in task embeddings
            if self.tam:
                support.update({
                    'task_embeds': torch.tensor(self.learning_goal_embeddings[index]).unsqueeze(0).repeat(2 * self.num_support, 1).unsqueeze(1)
                })
                query.update({
                    'task_embeds': torch.tensor(self.learning_goal_embeddings[index]).unsqueeze(0).repeat(2 * self.num_query, 1).unsqueeze(1)
                })

            labels_support, labels_query = torch.tensor(labels_support), torch.tensor(labels_query)

            return support, labels_support, query, labels_query

    def __len__(self) -> int:
        return self.data_by_learning_goal.shape[0]


class nWaykShotSampler(sampler.Sampler):
    def __init__(self, split_idx, num_tasks, seed=SEED) -> None:
        super().__init__(None)
        self._num_tasks = num_tasks
        self._indices = split_idx
        self.rng = np.random.default_rng(seed=seed)
    
    def __iter__(self):
        if self._num_tasks is None:
            return (i for i in self._indices)
        else:
            return (
                self.rng.choice(self._indices, replace=False)
                for _ in range(self._num_tasks)
            )
    
    def __len__(self):
        if self._num_tasks is None:
            return len(self._indices)
        else:
            return self._num_tasks


class ContrastiveSampler(sampler.Sampler):
    def __init__(self, dataset : nWaykShotDataset, split_idx, num_tasks, seed=SEED) -> None:
        super().__init__(dataset)
        self._num_tasks = num_tasks
        self._indices = split_idx
        self.rng = np.random.default_rng(seed=seed)
        self.data_by_learning_goal = dataset.get_data_by_learning_goal().reset_index()

    def _sample_learning_goal_index(self, index):
        lg = self.data_by_learning_goal.iloc[index]
        safe_lgs = self.data_by_learning_goal[
            self.data_by_learning_goal['question'].apply(lambda l: all([q not in lg.question for q in l]))
        ]
        return self.rng.choice(safe_lgs.index)
    
    def __iter__(self):
        return (
            (i, self._sample_learning_goal_index(i))
            for i in range(self.rng.choice(self.data_by_learning_goal.index, size=self.num_tasks, replace=False))
        )


    def __len__(self):
        return self._num_tasks


def identity(x):
    return x

def get_nway_kshot_dataloader(
    split,
    batch_size,
    num_support,
    num_query,
    num_tasks_per_epoch,
    tokenizer,
    task_embedding_model_type=None, # SBERT model to encode tasks
    num_workers=8,
    max_length=128,
    sample_by_learning_goal=False,
    balanced=True,
    seed=SEED
):
    if task_embedding_model_type is not None:
        task_embedding_model = SentenceTransformer(task_embedding_model_type)
    else:
        task_embedding_model = None
    dataset = nWaykShotDataset(
        num_support, 
        num_query, 
        tokenizer, 
        task_embedding_model=task_embedding_model, 
        max_length=max_length,
        balanced=balanced,
        seed=seed
    )
    num_train_classes = int(len(dataset) * FRAC_TRAIN_CLASSES)
    num_val_classes = int(len(dataset) * FRAC_VAL_CLASSES)
    num_test_classes = int(len(dataset) * FRAC_TEST_CLASSES)

    if split == 'train':
        split_idxs = range(num_train_classes)
    elif split == 'val':
        split_idxs = range(num_train_classes, num_train_classes + num_val_classes)
    else:
        split_idxs = range(num_train_classes + num_val_classes, num_train_classes + num_val_classes + num_test_classes)
    
    if sample_by_learning_goal:
        sampler = ContrastiveSampler(dataset, split_idxs, num_tasks_per_epoch, seed)
    else:
        sampler = nWaykShotSampler(split_idxs, num_tasks_per_epoch, seed)

    return dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )