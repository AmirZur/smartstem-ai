import os
import re
import json

import torch
import pandas as pd

OPENSTAX_DIR = 'OpenStax Dataset'

def get_model_name(size):
    size_to_model = {
        'tiny': 'prajjwal1/bert-tiny',
        'small': 'prajjwal1/bert-small',
        'bert': 'bert-base-uncased'
    }
    return size_to_model[size]


def parse_openstax_questions_file(filename, folder_path):
    chapter_num = int(filename[:-4]) # everything before .txt extension
    with open(os.path.join(folder_path, filename), encoding='utf-8') as f:
        lines = [l.strip() for l in f]

    questions = {}
    question_nums = []
    current_subchapter = ''
    for line in lines:
        # when we encounter subchapter heading
        subchapter_num = re.match('[0-9]+\.[0-9]+', line)
        if subchapter_num:
            current_subchapter = subchapter_num.group(0)
            questions[current_subchapter] = []
            continue

        # when we encounter questions
        question_num = re.match('[0-9]+\. ', line)
        if question_num:
            question_num = question_num.group(0)
            questions[current_subchapter].append(line[len(question_num):])
            question_nums.append(question_num)
            continue

        # if this is part of a previous question
        questions[current_subchapter][-1] += line

    return questions, question_nums

def parse_openstax_questions_folder(folder_path):
    questions, question_nums = {}, []
    for filename in os.listdir(folder_path):
        if filename.endswith('txt'):
            q, q_num = parse_openstax_questions_file(filename, folder_path)
            questions.update(q)
            question_nums.extend(q_num)
    return questions, question_nums


def read_openstax_textbook_info(path):
    return pd.read_csv(path)


def load_openstax_course(course_name):
    course_code = course_name.replace(' ', '').lower()
    with open(f'{course_code}_subchapter_to_learning_goal.json') as f:
        subchapter_to_lgs = json.load(f)
    
    subchapter_to_lgs = {
        re.findall('[0-9]+\.[0-9]+', k)[0]: v for k, v in subchapter_to_lgs.items()
    }

    questions, question_nums = parse_openstax_questions_folder(
        os.path.join(OPENSTAX_DIR, course_name)
    )

    dataset = []
    for subchapter, question_list in questions.items():
        for question in question_list:
            for learnning_goal in subchapter_to_lgs[subchapter]:
                dataset.append([question, learnning_goal])

    dataset = pd.DataFrame(data=dataset, columns=['question', 'learning_goal'])
    dataset['course'] = course_name
    return dataset


def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()
