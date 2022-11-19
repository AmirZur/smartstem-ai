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
        questions[current_subchapter][-1] += '\n' + line

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

def parse_principles_of_chemistry_question_file(filename):
    with open(filename, encoding='utf-8') as fin:
        lines = [l.strip() for l in fin]

    current_question_num = -1
    question_nums = []
    questions = {}
    for line in lines:
        if line == '':
            continue
        # search for question heading
        question_num = re.match('[0-9]+\. ', line)
        if question_num:
            question_num = question_num.group(0)
            num = int(question_num[:question_num.find('.')])
            assert num not in questions
            questions[num] = line[len(question_num):]
            question_nums.append(num)
            current_question_num = num
        # otherwise just postfix to current question
        else:
            assert current_question_num != -1
            questions[current_question_num] += '\n' + line
    return questions


def parse_principles_of_chemistry_learning_goal_file(filename):
    with open(filename, encoding='utf-8') as fin:
        lines = [l.strip() for l in fin]
    
    learning_goals = {}
    chapter_nums = []
    current_chapter = -1
    for line in lines:
        if line == '':
            continue
        # search for chapter heading
        chapter_name = re.match('Chapter [0-9]+', line)
        if chapter_name:
            chapter_name = chapter_name.group(0)
            current_chapter = int(chapter_name[len('Chapter '):])
            assert current_chapter not in learning_goals
            chapter_nums.append(current_chapter)
        else:
            # find end of learning goal
            learning_goal = re.search(' \([0-9]+\.[0-9]+\)', line)
            learning_goal = line[:learning_goal.span()[0]]

            learning_goals[learning_goal] = {'chapter': current_chapter, 'question_numbers': []}
            # get all question numbers following exercises
            exercises = line[line.find('Exercise') + len('Exercises '):].split(',')
            for question_num in exercises:
                if question_num == '':
                    continue
                # some question numbers are ranges, e.g. 9-12
                dash_index = question_num.find('–')
                # if not a range, just add the question
                if dash_index == -1:
                    learning_goals[learning_goal]['question_numbers'].append(int(question_num))
                # if it's a range, we should parse the start and end
                else:
                    learning_goals[learning_goal]['question_numbers'].extend(list(range(
                        int(question_num[:dash_index]), int(question_num[dash_index + 1:]) + 1
                    )))
    return learning_goals, chapter_nums


def load_principles_of_chemistry_course(folder):
    lg_path = os.path.join(folder, 'Learning Goals.txt')
    learning_goals, _ = parse_principles_of_chemistry_learning_goal_file(lg_path)
    chapter_to_questions = {
        i: parse_principles_of_chemistry_question_file(
            os.path.join(folder, f'Chapter {i} Questions.txt')
        )
        for i in range(1, len(os.listdir(folder)))
    }
    data = []
    for learning_goal, meta_data in learning_goals.items():
        chapter_questions = chapter_to_questions[meta_data['chapter']]
        for question_number in meta_data['question_numbers']:
            if question_number not in chapter_questions:
                print(f"We don't have question {question_number} from chapter {meta_data['chapter']}")
                continue
            # entries consist of question, learning goal, and course name
            data.append([chapter_questions[question_number], learning_goal, folder])
    return pd.DataFrame(data=data, columns=['question', 'learning_goal', 'course'])
    

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
