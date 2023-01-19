""" 
Utility file to load, parse, and organize question + learning objective data from OpenStax, 
Principles of Chemistry (3rd Edition), and Chem 31A. 
"""

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


def load_chem31a_learning_goals(filename='chem31a_learning_goal_list.txt'):
    learning_goals = {}

    with open(filename, encoding='utf-8') as f:
        current_unit = None
        current_letter = ord('a')
        lines = [l.strip() for l in f.readlines()]
        for line in lines[1:]:
            unit = re.match('[0-9]\.', line)
            if unit:
                number = int(unit.group(0)[:-1])
                assert number not in learning_goals
                learning_goals[number] = {}
                learning_goals[number]['title'] = line[len(unit.group(0)):]
                current_unit = number
                current_letter = ord('a')
            else:
                assert current_unit is not None
                assert current_unit in learning_goals
                learning_goals[current_unit][chr(current_letter)] = line
                current_letter += 1
    return learning_goals


def parse_chem31a_questions(filename, year='2021'):
    exam_path = os.path.join('Chem 31A', year, filename)
    with open(exam_path, encoding='utf-8') as f:
        questions = {}
        lines = [l.strip() for l in f.readlines() if l.strip() != '']
        current_q_num = None
        current_q_title = None
        current_sub_q_num = None
        for line in lines:
            q_num = re.match('[0-9]+\) ', line)
            if q_num:
                q_num = q_num.group(0)
                current_q_num = int(q_num[:q_num.find(')')])
                current_q_title = line[len(q_num):]
                assert q_num not in questions
                questions[current_q_num] = {}
            # first question is always multiple choice with roman numerals
            elif current_q_num == 1:
                sub_q_num = re.match('[iv]+\) ', line)
                if sub_q_num:
                    sub_q_num = sub_q_num.group(0)
                    current_sub_q_num = sub_q_num[:-2]
                    assert current_q_num in questions
                    assert current_sub_q_num not in questions[current_q_num]
                    questions[current_q_num][current_sub_q_num] = line[len(sub_q_num):]
                else:
                    assert current_q_num in questions
                    if current_sub_q_num in questions[current_q_num]:
                        questions[current_q_num][current_sub_q_num] += line
                    else:
                        current_q_title += line
            # second question is always short answer with letters
            elif current_q_num == 2:
                # skip roman numerals i and v
                sub_q_num = re.match('[a-h|j-u|w-z]\) ', line)
                if sub_q_num:
                    sub_q_num = sub_q_num.group(0)
                    current_sub_q_num = sub_q_num[:-2]
                    assert current_q_num in questions
                    assert current_sub_q_num not in questions[current_q_num]
                    questions[current_q_num][current_sub_q_num] = line[len(sub_q_num):]
                else:
                    assert current_q_num in questions
                    if current_sub_q_num in questions[current_q_num]:
                        questions[current_q_num][current_sub_q_num] += line
                    else:
                        current_q_title += line
            # all other questions use the question title, always letters
            else:
                # skip roman numerals i and v
                sub_q_num = re.match('[a-h|j-u|w-z]\) ', line)
                if sub_q_num:
                    sub_q_num = sub_q_num.group(0)
                    current_sub_q_num = sub_q_num[:-2]
                    assert current_q_num in questions
                    assert current_sub_q_num not in questions[current_q_num]
                    # make sure to include the question title for these!
                    questions[current_q_num][current_sub_q_num] = current_q_title + line[len(sub_q_num):]
                else:
                    assert current_q_num in questions
                    if current_sub_q_num in questions[current_q_num]:
                        questions[current_q_num][current_sub_q_num] += line
                    else:
                        current_q_title += line
    return questions


def load_chem31a_questions_to_learning_goals(filename, year='2021'):
    file_path = os.path.join('Chem 31A', year, filename)
    df = pd.read_csv(file_path)

    questions_to_lgs = {}
    for i, row in df.iterrows():
        question = row['Exam Q#']
        # ignore empty question cells (only applicable for Exam 3)
        if pd.isna(question):
            continue
        q_num = re.match('[0-9]+', question).group(0)
        sub_q_num = question[len(q_num):]
        # prevent cases like bi, bii
        if not sub_q_num.startswith('i') and not sub_q_num.startswith('v'):
            sub_q_num = sub_q_num[0]
        q_num = int(q_num)
        if q_num not in questions_to_lgs:
            questions_to_lgs[q_num] = {}
        if sub_q_num in questions_to_lgs[q_num]:
            continue
        questions_to_lgs[q_num][sub_q_num] = []
        for lg_num, col in enumerate(df.columns[2:10], start=1):
            # ignore empty cells
            if pd.isna(row[col]):
                continue
            lgs = [c.lower() for c in row[col] if c.isalpha()]
            questions_to_lgs[q_num][sub_q_num].extend(
                [(lg_num, lg) for lg in lgs]
            )
    return questions_to_lgs


def load_chem31a_course(year='2021'):
    learning_goals = load_chem31a_learning_goals()

    data = []
    for e in range(1, 5):
        questions = parse_chem31a_questions(f'Chem31A_2021_Exam{e}.txt', year=year)
        questions_to_learning_goals = load_chem31a_questions_to_learning_goals(
            f'questions_to_learning_goals_exam{e}.csv', year=year
        )
        for q_num in questions:
            for sub_q_num in questions[q_num]:
                for lg in questions_to_learning_goals[q_num][sub_q_num]:
                    data.append([
                        questions[q_num][sub_q_num],
                        learning_goals[lg[0]][lg[1]]
                    ])
    df = pd.DataFrame(data, columns=['question', 'learning_goal'])
    df['course'] = 'Chem31A'
    return df