import os
import re
import pandas as pd

OPENSTAX_DIR = 'OpenStax Dataset'


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
    textbook_info = read_openstax_textbook_info(
        os.path.join(OPENSTAX_DIR, course_name, course_name.replace(' ', '') + '_textbook_info.csv')
    )
    subchapter_to_lgs = dict(zip(
        textbook_info['Subchapter Number'].astype(str), textbook_info['Subchapter Learning Objectives']
    ))

    questions, question_nums = parse_openstax_questions_folder(
        os.path.join(OPENSTAX_DIR, course_name)
    )

    dataset = []
    for subchapter, question_list in questions.items():
        for question in question_list:
            for learnning_goal in subchapter_to_lgs[subchapter].split('\n'):
                dataset.append([question, learnning_goal])

    dataset = pd.DataFrame(data=dataset, columns=['question', 'learning_goal'])
    return dataset

