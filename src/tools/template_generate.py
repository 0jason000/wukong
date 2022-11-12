import os
from .simple_tokenizer import set_tokenizer_lang, tokenize


def generate_zh_template(label_list):
    set_tokenizer_lang('zh', 32)
    template_list = []
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'zh_templates.txt'
    )

    templates = []
    for line in open(template_path, 'r'):
        templates.append(line.strip())
    num_prompts = len(templates)
    num_labels = len(label_list)
    for label in label_list:
        for template in templates:
            template_list.append(template.replace('{}', label))
    token = tokenize(template_list).reshape((num_labels, num_prompts, -1))
    return token
