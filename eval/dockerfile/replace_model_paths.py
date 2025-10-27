import re


def replace_model_paths(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配 model_path='...' 或 model_path="..."
    def replacer(match):
        quote = match.group(1)  # 获取引号类型：单引号或双引号
        orig_path = match.group(2)  # 获取原始路径
        # 构造新路径并用相同的引号包裹
        local_path = f"/models/{orig_path}"
        return f'model_path={quote}{local_path}{quote}'

    # 正则匹配单引号或双引号
    pattern = r'model_path=([\'"])([^\'"]*)\1'
    new_content = re.sub(pattern, replacer, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    replace_model_paths('./config.py')