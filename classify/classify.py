import json
import os
import random
import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def classify_text(content: str, model_name: str, max_retries: int = 3) -> str:
    """
    使用Langchain的LCEL表达式对文本进行分类

    Args:
        content: 需要分类的文本

    Returns:
        分类标签
    """
    classification_template = """
    请你作为专业文本分类专家，对传入文本进行标签打标。遵循以下要求
    1. 使用下列分类及定义，对文本进行标签打标。若文本符合多个类别，可同时打上多个标签；同一文本不得出现重复标签。
    
    ```markdown
    | 分类名称 | 定义与核心内容                                                        |
    |------|----------------------------------------------------------------|
    | 文学小说 | 以语言文字为工具，通过塑造人物形象、构建故事情节和描绘具体环境来艺术地反映和表现社会生活。核心在于其艺术性和思想深度。    |
    | 青春文学 | 主要表现青春期生活、情感、成长与迷茫的文学作品，核心受众是青少年。                              |
    | 亲子育儿 | 关注父母与子女互动关系，提供育儿理念、方法和指导的实用型读物。                                |
    | 科普读物 | 以通俗易懂的方式向非专业读者普及科学知识、传播科学思想、弘扬科学精神的书籍。                         |
    | 动漫幽默 | 内容与动画、漫画相关，或以制造笑料、体现幽默感为主要目的的读物。                               |
    | 人文社科 | 研究人类文化、社会现象、思想传承的学科总称，注重理论分析、批判性思考和社会洞察。                       |
    | 艺术收藏 | 涵盖艺术理论、艺术史、艺术家、创作技法、鉴赏知识以及各类收藏品（如古玩、字画）的鉴赏与收藏指南。               |
    | 古籍地理 | 古籍：1912年以前出版的线装书、重要学术整理本（如《史记》校注本）。地理：系统描述地域自然与人文风貌的著作（非旅游指南）。 |
    | 旅游休闲 | 为旅行或日常休闲活动提供信息、灵感或指导的实用性读物。                                    |
    | 经济管理 | 研究价值创造、交换、消费等经济活动，以及组织机构如何有效运作和决策的领域。                          |
    | 励志成长 | 旨在激励个人通过改变态度、思维方式或行为习惯，以提升自我、实现个人目标的作品。                        |
    | 外语学习 | 以教授外语或帮助提升外语能力为核心目的的出版物。                                       |
    | 法律哲学 | 法律：阐述法理、法律制度、法律条文的著作。哲学：探究世界本源、知识、存在、价值等基本问题的系统性思考。            |
    | 政治军事 | 政治：研究权力分配、国家治理、公共政策、国际关系的学说与实践。军事：涉及国防、战争、军队、武器装备、战略战术的历史与理论。  |
    | 自然科学 | 研究自然界各现象及其规律的知识体系，包括物理学、化学、生物学、天文学、地球科学等。                      |
    | 家庭教育 | 聚焦于家庭环境中对子女的教育理念、方式、氛围及其影响，具有较强的教育学和心理学背景。                     |
    | 两性关系 | 探讨恋爱、婚姻、家庭中男性与女性之间的互动、情感、心理及社会关系的读物。                           |
    | 孕产育儿 | 专门针对女性孕期、分娩、产后恢复以及0-3岁左右婴幼儿的喂养、保健、早期启蒙等知识的指导用书。                |
    | 家居生活 | 提供关于改善居家环境、提升日常生活品质的实用建议和灵感的读物。                                |
    | 生活时尚 | 关注个人外在形象、生活方式、消费潮流与生活品味的指南性读物。                                 |
    ```
    
    2. 输出要求：
     - 请仅回复最适合的分类标签，不存在多个标签，不要包含其他解释或说明，不要输出思考过程、推理步骤或任何元认知内容。
     - 只输出上面表格中中文分类名称（如：文学小说），没有额外文字解释。
    
    3. 判定原则与边界规则：
    - 优先级：若文本同时符合多类别，优先考虑核心主题与目标受众；如核心是成长、教育或社会洞察，优先对应"青春文学"或"人文社科/家庭教育"之类。若内容偏向技术性知识或科普事实，应优先"科普读物/自然科学"等。
    - 边界处理：如文本具备文学性与娱乐性，且以故事情节为主，优先"文学小说"；如文本更多为知识性、方法论导向，优先"经济管理/科普读物/自然科学"等。
    
    示例：
    文本1：这是一部以校园成长为主线的小说，描写少年在友情与初恋中的困惑与成长，语言优雅，富有社会批判意识。
    输出：
    文学小说
    文本2：文章系统介绍了宏观经济运行与企业决策的分析框架，含财务模型与市场策略。
    输出：
    经济管理

    文本内容：
    {content}

    分类标签：
    """

    prompt = PromptTemplate.from_template(classification_template)
    model = init_chat_model(model=model_name, temperature=0)
    chain = prompt | model | StrOutputParser()
    for attempt in range(max_retries):
        try:
            # 执行分类
            result = chain.invoke({"content": content})
            return result.strip()
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试分类失败: {e}")
            if attempt < max_retries - 1:
                # 指数退避策略，随机等待一段时间再重试
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"等待 {wait_time:.2f} 秒后进行第 {attempt + 2} 次重试...")
                time.sleep(wait_time)
            else:
                print("达到最大重试次数，分类失败")
                raise e

def classify_dataset(dataset_path: str, model_name: str, output_path: str):
    """
    对整个数据集进行分类

    Args:
        dataset_path: 包含指令的JSONL文件路径
        model_name: 模型名称
        output_path: 输出分类结果的文件路径
    """
    start_line = 0
    classification_stats = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                start_line = max(start_line, data.get("id"))
                classification = data.get("classification")
                if classification in classification_stats:
                    classification_stats[classification] += 1
                else:
                    classification_stats[classification] = 1

    print(f"从第 {start_line + 1} 行开始处理")
    print(f"当前分类统计: {classification_stats}")

    base_name = os.path.splitext(output_path)[0]
    error_path = f"{base_name}_errors.jsonl"

    with open(dataset_path, "r", encoding="utf-8") as infile, \
            open(output_path, "a", encoding="utf-8") as outfile, \
            open(error_path, "a", encoding="utf-8") as errorfile:
        # 跳过已经处理过的行
        for _ in range(start_line):
            next(infile)

        for line in infile:
            data = json.loads(line)
            text = data["text"]
            id = data.get("id")
            try:
                # 仅测试使用
                # if id == 10:
                #     raise Exception("测试异常抛出")

                classification = classify_text(text, model_name)
                data["classification"] = classification
                if classification in classification_stats:
                    classification_stats[classification] += 1
                else:
                    classification_stats[classification] = 1

                # 仅测试使用
                # if id == 15:
                #     sys.exit(1)

                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                outfile.flush()

            except Exception as e:
                error_data = {
                    "id": id,
                    "text": text,
                    "error": str(e)
                }
                errorfile.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                errorfile.flush()

    print("\n" + "=" * 50)
    print("分类处理完成！最终统计结果:")
    print("=" * 50)
    sorted_final_stats = sorted(classification_stats.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_final_stats:
        print(f"{category}: {count}")
    print("=" * 50)

def main():
    # dataset_folder = "./datasets/infinity_instruct/extracted_data"
    # extract_conversations(dataset_folder)

    load_dotenv()
    input_path = "./datasets/chinese_cosmopedia/extracted_data/chinese_cosmopedia.jsonl"
    output_path = "./datasets/chinese_cosmopedia/output/chinese_cosmopedia_classify.jsonl"
    model = "openai:gpt-5-nano"
    classify_dataset(input_path, model, output_path)


if __name__ == '__main__':
    main()