import os
import random
import time
from tqdm import tqdm
import dashscope
from http import HTTPStatus

# 1. 配置
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
dashscope.api_key = api_key

test_dir = 'TEST'
SAMPLES_PER_CLASS = 300

# 2. 准备类别列表和 Prompt
try:
    class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    class_list_str = ", ".join(class_names)
    print(f"找到 {len(class_names)} 个类别。")
except FileNotFoundError:
    print(f"找不到目录 '{test_dir}'。")
    exit()

PROMPT_TEMPLATE = f"""
你是一位植物病理学专家。你的任务是根据提供的植物叶片图片，准确识别其健康状况或所患疾病。

请从以下 {len(class_names)} 个预定义类别中选择最匹配的一个：
---
{class_list_str}
---

重要要求：你的回答必须 **仅仅** 包含列表中最准确的类别名称，不要添加任何解释、描述或其他文字。
"""

# 3. 定义 API 调用函数
def classify_image_with_qwen(image_path: str) -> str:
    """
    调用通义千问 qwen-vl-max 模型对单张图片进行分类。

    Args:
        image_path: 本地图片文件的路径。

    Returns:
        模型的预测类别名称，如果出错则返回相应的错误信息。
    """
    local_file_path = f'file://{os.path.abspath(image_path)}'
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": local_file_path},
                {"text": PROMPT_TEMPLATE}
            ]
        }
    ]
    
    try:
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages
        )

        if response.status_code == HTTPStatus.OK:
            # API返回的content是一个列表，提取列表中第一个元素的'text'字段
            content_list = response.output.choices[0].message.content
            if isinstance(content_list, list) and len(content_list) > 0 and 'text' in content_list[0]:
                prediction_text = content_list[0]['text']
                return prediction_text.strip()
            else:
                print(f"API 响应格式异常: {content_list}")
                return "Error: Unexpected Response Format"
        else:
            print(f"API 请求失败: Code={response.code}, Message={response.message}")
            return "Error: API Request Failed"
            
    except Exception as e:
        print(f"调用 API 时发生异常: {e}")
        return f"Error: Exception Occurred ({type(e).__name__})"


# 4. 执行分类测试
if __name__ == "__main__":
    total_processed = 0
    total_correct = 0
    incorrect_predictions = []

    print(f"使用 qwen-vl-max 模型进行分类测试")
    print(f"将为每个类别随机测试 {SAMPLES_PER_CLASS} 张图片。")
    
    total_images_to_test = 0
    for true_label in class_names:
        class_path = os.path.join(test_dir, true_label)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        num_samples = min(SAMPLES_PER_CLASS, len(image_files))
        total_images_to_test += num_samples
        
    try:
        with tqdm(total=total_images_to_test, desc="Overall Progress") as pbar:
            for true_label in class_names:
                print(f"正在处理类别: {true_label}")
                class_path = os.path.join(test_dir, true_label)
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                
                if not image_files:
                    print(f"类别 '{true_label}' 中没有找到图片，已跳过。")
                    continue

                num_to_sample = min(SAMPLES_PER_CLASS, len(image_files))
                sampled_images = random.sample(image_files, num_to_sample)
                
                for image_name in sampled_images:
                    image_path = os.path.join(class_path, image_name)
                    
                    predicted_label = classify_image_with_qwen(image_path)
                    
                    if true_label == predicted_label:
                        total_correct += 1
                        print(f"正确: {image_name} -> 预测为 '{predicted_label}'")
                    else:
                        print(f"错误: {image_name} -> 真实: '{true_label}', 预测: '{predicted_label}'")
                        incorrect_predictions.append({
                            'image': image_name,
                            'true': true_label,
                            'predicted': predicted_label
                        })
                    
                    total_processed += 1
                    pbar.update(1)

                    time.sleep(2)
    except KeyboardInterrupt:
        print("手动中断了程序。正在生成当前结果")
    
    # 5. 输出最终结果
    print("\n测试结果")
    if total_processed > 0:
        accuracy = (total_correct / total_processed) * 100
        print(f"总计测试图片数: {total_processed}")
        print(f"正确预测数: {total_correct}")
        print(f"准确率: {accuracy:.2f}%")

        if incorrect_predictions:
            print("\n错误分类详情")
            for item in incorrect_predictions:
                print(f"图片: {item['image']}, 真实类别: '{item['true']}', 错误预测为: '{item['predicted']}'")
    else:
        print("没有处理任何图片，请检查数据集路径和内容。")