import json
import os
from dashscope import MultiModalConversation
import base64
import mimetypes
import dashscope
from PIL import Image

# 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# ---用于 Base64 编码 ---
# 格式为 data:{mime_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")


def get_image_size(file_path):
    """获取图像尺寸"""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            return f"{width}*{height}"
    except Exception as e:
        raise ValueError(f"无法读取图像尺寸: {file_path}, 错误: {str(e)}")


# 图像文件路径
image_path = "/home/chen/seg6/predict_no_label/experiment/in/experiment/2-4.jpg"

# 获取图像的 Base64 编码
image = encode_file(image_path)

# 获取输入图像的尺寸
input_image_size = get_image_size(image_path)
print(f"输入图像尺寸: {input_image_size}")

messages = [
    {
        "role": "user",
        "content": [
            {"image": image},
            {"text": "这是一张钙钛矿sem晶粒图，要求在原图上对晶粒边界用黑色线条进行加粗，线条宽度控制很细，要求线条清晰且粗细一致，所有晶粒边界都要加粗，保持原图其他部分不变"}
        ]
    }
]

# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
# 若没有配置环境变量，请用百炼 API Key 将下行替换为：api_key="sk-xxx"
api_key = "sk-e7d2c07fd58249198fdfcff390d8614c"

# qwen-image-edit-plus支持输出1-6张图片，此处以2张为例
response = MultiModalConversation.call(
    api_key=api_key,
    model="wan2.6-image",
    messages=messages,
    stream=False,
    n=1,
    watermark=False,
    negative_prompt="低质量，画非晶粒的边界",
    prompt_extend=True,
    # 仅当输出图像数量n=1时支持设置size参数，否则会报错
    # 使用输入图像的实际尺寸
    size=input_image_size,
)

if response.status_code == 200:
    # 如需查看完整响应，请取消下行注释
    # print(json.dumps(response, ensure_ascii=False))
    for i, content in enumerate(response.output.choices[0].message.content):
        print(f"输出图像{i+1}的URL:{content['image']}")
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")
