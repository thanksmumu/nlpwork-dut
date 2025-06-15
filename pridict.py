import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==== 配置 ====
lora_model_path = "lora/train_2025-06-13-00-17-28"
test_file = "test1.json"
output_file = "predict_output.txt"

# ==== 获取 base 模型路径 ====
with open(os.path.join(lora_model_path, "adapter_config.json"), encoding="utf-8") as f:
    adapter_cfg = json.load(f)
base_model_path = os.path.abspath(adapter_cfg["base_model_name_or_path"])
print("✅ 使用 base 模型路径:", base_model_path)

# ==== Prompt 模板 ====
def make_prompt(text):
    return f"""你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个四元组:
回答格式为：target[即句子中的评论对象] | arguement[...] | target group[...] | hateful[...] 
注意，每个四元组中各个元素之间用 " | " 分割，并利用 [END] 结尾；如果一条样本中包含多个四元组，不同四元组之间利用 [SEP] 分割。
请严格按照顺序和格式提交，不要省略空格。{text}四元组："""

# ==== 设置 device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ==== 加载 tokenizer 和 base 模型 ====
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=dtype
).to(device)

# ==== 加载 LoRA 模型 ====
model = PeftModel.from_pretrained(base_model, lora_model_path, is_trainable=False)
model = model.to(device).eval()

# ==== 加载测试数据 ====
with open(test_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# ==== 推理并输出 ====
with open(output_file, 'w', encoding='utf-8') as f_out:
    for idx, item in enumerate(data_list):
        try:
            text = item["content"]
            prompt = make_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            quadruples = decoded.split("四元组：")[-1].strip()
            f_out.write(quadruples + "\n")
        except Exception as e:
            f_out.write(f"[ERROR] 第{idx}条出错: {str(e)}\n")
            print(f"❌ 第{idx}条推理失败: {e}")

print(f"✅ 推理完成，输出结果保存至 {output_file}")
