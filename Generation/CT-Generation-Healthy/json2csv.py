import json
import csv
import os

def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # 尝试判断是标准JSON还是JSONL格式
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError:
        # JSONL格式：每行一个JSON对象
        data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Names', 'Text_prompts'])
        
        for item in data:
            volume_path = item.get('volume_path', '')
            name = os.path.basename(volume_path)  # 提取文件名
            findings = item.get('findings', '')
            writer.writerow([name, findings])
    
    print(f"转换完成，共 {len(data)} 条记录，已保存到 {csv_file}")

# 使用示例
json_to_csv('infer_valid_report_prompt_no_disease_mask.json', 'inference_valid_normal_new.csv')