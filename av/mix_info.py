
import json
import os


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 读取 files_status.json
    files_status_path = os.path.join(script_dir, 'files_status.json')
    with open(files_status_path, 'r', encoding='utf-8') as f:
        files_status = json.load(f)

    # 读取 translated.json
    translated_path = os.path.join(script_dir, 'translated.json')
    with open(translated_path, 'r', encoding='utf-8') as f:
        translated = json.load(f)

    # 构建结果
    result = {}
    for key, path in files_status.items():
        # 获取影片名字
        if key in translated:
            name = translated[key]
        else:
            # 使用路径中最后的子目录名
            name = os.path.basename(path)

        result[key] = {
            "path": path,
            "name": name
        }

    # 写入 mix_info.json
    output_path = os.path.join(script_dir, 'mix_info.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已生成 mix_info.json，共 {len(result)} 条记录")


if __name__ == '__main__':
    main()

