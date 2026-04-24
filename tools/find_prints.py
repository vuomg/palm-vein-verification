import re
import sys

def main():
    file_path = r'c:\Research\Research\PalmVein\train.py'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-16') as f:
            lines = f.readlines()
            
    vietnamese_pattern = re.compile(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]')
    
    print("--- Vietnamese text found ---")
    for i, line in enumerate(lines):
        if vietnamese_pattern.search(line):
            print(f"L{i+1}: {line.strip()}")
            
    print("\n--- Print statements (first 20) ---")
    print_lines = []
    for i, line in enumerate(lines):
        if 'print(' in line:
            print_lines.append(f"L{i+1}: {line.strip()}")
            
    for pl in print_lines[:20]:
        print(pl)
    print(f"Total print lines: {len(print_lines)}")

if __name__ == '__main__':
    main()
