import re

def main():
    file_path = r'c:\Research\Research\PalmVein\train.py'
    out_path = r'c:\Research\Research\PalmVein\vietnamese_lines.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
            
    vietnamese_pattern = re.compile(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]')
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            if vietnamese_pattern.search(line) or 'NaN/Inf' in line:
                f.write(f"L{i+1}: {line}")

if __name__ == '__main__':
    main()
