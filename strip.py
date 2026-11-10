import os
import re

directories = ['C:/Users/Adi/AM/core', 'C:/Users/Adi/AM/utils', 'C:/Users/Adi/AM']
files_to_process = []

for d in directories:
    for f in os.listdir(d):
        if f.endswith('.py') and os.path.isfile(os.path.join(d, f)):
            files_to_process.append(os.path.join(d, f))

for filepath in files_to_process:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^#\s*[=|-]{5,}', stripped):
            continue
        new_lines.append(line)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
print('Done stripping separators')
