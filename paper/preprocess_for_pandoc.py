import re
import sys

input_file = 'palm_vein_docs.tex'
output_file = 'palm_vein_docs_pandoc.tex'

with open(input_file, 'r', encoding='utf-8') as f:
    tex = f.read()

# 1. Remove \resizebox{\textwidth}{!}{% or \resizebox{\columnwidth}{!}{%
#    These wrap tables and pandoc can't parse them
tex = re.sub(r'\\resizebox\{[^}]*\}\{[^}]*\}\{%\s*\n', '', tex)

# 2. Remove the closing } of resizebox (appears after \end{tabular}% on its own line)
#    Pattern: \end{tabular}% followed by } on the next line
tex = re.sub(r'(\\end\{tabular\})%\s*\n\}', r'\1', tex)

# 3. Remove \renewcommand{\arraystretch}{...}
tex = re.sub(r'\\renewcommand\{\\arraystretch\}\{[^}]*\}\n?', '', tex)

# 4. Replace \usepackage[utf8]{vietnam} with \usepackage[utf8]{inputenc}
#    The vietnam package causes pandoc issues
tex = tex.replace(r'\usepackage[utf8]{vietnam}', r'\usepackage[utf8]{inputenc}')

# 5. Remove \centerline wrapper from \includegraphics — pandoc handles centering
tex = re.sub(r'\\centerline\{(\\includegraphics[^}]*\})\}', r'\1', tex)

# 6. Remove fancyhdr commands that pandoc doesn't understand
tex = re.sub(r'\\pagestyle\{fancy\}\n?', '', tex)
tex = re.sub(r'\\fancyhf\{\}\n?', '', tex)
tex = re.sub(r'\\fancyfoot\[C\]\{\\thepage\}\n?', '', tex)
tex = re.sub(r'\\renewcommand\{\\headrulewidth\}\{0pt\}\n?', '', tex)
tex = re.sub(r'\\renewcommand\{\\footrulewidth\}\{0pt\}\n?', '', tex)

# 7. Remove \medskip (pandoc ignores it but it can cause issues)
tex = tex.replace(r'\medskip', '')

# 8. Remove \noindent
tex = tex.replace(r'\noindent', '')

# 9. Fix \textcolor for pandoc (convert to just the text content)
tex = re.sub(r'\\textcolor\{[^}]*\}\{([^}]*)\}', r'\1', tex)

# 10. Simplify \multicolumn for pandoc — keep as-is, pandoc handles basic ones

# 11. Replace \begin{center}...\end{center} around tables with nothing
#     (pandoc handles table centering itself)
# Don't remove these — they're needed for table context

# 12. Fix \begin{table}[H] — remove [H] since pandoc ignores float placement
tex = re.sub(r'\\begin\{table\}\[H\]', r'\\begin{table}', tex)
tex = re.sub(r'\\begin\{figure\}\[H\]', r'\\begin{figure}', tex)

# 13. Fix equation display — ensure $...$ and $$...$$ are preserved
# Pandoc handles these natively

# 14. Remove \def\BibTeX... (custom macro pandoc can't handle)
tex = re.sub(r'\\def\\BibTeX\{.*?\}\n', '', tex, flags=re.DOTALL)

# 15. Replace \BibTeX with BibTeX if used anywhere
tex = tex.replace(r'\BibTeX', 'BibTeX')

# 16. Fix unresolved equation references — pandoc doesn't resolve \ref{eq:...}
#     eq:ca is the 8th equation in the document
tex = tex.replace(r'(\ref{eq:ca})', '(8)')
tex = tex.replace(r'\ref{eq:ca}', '8')

# 17. Fix bibliography — assign sequential numbers to \bibitem and replace \cite
#     Pandoc doesn't auto-number \thebibliography correctly
bibitem_keys = re.findall(r'\\bibitem\{([^}]+)\}', tex)
key_to_num = {}
for idx, key in enumerate(bibitem_keys, 1):
    key_to_num[key] = idx

# Replace \cite{key1}, \cite{key2} patterns (adjacent cites) first
# Then replace individual \cite{key1, key2, ...} with [num1, num2, ...]
def replace_cite(match):
    keys_str = match.group(1)
    keys = [k.strip() for k in keys_str.split(',')]
    nums = []
    for k in keys:
        if k in key_to_num:
            nums.append(str(key_to_num[k]))
        else:
            nums.append(k)
    return '[' + ', '.join(nums) + ']'

tex = re.sub(r'\\cite\{([^}]+)\}', replace_cite, tex)

# Replace \bibitem{key} with [num] formatted text, each on its own paragraph
for key, num in key_to_num.items():
    tex = tex.replace(f'\\bibitem{{{key}}}', f'\n\n[{num}]')

# Fix \begin{thebibliography}{00} to just a section header
tex = re.sub(r'\\begin\{thebibliography\}\{[^}]*\}',
             r'\\section*{Tài liệu tham khảo}', tex)
tex = tex.replace(r'\end{thebibliography}', '')

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(tex)

# Diagnostics
remaining_resizebox = tex.count('resizebox')
remaining_arraystretch = tex.count('arraystretch')
ref_count = len(re.findall(r'\\ref\{', tex))
cite_count = len(re.findall(r'\\cite\{', tex))
table_count = len(re.findall(r'\\begin\{tabular', tex))
figure_count = len(re.findall(r'\\includegraphics', tex))
print(f'Remaining resizebox: {remaining_resizebox}')
print(f'Remaining arraystretch: {remaining_arraystretch}')
print(f'Tables (tabular): {table_count}')
print(f'Figures (includegraphics): {figure_count}')
print(f'\\ref count: {ref_count}')
print(f'\\cite count: {cite_count}')
print(f'Created {output_file}')
