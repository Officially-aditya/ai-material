#!/usr/bin/env python3
"""
Compile all .tex files into a single master file
"""

import re

# Read the main.tex file
with open('main.tex', 'r', encoding='utf-8') as f:
    main_content = f.read()

# Extract the preamble (everything before \begin{document})
preamble_match = re.search(r'(.*?)\\begin\{document\}', main_content, re.DOTALL)
preamble = preamble_match.group(1) if preamble_match else ''

# Extract the title page and TOC section (from \begin{document} to first \include)
doc_start_match = re.search(
    r'\\begin\{document\}(.*?)\\include\{chapter0\}',
    main_content,
    re.DOTALL
)
doc_start = doc_start_match.group(1) if doc_start_match else ''

# Read all chapter files
chapters = []
for i in range(8):
    try:
        with open(f'chapter{i}.tex', 'r', encoding='utf-8') as f:
            chapters.append(f.read())
            print(f"[OK] Read chapter{i}.tex")
    except FileNotFoundError:
        print(f"[SKIP] chapter{i}.tex not found")

# Read appendix
try:
    with open('appendix.tex', 'r', encoding='utf-8') as f:
        appendix_content = f.read()
        print(f"[OK] Read appendix.tex")
except FileNotFoundError:
    appendix_content = ''
    print(f"[SKIP] appendix.tex not found")

# Combine everything
compiled_content = preamble + '\n'
compiled_content += '\\begin{document}\n'
compiled_content += doc_start
compiled_content += '\n% ============================================================================\n'
compiled_content += '% CHAPTERS (compiled inline)\n'
compiled_content += '% ============================================================================\n\n'

for i, chapter in enumerate(chapters):
    compiled_content += f'\n% --- Chapter {i} ---\n'
    compiled_content += chapter
    compiled_content += '\n\\blankpage\n\n'

compiled_content += '\n% ============================================================================\n'
compiled_content += '% APPENDICES\n'
compiled_content += '% ============================================================================\n\n'
compiled_content += appendix_content

compiled_content += '\n% ============================================================================\n'
compiled_content += '% END DOCUMENT\n'
compiled_content += '% ============================================================================\n\n'
compiled_content += '\\end{document}\n'

# Write the compiled file
with open('AI_From_First_Principles_Complete.tex', 'w', encoding='utf-8') as f:
    f.write(compiled_content)

print(f"\n[SUCCESS] Created AI_From_First_Principles_Complete.tex")
print(f"          Total size: {len(compiled_content):,} characters")
