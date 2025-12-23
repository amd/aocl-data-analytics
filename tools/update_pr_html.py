#! /usr/bin/python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# pylint: disable = line-too-long invalid-name missing-function-docstring broad-exception-caught
# pylint: disable = consider-using-with missing-module-docstring

# Utility to convert pr.md into an editable HTML document to perform
# out-of-pull-request Peer Reviewing of a branch

import sys
import shutil
import subprocess as sp
import tempfile
import argparse
import re
import datetime
from bs4 import BeautifulSoup

head = """
<html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:w="urn:schemas-microsoft-com:office:word"
xmlns:dt="uuid:C2F41010-65B3-11d1-A29F-00AA00C14882"
xmlns:m="http://schemas.microsoft.com/office/2004/12/omml"
xmlns="http://www.w3.org/TR/REC-html40">

<head>
<meta http-equiv=Content-Type content="text/html; charset=utf-8">
<meta name=ProgId content=Word.Document>
<meta name=Generator content="Microsoft Word 15">
<meta name=Originator content="Microsoft Word 15">
</head>
<body>
"""

tail="""
</body>
</html>
"""

def replace_task_list_content(filename, replacement_string, org):
    with open(filename, 'r', encoding='utf-8') as file:
        html = file.read()

    soup = BeautifulSoup(html, 'html.parser')
    task_lists = soup.find_all('ul', class_='task-list')

    if not task_lists:
        raise ValueError("No <UL> tag(s) with class 'task-list' found.")

    replacement = "!|check!box|!"
    for task_list in task_lists:
        # Replace the entire UL tag with the provided text
        task_list.replace_with(replacement)

    # now use literal injection
    body = str(soup).replace(replacement, replacement_string)
    body = re.sub(r"(Revision\s+[0-9\/]+)", f"\\1  [distilled from {org} on {datetime.datetime.now().date()}]", body, 1)
    stream = head + body + tail

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(stream)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Markdown pr file into MS-Word flavoured HTML file")
    parser.add_argument('-f', '--file', required=True, help="Markdown file to be processed", type=argparse.FileType('r', encoding='utf-8'))
    parser.add_argument('-o', '--out', required=True, help="HTML file to save", type=argparse.FileType('w', encoding='utf-8'))
    # Word version of a clickable checkbox
    checkbox = """
    <p class=MsoNormal><span style='mso-fareast-font-family:"Times New Roman"'>
        <w:Sdt CheckBox="t" CheckBoxIsChecked="f" CheckBoxValueChecked="&#9746;"
    CheckBoxValueUnchecked="&#9744;" CheckBoxFontChecked="MS Gothic"
    CheckBoxFontUnchecked="MS Gothic" ID="1210377956">
        <span style='font-family: "MS Gothic"'>&#9744;</span></w:Sdt></span></p>
    """
    tmp = tempfile.NamedTemporaryFile()
    try:
        # parse and check inputs
        args = parser.parse_args()
        # Step 1: convert markdown to HTML using pandoc
        pandoc = sp.run("pandoc -f markdown -t html -o " + f"{tmp.name}" + " " + f"{args.file.name}", shell=True, check=True)
        # pandoc.check_returncode()
        # Replace UL tags with word checkboxes
        replace_task_list_content(tmp.name, checkbox, args.file.name)
        # Copy to final destination
        shutil.copyfile(tmp.name, args.out.name)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
