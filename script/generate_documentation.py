# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:48:41 2022
Script for generating femtoscope documentation using pdoc.

For numpy style syntax, visit:
https://numpydoc.readthedocs.io/en/latest/example.html

For pdoc, visit:
https://github.com/mitmproxy/pdoc
https://pdoc.dev/docs/pdoc.html

"""

import pdoc
import femtoscope
from femtoscope import IMAGES_DIR

from pathlib import Path
import os
import shutil

if __name__ == '__main__':

    module_path = Path(femtoscope.__file__).parent
    output_directory = Path(femtoscope.__file__).parent.parent / 'doc'

    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)

    doc = pdoc.doc.Module(femtoscope)

    logo_str = (IMAGES_DIR / 'logo.png').as_posix()
    logo_str = 'file://' + logo_str

    pdoc.render.configure(
        docformat='numpy',
        footer_text='Hugo Lévy PhD Thesis',
        logo=logo_str,
        math=True)

    pdoc.pdoc(module_path, output_directory=output_directory)
