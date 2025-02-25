# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 14:48:41 2022
Script for generating femtoscope documentation using pdoc.

For numpy style syntax, visit:
https://numpydoc.readthedocs.io/en/latest/example.html

For pdoc, visit:
https://github.com/mitmproxy/pdoc
https://pdoc.dev/docs/pdoc.html

Warning: be sure to use a pdoc version >= 14.5.1 as
previous versions are subject to the Polyfill issue. See
https://sansec.io/research/polyfill-supply-chain-attack
https://github.com/advisories/GHSA-5vgj-ggm4-fg62

"""

import os
import shutil

import pdoc

import femtoscope
from femtoscope import FEMTOSCOPE_BASE_DIR
from femtoscope import IMAGES_DIR
from femtoscope import INSTALL_DIR

if __name__ == '__main__':

    output_directory = FEMTOSCOPE_BASE_DIR / 'doc'

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

    pdoc.pdoc(INSTALL_DIR, output_directory=output_directory)
