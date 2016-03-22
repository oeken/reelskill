#!/bin/bash
cd /Users/oeken/Desktop/ReelSkill/
echo 'Generating API Documentation'
sphinx-apidoc -f -o docs/source/apidoc/ source/
cd docs
make html
cd build/html
open index.html
echo 'Done'
