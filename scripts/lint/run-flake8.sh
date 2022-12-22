#!/bin/bash
ls
if [[ $(poetry config virtualenvs.create) = true ]]
then
    poetry run flake8 ./toydl
else
    flake8 ./toydl
fi
