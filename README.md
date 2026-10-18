# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/drice987/multispec_fitting/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                           |    Stmts |     Miss |   Cover |   Missing |
|------------------------------- | -------: | -------: | ------: | --------: |
| src/multispec/\_\_init\_\_.py  |       10 |        2 |     80% |       5-6 |
| src/multispec/\_\_main\_\_.py  |        3 |        3 |      0% |       1-4 |
| src/multispec/bands.py         |       95 |        0 |    100% |           |
| src/multispec/constants.py     |        2 |        0 |    100% |           |
| src/multispec/data.py          |       85 |        0 |    100% |           |
| src/multispec/magnetization.py |      252 |       40 |     84% |253-254, 285-287, 313-340, 347-354, 360, 376, 378, 380 |
| src/multispec/main.py          |      157 |        8 |     95% |117, 127-128, 132, 209, 279, 281, 304 |
| src/multispec/outputs.py       |      246 |        5 |     98% |282, 300, 310, 338, 417 |
| src/multispec/parameters.py    |      108 |       10 |     91% |97, 100-105, 144-146 |
| src/multispec/spectral\_fit.py |      112 |        3 |     97% |134, 152, 162 |
| **TOTAL**                      | **1070** |   **71** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/drice987/multispec_fitting/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/drice987/multispec_fitting/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/drice987/multispec_fitting/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/drice987/multispec_fitting/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fdrice987%2Fmultispec_fitting%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/drice987/multispec_fitting/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.