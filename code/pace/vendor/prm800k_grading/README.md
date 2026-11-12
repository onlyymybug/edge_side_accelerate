# PRM800K answer grader

`grader.py` and `math_normalize.py` are vendored from
<https://github.com/openai/prm800k> at commit
`7ecc794703b2877f63226f2477a49b34f9b25163`.

The upstream project is licensed under the MIT License; see `LICENSE`.
The only local source change is a package-relative import of
`math_normalize`, so the grader can be imported from this repository.

Runtime dependencies: `sympy` and `pylatexenc`.
