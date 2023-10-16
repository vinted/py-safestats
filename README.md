# Py-safestats

Python version of the R [safestats](https://github.com/AlexanderLyNL/safestats) library.

`Safestats` is for designing and applying safe hypothesis tests. It can be used for designing hypothesis tests in the prospective or randomised controlled trial (RCT) setting, where the tests can be used under optional stopping and experiments often can be stopped early. The 'pilot' functions in the package also enable using the safe tests in the observational/ retrospective setting. For examples and explanation about which test to choose for which study setup, our `walkthrough` can be used. The current version includes safe t-tests and tests of proportions. The initial paper on the theory of safe testing and a worked-out example for the t-test can be found [in this paper](https://arxiv.org/abs/1906.07801). More on the theory behind the development of the safe tests for proportions can be found [here](https://arxiv.org/abs/2106.02693).

## Project installation
You can configure your environment through `pyenv` and `poetry`:
1. Make sure that you have installed both applications (e.g. through `brew`)
2. Clone this repository: `git clone git@github.com:vinted/py-safestats.git`
3. Install a valid Python 3.8.x version, e.g. `pyenv install 3.8.13`
4. Install your environment:
```
pyenv local 3.8.13
poetry env use 3.8.13
poetry install
```
5. Launch a local notebook:
```
poetry shell
jupyter lab
```
6. Check out `walkthrough.ipynb` to explore the package

## Committing

Before committing, install the pre-commit hooks: `pre-commit install`
