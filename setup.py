import os
from setuptools import setup


# Utility method to read the README.rst file.
def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


CLASSIFIERS = [
    'Development Status :: ????',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',  # TODO: test these versions
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering'
]

install_requirements_list = [
    'arviz == 0.5.1',
    'graphviz == 0.13',
    'matplotlib == 3.1.1',
    'numpy >= 1.17.2',
    'pandas >= 0.25.1',
    'pymc3 == 3.7',
    'scipy >= 1.3.1',
    'seaborn == 0.9.0',
    'Theano == 1.0.4',
]

setup(
    name='hypythesis',
    version='0.0.1',
    description='Bayesian Assessment of Hypotheses ',
    long_description=read('README.md'),
    url='https://github.com/allenai/HyPyThesis',
    author='Erfan Sadeqi Azer, Daniel Khashabi',
    author_email='esamath@gmail.com',
    license='BY-NC-SA',
    keywords="Bayesian Statistics, two groups test, Hypothesis Testing, Bayes Factor"
             "NLP, natural language processing, ",
    packages=['HyPyThesis'],
    classifiers=CLASSIFIERS,
    install_requires=install_requirements_list,
    zip_safe=False
)

