import io
import os

from setuptools import find_packages
from setuptools import setup


name = 'baleian-ml-keyword_classifier'
description = 'Korean keyword classifier implemented by tensorflow'
release_status = 'Development Status :: 3 - Alpha'
dependencies = [
    'pandas',
    'numpy',
    'scikit-learn',
    'tensorflow==2.5.0',
    'tensorflowjs==3.7.0'
]

root_path = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(root_path, 'baleian/ml/keyword_classifier/version.py')) as f:
    exec(f.read(), version)
version = version['__version__']

readme_filename = os.path.join(root_path, 'README.rst')
with io.open(readme_filename, encoding='utf-8') as readme_file:
    readme = readme_file.read()

packages = [
    package for package in find_packages() if package.startswith('baleian')
]

namespaces = ['baleian', 'baleian.ml']

setup(
    name=name,
    version=version,
    description=description,
    long_description=readme,
    author='baleian',
    author_email='baleian90@gmail.com',
    license='MIT',
    url='https://github.com/baleian/python-ml-keyword_classifier',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    packages=packages,
    namespace_packages=namespaces,
    install_requires=dependencies,
    python_requires='>=3.6'
)
