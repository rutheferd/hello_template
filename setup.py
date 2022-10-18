from setuptools import setup, find_packages
from io import open
from os import path
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()
# automatically captured required modules for install_requires in requirements.txt and as well as configure dependency links
install_requires = [
    'click',
    'black',
    'flake8',
    'pre-commit',
    'setuptools',
    'stonesoup',
    'plotly',
    'kaleido'
]

setup(
    name="hello_template",
    description="Hello Template is a quick example and template for easily starting a pypi project.",
    include_package_data=True,
    package_data={"": ["*.txt"]},
    version="0.0.1",
    packages=find_packages(),  # list of all packages
    install_requires=install_requires,
    python_requires=">=3.8",  # any python greater than 2.7
    entry_points="""
        [console_scripts]
        hello=hello.__main__:main
    """,
    author="Austin Ruth",
    keywords="template, hello, easy, project, python",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/rutheferd/hello",
    download_url="",
    author_email="aruth3@gatech.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
