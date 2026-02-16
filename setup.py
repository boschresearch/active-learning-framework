from setuptools import find_packages, setup

from alef import __version__

name = "alef"
version = __version__
description = "Active Learning framework"
url = "https://github.com/boschresearch/active-learning-framework"

setup(name=name, version=version, packages=find_packages(exclude=["tests"]), description=description, url=url)
