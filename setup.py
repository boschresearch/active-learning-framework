from setuptools import find_packages, setup

from alef import __version__

name = "alef"
version = __version__
description = "Active Learning framework"
url = "https://sourcecode.socialcoding.bosch.com/projects/BCAI_R11/repos/active-learning-framework/browse"

setup(name=name, version=version, packages=find_packages(exclude=["tests"]), description=description, url=url)
