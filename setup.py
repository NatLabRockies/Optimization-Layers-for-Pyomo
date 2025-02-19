import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
# replace requirements.txt with requirements_laptop.txt when running on the laptop
with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

setup(
    name="OPT_layer",
    package_data={
        "": [
            "*requirements.txt",
        ]
    },
    include_package_data=True,
    install_requires=install_requires,
)
