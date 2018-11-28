from setuptools import setup

setup(
    name="LpsToolbox",
    version="0.1dev",
    long_description=open('README.md').read(),
    license='GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007', install_requires=['scikit-learn', 'pandas', 'keras',
                                                                                    'seaborn', 'joblib']
)