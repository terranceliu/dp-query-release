from setuptools import setup

setup(
    name='dp-query-release',
    version='1.0',
    description='Generating private synthetic data via query release',
    url='https://github.com/terranceliu/dp-query-release',
    author='Terrance Liu',
    license='MIT',
    packages=['src'],
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'tqdm', 'rdt==0.4.0', 'folktables'],
)