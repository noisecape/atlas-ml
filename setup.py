from setuptools import find_packages, setup


def read_requirements(file):
    try:
        with open(file, encoding='utf-8') as f:
            return f.read().splitlines()
    except UnicodeError:
        with open(file, encoding='utf-16') as f:
            return f.read().splitlines()
        

requirements = read_requirements('requirements.txt')

setup(
    name='atlas_ml',
    version='0.1',
    packages=find_packages(),
    description='Atlas-ML, a library with lots of ML/DL architectures implemented.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tommaso Capecchi',
    author_email='tommycaps@hotmail.it',
    url='https://github.com/noisecape/atlas-ml.git',
    install_requires=requirements,
    python_requires='>=3.12.4'
)
