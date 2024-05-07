from setuptools import setup, find_packages

setup(
    name='atlas_ml',
    version='0.1',
    packages=find_packages(),
    description='A comprehensive machine learning toolkit with implementations of classical and deep learning algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tommaso Capecchi',
    author_email='tommycaps@hotmail.it',
    url='https://github.com/noisecape/atlas_ml',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'matplotlib',
        'torchinfo'
    ],
    python_requires='>=3.10',
)
