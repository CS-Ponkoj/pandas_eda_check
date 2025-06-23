from setuptools import setup, find_packages

setup(
    name='pandas_eda_check',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['pandas'],
    description='Quick EDA utility to summarize unique and missing values in pandas DataFrames.',
    author='Ponkoj Shill',
    author_email='csponkoj@gmail.com',
    url='https://github.com/CS-Ponkoj/pandas_eda_check',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)