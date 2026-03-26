from setuptools import setup, find_packages

setup(
    name='conversion_toolchain',
    version='1.0.0',
    packages=find_packages(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['conversion_toolchain=conversion_toolchain.main:cli']
    },
)
