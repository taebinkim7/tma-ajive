from setuptools import setup, find_packages
# the analysis was done with Python version 3.7.2.

install_requires = []

setup(name='tma_ajive',
      version='0.0.1',
      description='Code to implement AJIVE on TMA dataset',
      author='Taebin Kim',
      author_email='taebinkim@unc.edu',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=install_requires,
      zip_safe=False)
