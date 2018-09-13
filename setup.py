from setuptools import setup, find_packages


setup(name='pytorch-rnng',
      version='0.0.1',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      python_requires='>=3.6, <4',
      )
