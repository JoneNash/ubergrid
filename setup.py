from setuptools import setup, find_packages

setup(name='ubergrid',
      version='0.1.dev',
      packages=find_packages(exclude=["test/", "sample/"]),
      license='MIT',
      author="Tim Renner",
      classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.5"
      ],
      install_requires = [
          'click',
          'scikit-learn>=0.18',
          'toolz',
          'pandas',
          'numpy',
          'sklearn_pandas'
      ],
      entry_points = {
          "console_scripts": ["ubergrid=ubergrid.ubergrid_cli:cli"]
          }
)