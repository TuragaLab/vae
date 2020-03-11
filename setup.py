from setuptools import setup

setup(name='vae',
      version='0.1',
      description='Model-agnostic variational autoencoder tools',
      author='Roman Vaxenburg, Srini Turaga',
      author_email='vaxenburgr@janelia.hhmi.org',
      license='MIT',
      url='https://github.com/TuragaLab/vae',
      packages=[
        'vae',
      ],
      install_requires=[
        'pytorch',
        'numpy',
      ])
