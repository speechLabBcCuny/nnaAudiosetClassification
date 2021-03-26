"""Info about project and requirements."""

from setuptools import find_packages
from setuptools import setup

# strict
    # 'ffmpeg==4.3',

requirements = [
    'pandas==1.0.3',
    'pydub==0.23.1',
    'pillow==7.1.1',
    'tensorflow-gpu==1.14.0',
    'numpy==1.19',
    'scikit-learn==0.21.3',
    'scipy==1.5.3',
    'keras==2.3.1',
    'matplotlib==3.1.1',
    'pydub==0.23.1',
    'pillow==7.1.1',
    'pyyaml==5.4',
    'sqlite==3.33',
    'pysoundfile==0.10.2',
    'resampy==0.2.2',
]



setup(name='nna',
      version='0.1.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      url='https://github.com/speechLabBcCuny/nnaAudiosetClassification',
      author='Enis Berk Çoban',
      author_email='me@enisberk.com',
      description='NNA project tools',
      install_requires=requirements,
      extras_require={'dev': [
          'pytest',
          'pytest-pep8',
          'pytest-cov',
          'mypy',
      ]})
