from setuptools import find_packages
from setuptools import setup

setup(
    name='nna',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/speechLabBcCuny/nnaAudiosetClassification',
    author='Enis Berk Çoban',
    author_email='me@enisberk.com',
    description='NNA project tools'
)
