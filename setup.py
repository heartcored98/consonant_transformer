from setuptools import setup, find_packages
 
 
 
setup(name='consonant',
 
      version='0.1',
 
      url='https://github.com/heartcored98/consonant_transformer',
 
      license='MIT',
 
      author='Jaeyoung Jo',
 
      author_email='kevin.jo@dingbro.ai',
 
      description='consonant predictor',
 
      packages=find_packages(exclude=['dataset', 'deploy', 'examples', 'output', 'preprocess', 'train']),
 
      long_description=open('README.md').read(),
 
      zip_safe=False,
 
 )