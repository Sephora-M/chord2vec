from setuptools import setup

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='chord2vec',
      version='3.2',
      description='Embedding chords',
      long_description=readme(),
      url='https://github.com/Sephora-M/chord2vec',
      author='Sephora Madjiheurem',
      author_email='sephora.madjiheurem@gmail.com',
      packages=['chord2vec'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
