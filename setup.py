from setuptools import setup, find_packages

setup(
  name = 'optimus-prime-transformers',
  packages = find_packages(exclude=['examples']),
  version = '1.1.9',
  license='MIT',
  description = 'optimus-prime - Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  url = 'https://github.com/kyegomez/Optimus-Prime',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'torch',
    'einops'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
