from setuptools import setup, find_packages

setup(
    name='AutoCellDetect',
    version='0.1.0',
    description='Automatic Cell Detection for Microscopic Images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Kathleen De Key',
    author_email='kkey35@gatech.edu',
    packages=find_packages(),
    install_requires=[
        'pandas','cv2','numpy'
    ],
)