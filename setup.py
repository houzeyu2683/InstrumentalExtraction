import setuptools

name = "InstrumentalExtraction"
setuptools.setup(
    name=name,
    version="0.1.0",
    packages=[
        'InstrumentalExtraction', 
        'InstrumentalExtraction.architecture', 
        'InstrumentalExtraction.machine'
    ],
    package_dir={
        'InstrumentalExtraction': '.', 
        'InstrumentalExtraction.architecture': 'architecture', 
        'InstrumentalExtraction.machine': 'machine'
    },
    install_requires=[
        'gdown==5.2.0',
        'resampy==0.4.3',
        'scipy==1.15.2',
        'soundfile==0.13.1',
        'tqdm==4.67.1',
    ],
    author="Greg",
    author_email="houzeyu2683@gmail.com",
    description="",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/houzeyu2683/InstrumentalExtraction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
