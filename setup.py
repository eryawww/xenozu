from setuptools import find_packages, setup

with open("readme.md", "r") as f:
    long_description = f.read()

setup(
    name="xenozu",
    version="0.0.1",
    description="Data science library built through competition experience",
    package_dir={"": "xenozu"},
    packages=find_packages(where="xenozu"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Eryaw",
    author_email="zazaneryawan@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9",
)