import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slidingwindow_generator",  # Replace with your own username
    version="1.0.6",
    author="Jaeyeon Kim",
    author_email="anencore94@gmail.com",
    description="Generate Sliding Window from Time Series Dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anencore94/SlidingWindowGenerator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)