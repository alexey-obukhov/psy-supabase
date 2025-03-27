# use python setup.py sdist bdist_wheel to build the package
import os
from setuptools import setup, find_packages
from setuptools.command.install import install

# Function to download spaCy language model during package installation
def download_spacy_model():
        import spacy
        spacy.load("en_core_web_sm")
        print("Downloaded spaCy English language model")

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        download_spacy_model()

# Read README if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="psy-supabase",
    version="0.1.0",  # Adjust version as needed
    description="Psychological AI with Supabase Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexey-obukhov/psy-supabase.git",
    author="Alexey Obukhov",
    author_email="alexey.obukhov@hotmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.4.1",
        "transformers>=4.46.3",
        "flask==3.0.3",
        "supabase>=2.6.0",
        "python-dotenv>=1.0.1",
        "pandas>=2.0.3",
        "numpy>=1.24.4",
        "httpx[http2]>=0.26.0,<0.28.0",
        "werkzeug==3.0.6",
        "spacy==3.7.5",
        "school-logging @ git+https://github.com/vertok/school_logging.git@main#egg=school-logging",
    ],
    dependency_links=[
        "git+https://github.com/vertok/school_logging.git@main#egg=school-logging",
    ],
    entry_points={
        'console_scripts': [
            'psy-supabase=psy_supabase.__main__:main',
        ],
    },
    python_requires='>=3.8',
    cmdclass={
        'install': PostInstallCommand,
    },
    # License information
    license="MIT",
    license_files=["LICENSE"],
    # Metadata for PyPI
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
