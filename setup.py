from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='amcp',
    version='1.0',
    description='Aggregated Mondrian Conformal Predictor',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
    'Development Status :: Beta',
    'License :: MIT License',
    'Programming Language :: Python :: 3.8',
    "Operating System :: OS Linux",
    'Topic :: Molecular Mechanics :: Docking :: Machine Learning :: Conformal Predictor :: ML Classifier',
    ],
    keywords='docking, amcp, classifier',
    url='https://github.com/carlssonlab/conformalpredictor.git',
    author='Israel Cabeza de Vaca Lopez, Leonard Sparring, Andreas Luttens, Ulf Norinder',
    author_email='israel.cabezadevaca@icm.uu.se, andreas.luttens@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'markdown', 'pandas', 'numpy', 'scipy', 'scikit-learn'
    ],
    entry_points={
        'console_scripts': ['amcp=amcp.amcp:main', 'amcp_preparation=scripts.amcp_preparation:main', 'amcp_inputFromDock=scripts.amcp_inputFromDock:main'],
    },
    include_package_data=True,
    python_requires='>=3.6',
    zip_safe=False)


