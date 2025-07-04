from setuptools import setup, find_packages

setup(
    name='engine-thermal-visualiser',
    version='0.1.0',
    description='Visualiser for engine thermal data using CAD models and sensor input.',
    author='Jack Johnston',
    author_email='your.email@example.com',  # Optional but standard
    url='https://github.com/yourusername/engine-thermal-visualiser',  # Optional but helpful
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        'cadquery==2.5.2',
        'numpy>=1.22.4',
        'pandas==2.3.0',
        'plotly==6.0.1',
        'pytest==8.4.0',
        'PyYAML==6.0.2',
        'setuptools==59.6.0'
    ],
    entry_points={
        'console_scripts': [
            'run-demo=scripts.run_viewer_demo:main',
        ]
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License', 
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.8',
)
