from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='meltwater',
    version='0.1',
    description='Python client for the meltwater API',
    long_description=readme(),
    classifiers=[
        'Development Status :: 0 - POC',
        'License :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Meltwater',
    ],
    keywords='meltwater',
    url='https://github.com/emergent-analytics/workstreams/tree/master/ws2/meltwater',
    author='Vincent Nelis',
    author_email='vincent.nelis@ibm.com',
    license='MIT',
    packages=[
        'meltwater'
    ],
    install_requires=[
        'requests',
        'jsonschema',
        'wget',
        'pandas'
    ],
    include_package_data=True,
    zip_safe=False
)