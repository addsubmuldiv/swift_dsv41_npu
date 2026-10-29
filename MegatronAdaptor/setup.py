import sys
from setuptools import setup, find_packages

if sys.version_info < (3,):
    raise RuntimeError("Python 2 is not supported by MegatronAdaptor.")

__package_name__ = 'megatron_adaptor'
__version__ = '0.1.0'
__description__ = 'Megatron-LM adaptor for NPU'
__author__ = 'Ascend'
__url__ = 'https://gitcode.com/Ascend/MegatronAdaptor'


setup(
    name=__package_name__,
    version=__version__,
    description=__description__,
    author=__author__,
    url=__url__,
    packages=find_packages(),
    package_data={
        # npu_datadump loads config.json via Path(__file__).parent at runtime,
        # so it must be shipped with the wheel, not only in the source tree.
        'megatron_adaptor.patches.megatron.npu_datadump': ['*.json'],
    },
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.7.1',
    ],
)
