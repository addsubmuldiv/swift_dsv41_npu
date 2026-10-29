import os
import sys

import setuptools

if sys.version_info < (3,):
    raise RuntimeError("Python 2 is not supported by TransformerEngineNPU.")

__description__ = (
    'Transformer Engine NPU - A lightweight framework for Transformer models on Ascend NPUs'
)
__version__ = '2.17.1'
__author__ = 'Ascend'
__long_description__ = (
    'Transformer Engine NPU - A lightweight framework for Transformer models on Ascend NPUs'
)
__url__ = 'https://gitcode.com/Ascend/TransformerEngineNPU'
__download_url__ = 'https://gitcode.com/Ascend/TransformerEngineNPU/release'
__keywords__ = 'Ascend, transformer, npu, deep learning, NLP'
__license__ = 'See https://gitcode.com/Ascend/TransformerEngineNPU'
__package_name__ = 'transformer_engine'
__contact_names__ = 'Ascend'

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ''

###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

cmd_class = {}
exts = []


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformer_engine')

setuptools.setup(
    name=__package_name__,
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__url__,
    author=__contact_names__,
    maintainer=__contact_names__,
    license=__license__,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        "torch>=2.1.0",
        "torch-npu",
        "triton-ascend",
        "einops",
        "packaging",
        "nvdlfw-inspect",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_package_data=True,
    exclude_package_data={'': ['**/*.md']},
    package_data={'': package_files(src_path)},
    zip_safe=False,
    keywords=__keywords__,
    cmdclass={},
    ext_modules=exts,
)
