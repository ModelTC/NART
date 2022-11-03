# Copyright 2022 SenseTime Group Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
import sys
from nart import __version__


def read_requirements():
    reqs = []
    with open("requirements.txt", "r") as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


project_name = "nart-py"
if "--project-name" in sys.argv:
    project_name_idx = sys.argv.index("--project-name")
    project_name = sys.argv[project_name_idx + 1]
    sys.argv.remove("--project-name")
    sys.argv.pop(project_name_idx)


version = __version__
if "--version-suffix" in sys.argv:
    version_suffix_idx = sys.argv.index("--version-suffix")
    version_suffix = sys.argv[version_suffix_idx + 1]
    sys.argv.remove("--version-suffix")
    sys.argv.pop(version_suffix_idx)
    version += version_suffix


setuptools.setup(
    name=project_name,
    version=version,
    author="compile-link",
    author_email="compile-link@sensetime.com",
    description=(""),
    packages=setuptools.find_packages(
        exclude=[".models", ".models.*", "models.*", "models"]
    ),
    package_data={
        "nart": ["*.so"],
        "nart.art": ["*.so"],
        "nart.art.libmodules": ["*.so"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
    install_requires=read_requirements(),
)
