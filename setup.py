
# !/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="frisky",
    version="0.1.0",
    description="Risk-Aware Proximal Policy Optimization",
    author="Lauren Moos",
    author_email="lauren@special-circumstanc.es",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    zip_safe=False,
    keywords=["search", "reinforcement learning", "AI"],
    python_requires=">=3.7",
    setup_requires=[],
    entry_points={
        'console_scripts': ['project=frisky.frisky:train']
    },
    classifiers=[
        "Reinforcement Learning",
        "Policy Gradient",
        "Machine Learning",
        "PyTorch Lightning",
    ],
)
