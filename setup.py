from setuptools import setup

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

if __name__ == "__main__":
    setup(
        name = "SEDR",
        version = "1.0.0",
        description = "Unsupervised spatially embedded deep representation of spatial transcriptomics.",
        url = "https://github.com/JinmiaoChenLab/SEDR",
        author = "Hang Xu",
        author_email = "xu_hang@immunol.a-star.edu.sg",
        license = "MIT",
        packages = ["SEDR"],
        install_requires = ["requests"],
        zip_safe = False,
        include_package_data = True,
        long_description = """ Long Description """,
        long_description_content_type="text/markdown",
    )
