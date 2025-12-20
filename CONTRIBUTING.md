# Contributing
If you want to contribute to the source code:

1. Fork the repository on :simple-github:.
1. Create a new branch where to commit your changes.
1. Clone your branch locally.
    ```bash
    git clone -b <your_branch> https://github.com/{your-account}/centered-kernel-alignment
    cd centered-kernel-alignment
    ```
1. Create a virtual environment and activate it.

Install the package in editable mode with development dependencies
```bash
# You can either use uv
uv pip install -e .  --group dev

# Or pip
pip install -e . --group dev
```

It is advised to also install the pre-commit hooks in order to stop your commit if some check does not pass
```bash
# Install the hooks
prek install

# Update the hooks if needed
prek autoupdate
```
You're now ready to apply your desired changes. Feel free to open a pull request after you have commited your changes :heart:.
