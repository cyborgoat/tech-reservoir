# Sphinx

[Documentation](https://www.sphinx-doc.org/en/master/index.html)

## First steps

[Reference](https://www.sphinx-doc.org/en/master/usage/quickstart.html)

### Numpy Style docs

1. After setting up Sphinx to build your docs, enable napoleon in the Sphinx conf.py file:

```python
# conf.py

# Add napoleon to the extensions list
extensions = ['sphinx.ext.napoleon']
```

2. Use sphinx-apidoc to build your API documentation:

```shell
$ sphinx-apidoc -f -o docs/source projectdir
```

[Reference](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)