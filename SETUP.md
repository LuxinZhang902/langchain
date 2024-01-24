# Setup

1.download all needed requirements

```python
cd 3TAB
pip install -r requirements.txt
```

2. open jupyter notebook

```python
pip3 uninstall jupyterlab
pip3 install jupyterlab
jupyter lab build
jupyter lab
jupyter notebook
```

3. Jupyter Notebook

- open new notebook .ipynb
- Adding encoding='latin-1'
  loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt', encoding='latin-1')
