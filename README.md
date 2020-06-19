# DLAP

A static website built in mkdocs and mkdocs-material for DLAP. DLAP supports automatically build and deploy. The `master` branch is the trigger, once you push your update into this branch the static sites will be automatically built and deployed
into the branch of `gh-pages`.

## Requirements
```
pip install -r requirements.txt
```

## Local Development

```
cd DLAP/

# serve the site locally
mkdocs serve

# build the static site
mkdocs build
```