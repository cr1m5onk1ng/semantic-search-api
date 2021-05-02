# Semantic Search Web API

This is a simple dockerized Web API originally created as a semantic search service to use on android devices for my **Natural language processing** projects.

The current API is built with the help of the [FastApi](https://fastapi.tiangolo.com/) framework

<a href="https://fastapi.tiangolo.com/"><img src="https://camo.githubusercontent.com/86d9ca3437f5034da052cf0fd398299292aab0e4479b58c20f2fc37dd8ccbe05/68747470733a2f2f666173746170692e7469616e676f6c6f2e636f6d2f696d672f6c6f676f2d6d617267696e2f6c6f676f2d7465616c2e706e67" width="200"/></a>
<br>
although I'm slowly but steadily working on a Rust version using the [Rocket framework](https://rocket.rs/) as an excuse to learn Rust.

<a href="https://www.rust-lang.org//"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Rust_programming_language_black_logo.svg/1200px-Rust_programming_language_black_logo.svg.png" width="200"/></a>

The model used for the API is adapted from the library [sentence-transformers](https://github.com/UKPLab/sentence-transformers), which offers a wide variety of options for very performant sentence embeddings models, which is in turn based on the popular [transformers](https://huggingface.co/) library by Huggingface

<br> <a href="https://fastapi.tiangolo.com/"><img src="https://repository-images.githubusercontent.com/155220641/a16c4880-a501-11ea-9e8f-646cf611702e" width="200"/></a>

<a href="https://fastapi.tiangolo.com/"><img src="https://www.sbert.net/_static/logo.png" width="200"/></a>

## ✨ Features

- [x] Based on Approximate Nearest Neighbours algorithms
- [x] Given a text query, get the specified number of most similar sentences/documents taken from a corpus
- [x] Add sentences/documents to the corpus
- [ ] Given a query text, find the most similar sentences/document that contain a specific word
- [ ] Perform a word sense disambiguation search to find similar sentences in which the word is used with the same sense

## Author

**Mirco Cardinale**
[Personal website](https://mirco-cardinale-portfolio.herokuapp.com/)

## 🔖 LICENCE

[Apache-2.0](https://github.com/cr1m5onk1ng/nala_android_app/blob/dev/LICENSE)
