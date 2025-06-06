# Machine Learning Algorithms Implementation

Implementation and comparison of classical ML algorithms from scratch: K-Nearest Neighbors and Fisher's LDA (two approaches), tested on the Iris dataset.

## Project Structure

```
amp/
├── knn.ipynb           # K-Nearest Neighbors implementation
├── lda.ipynb           # Fisher's LDA implementations and analysis
└── iris/               # Iris dataset files
```

## Algorithms

### K-Nearest Neighbors (`knn.ipynb`)
- Custom implementation with Euclidean distance
- Configurable k parameter with majority voting

### Fisher's LDA (Two Implementations)

**Implementation 1 (Cell 6)**: Comprehensive FisherLDA class
- Solves generalized eigenvalue problem: `Sw^(-1) * Sb * w = λ * w`
- Uses pseudoinverse for numerical stability

**Implementation 2 (Cell 9)**: One-vs-rest classification using FisherLDA
- Applies binary LDA for each class vs all others
- Uses midpoint threshold for decision boundary


## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook knn.ipynb  # or lda.ipynb
```

## Usage

```python
# K-Nearest Neighbors
knn = KNearestNeighbors(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Fisher's LDA
lda = FisherLDA(n_components=1)
X_projected = lda.fit_transform(X_train, y_train_binary)
results = one_vs_rest_lda(X_train, y_train, X_test, y_test)
```

## Results

KNN demonstrates high accuracy on the Iris dataset with stable performance across different k values. Fisher's LDA can easily differenciate the Iris-setosa but encounters challenges when linearly separating the remaining two classes.

## License

This project is available for educational use. Please refer to individual dataset licenses for the Iris data.

## References

1. Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems"
2. Duda, R. O., Hart, P. E., and Stork, D. G. (2001). "Pattern Classification"
3. UCI Machine Learning Repository - Iris Dataset
