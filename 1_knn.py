from typing import List, NamedTuple, Dict, Tuple
from collections import Counter, defaultdict
import random
from utils.linear_algebra import Vector, distance
from utils.machine_learning import split_data
import csv
import tqdm
from matplotlib import pyplot as plt

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

class KNN:
    def raw_majority_vote(self, labels: List[str]) -> str:
        votes = Counter(labels)
        winner, _ = votes.most_common(1)[0]
        return winner

    def majority_vote(self, labels: List[str]) -> str:
        """Assumes that labels are ordered from nearest to farthest."""
        vote_counts = Counter(labels)
        winner, winner_count = vote_counts.most_common(1)[0]
        num_winners = len([count
                        for count in vote_counts.values()
                        if count == winner_count])

        if num_winners == 1:
            return winner                     # unique winner, so return it
        else:
            return self.majority_vote(labels[:-1]) # try again without the farthest

    def classify(self, k: int,
                    labeled_points: List[LabeledPoint],
                    new_point: Vector) -> str:

        # Order the labeled points from nearest to farthest.
        by_distance = sorted(labeled_points,
                            key=lambda lp: distance(lp.point, new_point))

        # Find the labels for the k closest
        k_nearest_labels = [lp.label for lp in by_distance[:k]]

        # and let them vote.
        return self.majority_vote(k_nearest_labels)

    def random_point(self, dim: int) -> Vector:
        return [random.random() for _ in range(dim)]

    def random_distances(self, dim: int, num_pairs: int) -> List[float]:
        return [distance(self.random_point(dim), self.random_point(dim))
                for _ in range(num_pairs)]


def main():    
    knn = KNN()
    def parse_iris_row(row: List[str]) -> LabeledPoint:
        """
        sepal_length, sepal_width, petal_length, petal_width, class
        """
        measurements = [float(value) for value in row[:-1]]
        # class is e.g. "Iris-virginica"; we just want "virginica"
        label = row[-1].split("-")[-1]
    
        return LabeledPoint(measurements, label)
    
    with open('data/iris.data') as f:
        reader = csv.reader(f)
        iris_data = [parse_iris_row(row) for row in reader]
    
    # We'll also group just the points by species/label so we can plot them.
    points_by_species: Dict[str, List[Vector]] = defaultdict(list)
    for iris in iris_data:
        points_by_species[iris.label].append(iris.point)
    
    metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
    pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
    marks = ['+', '.', 'x']  # we have 3 classes, so 3 markers
    
    fig, ax = plt.subplots(2, 3)
    
    for row in range(2):
        for col in range(3):
            i, j = pairs[3 * row + col]
            ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
    
            for mark, (species, points) in zip(marks, points_by_species.items()):
                xs = [point[i] for point in points]
                ys = [point[j] for point in points]
                ax[row][col].scatter(xs, ys, marker=mark, label=species)
    
    ax[-1][-1].legend(loc='lower right', prop={'size': 6})
    # plt.show()
    plt.savefig('data/iris_scatter.png')
    plt.gca().clear()
    
    random.seed(12)
    iris_train, iris_test = split_data(iris_data, 0.70)
    assert len(iris_train) == 0.7 * 150
    assert len(iris_test) == 0.3 * 150
    
    confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    num_correct = 0
    
    for iris in iris_test:
        predicted = knn.classify(5, iris_train, iris.point)
        actual = iris.label
    
        if predicted == actual:
            num_correct += 1
    
        confusion_matrix[(predicted, actual)] += 1
    
    pct_correct = num_correct / len(iris_test)
    print(pct_correct, confusion_matrix)
    
    dimensions = range(1, 101)
    
    avg_distances = []
    min_distances = []
    
    random.seed(0)
    for dim in tqdm.tqdm(dimensions, desc="Curse of Dimensionality"):
        distances = knn.random_distances(dim, 10000)  # 10,000 random pairs
        avg_distances.append(sum(distances) / 10000)  # track the average
        min_distances.append(min(distances))          # track the minimum
    
    min_avg_ratio = [min_dist / avg_dist
                     for min_dist, avg_dist in zip(min_distances, avg_distances)]
    
if __name__ == "__main__": 
    main()