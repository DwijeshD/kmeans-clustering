from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def perform_kmeans(data, k_range=(2, 10)):
    inertia = []
    models = {}
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        models[k] = kmeans
    return inertia, models

def plot_elbow(inertia, k_range=(2, 10)):
    plt.figure(figsize=(8, 6))
    plt.plot(range(k_range[0], k_range[1] + 1), inertia, marker='o')
    plt.title('Elbow Plot for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()