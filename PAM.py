import numpy
import random

# Input:
#   distance_matrix - a numpy array distance matrix
#   k - desired number of clusters
#   iteration - limiting the number of iterations, default(0) - no limitation
#   verbose - 1 for detailed printing of each iteration
# Output:
#   list of clusters(lists)

def pam(k, distance_matrix, iterations=0, verbose=0):
    # Initializing 
    clusters = [[] for x in range(k)]
    number_of_items = distance_matrix.shape[0]
    iteration = 0
    calculate = True
    
	# Using K-Means++ to initialize medoids
    medoids = kmeanspp(distance_matrix, k)
    
    if verbose == 1: print("K-means++ chosen medoids are: " + str(medoids))
    
    # Stop if: medoids have not changed or given number of iterations has passed
    while calculate:
		iteration += 1
		# Calculate clusters based on given medoids
		clusters = update_clusters(medoids, number_of_items, distance_matrix)
		if verbose == 1: print("Iteration " + str(iteration) + " clusters: " + str(clusters))
		# Recalculate medoids for given clsters
		update_medoid_indexes = update_medoids(clusters, distance_matrix)
		
		# Check whether the medoids have changed
		if medoids == update_medoid_indexes:
		    calculate = False
		# Check whether we have passed the given number of iterations
		if iterations != 0 and iterations<iteration:
		    calculate = False
		    
		medoids = update_medoid_indexes
    
    return clusters

# =============================================================
#                   Auxiliary functions
# =============================================================
def distance(i, j, matrix):
    return matrix.item(i, j)
    
# Calculating clusters based on the distance matrix and the selected medoids
def update_clusters(medoids, number_of_items, matrix):
    # initializing clusters, each starting with it's medoid
    clusters = [[medoids[i]] for i in range(len(medoids))]
    for index_point in range(number_of_items):
        if index_point in medoids:
            continue

        index_of_closest = -1
        shortest_distance = float('Inf')
        for index in range(len(medoids)):
            dist = distance(index_point, medoids[index], matrix)
            # updating the shortest distance (dist_optim)
            if dist < shortest_distance:
                index_of_closest = index
                shortest_distance = dist
        
        clusters[index_of_closest].append(index_point)
    
    return clusters
    
# returns list of cluster medoids for current number of clusters
def update_medoids(clusters, distance_maxtix):
    medoids = list() # Number o fmedoids equals the number of clusters
    
    for cluster in clusters:
        # find a medoid in this cluster
        medoid = calculate_medoid(cluster, distance_maxtix)
        # add found medoid to the list of medoids
        medoids.append(medoid)
        
    return medoids

# Given cluster is a list of indexes in a cluster
def calculate_medoid(cluster, distance_matrix):
    distances = [0] * len(cluster)
    i=0
    for index in cluster:
        for other_index in cluster:
            if other_index != index:
                distances[i] += distance(index, other_index, distance_matrix)    # Calulating sum of a row [distance from every other member]
        
        i+=1
    index_of_medoid = distances.index(min(distances))           # Medoid is an item where sum of all distances is the lowest
    
    return cluster[index_of_medoid]

# Initializes the first medoids
def kmeanspp(distance_matrix, k):
    number_of_items = distance_matrix.shape[0]
    medoids = list()
    # Choosing first medoid randomly
    medoids.append(random.randint(0, number_of_items - 1))
    
    for medoid in range(k - 1):
        # Choose the k-1 medoids that's left based on their distance from the closest medoid
        distances = calculate_shortest_distances(distance_matrix, medoids, number_of_items)
        probabilities = calculate_probabilities(distances)
        candidates = number_of_items - medoid + 1
        mediod_index = find_next_medoid(distances, probabilities, candidates)
        medoids.append(mediod_index)
        
    return medoids

# Calculates the distances of every item to it's closest medoid
def calculate_shortest_distances(distance_matrix, medoids, number_of_items):
    # calculate distances from each member to it's closest medoid
    distances = numpy.zeros(number_of_items)
    for item in range(number_of_items): # for every item
    	minimal_distance = float('Inf')	
    	for medoid in medoids: 				# find closest medoid
    		distance_to_medoid = distance(item, medoid, distance_matrix)
    		if distance_to_medoid < minimal_distance:
    			minimal_distance = distance_to_medoid
    			
    	distances[item] = minimal_distance
    return distances
    
    
    
# Returns a vector where each entry is the probability of an item being selected
def calculate_probabilities(distances):
    # clculate probabilities based on the calculated distances above
    total_distance = numpy.sum(distances)
    if total_distance != 0.0:
        probabilities = distances / total_distance
        # cumsum - cumulative sum of the elements along a given axis
        return numpy.cumsum(probabilities)
    else:
        return numpy.zeros(len(distances))

# Finds the next medoid based upon given probabilities
def find_next_medoid(distances, probabilities, candidates):
    index_best_candidate = -1
    # For each candidate
    for candidate in range(candidates):
        candidate_probability = random.random()
        index_candidate = 0
        # 
        for index_object in range(len(probabilities)):
            if candidate_probability < probabilities[index_object]:
                index_candidate = index_object
                break

        if index_best_candidate == -1: # random probability was higher than the calculated ones
            index_best_candidate = index_candidate
        elif distances[index_best_candidate] < distances[index_candidate]:
            index_best_candidate = index_candidate

    return index_best_candidate
