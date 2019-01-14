# PAM + K-MEANS++
This is my implementation of PAM(with k-means++) using Python's numpy.  
  
Input:  
>distance_matrix - a numpy array  
>k - desired number of medoids  
>verbose - 1 for detailed printing of each iteration  
>iteration - limiting the number of iterations, default (0) - no limitation  
Output:  
>List of clusters(lists)  

K-means++ is a sophisticated way of choosing the initial medoids;  
First medoid is chosen randomly, while the next ones are chosen based on a specific probability.  

Each iteration of PAM:  
>We calculate clusters based on the given medoids  
>Update medoids within the selected clusters  
  
We stop when:  
>Clusters have not been changed  
>Max number of iterations have been reached  
