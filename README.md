# Clustering

Proyecto de Clustering.

- Kmeans 

Ejemplo:

<h3>Compresión de datos</h3>

Podemos usar clustering para comprimir imágenes con pérdida de información. 

<p style="text-align:justify;">
La compresión, en este ejemplo, se hace en el número de colores diferentes que se usan.
Vamos a suponer que la imagen original utiliza una paleta de 255 colores. Para comprimir la imagen, podemos decidir usar menos bits por pixel, es decir, usar menos colores.
En la siguiente figura puedes ver cómo quedaría la foto original a medida que vamos usando menos y menos colores. Primero 7 colores, luego 5, luego 3 y por último sólo 2.
La técnica de clustering nos ayuda a decidir qué nuevos colores pueden representar mejor la imagen original cuando limitamos el número posible de colores a usar (K). </p>


<img src="https://emanuelcanizales.com/images/Clustering.png" alt="Cluster Image"></img>



Referencias:

- https://www.iartificial.net/clustering-agrupamiento-kmeans-ejemplos-en-python/
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
