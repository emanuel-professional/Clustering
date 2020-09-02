
"""

@author: ramon, bojana
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA
import math

def NIUs():
    return 1494802, 1494946, 1491186
    
def distance(X,C):
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
    

    """ Calculem la diferencia entre els arrays X i C per columnes,
        i guardem els valors a un nou array que és el que retornem. """
  
    K = C.shape[0] # k es numero de files o numero de pixels, es el mateix 
    distanciaCalculada = np.zeros((X.shape[0],K))
    acumulador=np.zeros(())
    for index in range(K): # recorrem tots els centroids, farem el calcul de la distancia euclidiana 
        # formula de la distancia euclidiana, resta cadascuna de les files de X per la fila[index] de centroid, es a dir, primer calculem per cada uns dels centroids , es equivalent a fer (x2-x1)    
        acumulador=((X-C[index])**2).sum(axis=1) # fem la corresponent suma per cada columna, es a dir, (x2-x1) ens dona un valor, per exemple i ho guardem a l'acumulador. Si les coordenades de cada centroid i punt son [x,y,z] tindrem y1+y2+y3 en altres paraules, (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2.  aixo es fa es dues lines de codi gracies a numpy, indicant fem servir axis=1 pero a que es faci columna a columna  
        distanciaCalculada[:,[index]]= np.sqrt(np.reshape(acumulador,(acumulador.shape[0],1)))
    return distanciaCalculada

class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """
        self._init_original_image(X)
        self._init_X(X)                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object
        
    def _init_original_image(self,X):
        self.original_image=X
        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
        
        if (len(X.shape) == 3): ##Si ens passen la imatge amb les 3 coordenades
            N = X.shape[0]
            M = X.shape[1]
            D = X.shape[2]
            
            P = N*M
            
            self.X = X.reshape((P, D)) ##Inicialitzarem la matriu amb el tamany indicat
        else: ##Si ens passen la imatge amb 2 coordenades
            self.X = X
            
    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first' # Editar para el exercici 7
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'fisher'


        self.options = options
        
        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K                                             # INT number of clusters
        if (self.K>0):
            self._init_centroids()                             # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster
            self._cluster_points()                             # sets the first cluster assignation
        else:
            self.bestK()
        self.num_iter = 0                                      # INT current iteration
        

    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
        
        listOfCentroids = []
        counter = 0
        index = 0
        if self.options['km_init'].lower() == 'first':           
            while (counter != self.K):
                found = False
                pixel = self.X[index].tolist()
                for centroid in listOfCentroids:
                    if (pixel == centroid):  
                        found = True      
                if not found:
                    counter = counter + 1
                    listOfCentroids.append(pixel) 
                index = index + 1
            self.centroids = np.array(listOfCentroids)

        else:
            self.centroids = (np.random.rand(self.K,self.X.shape[1])) 
        self.centroids=self.centroids.astype(float) 
        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X.
        """
        
        """ Calculem l'array de distancies entre els píxels de X i les píxels dels centroids.
            Després guardem a clústers l'array d'índexs dels mínims d'aquestes distancies per files"""
        distMatrix=distance(self.X,self.centroids) ##Retorna una llista que ens diu, per cada punt, la distancia a tots els centroids. Despres escollirem el punt mes petit
        self.clusters=np.argmin(distMatrix,axis=1) ##Agafa els elements de la llista distance (de la forma [[1,2,3],[3,4,5]]) i per cada subllista, retorna la posicio que te el valor minim
        ##AXIS per columnes --> perque ho faci be
  
    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
        
        """ Recalculem els nous punts centroids. Assignem els self.old_centroids, buidem els
            self.centroids i després, agrupem per cada centroid els clústers, i després realitzem
            la mitjana per columnes per calcular les noves posicions dels píxels dels centroides. """
           
        self.old_centroids=np.copy(self.centroids)
        for i in range(len(self.centroids)):     
            newList=self.X[self.clusters==i]##Guarda a llistaNova tots els punts que pertanyen al cluster corresponent (retornant el index de on es troba a self.clusters i buscantlos a self.X)
            if len(newList)>0:  
                mitjana=np.mean(newList, axis=0)##Fa la mitjana entre tots els pixels ([1,2,3], [4,5,6])que hi hagi a llista nova
                self.centroids[i]=mitjana
         
    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
        
        """ Realitzem un nou array que contingui les distàncies entre els centroid d'aquesta
            iteració i de la iteració anterior. Després recorrem l'array sencer, i anem comparant
            si algún valor supera la tolerància màxima definida a les opciones del K-Means. """
           
        
        distMatrix = distance(self.centroids, self.old_centroids)
        for i in range (distMatrix.shape[0]):
            if (distMatrix[i][i] > self.options['tolerance']):
                return False
        return True
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """ 
        
        if (self.K==0):
            self.bestK()
            return
                 
        self._iterate(True)
        self.options['max_iter'] = np.inf
        if self.options['max_iter'] > self.num_iter:
            while not self._converges() :
                self._iterate(False)
        
        
        
        
        """
        if self.K==0:
            self.bestK()
            return
            
        self._iterate(True)
        self.options['max_iter'] = np.inf
        if self.options['max_iter'] > self.num_iter:
            while not self._converges() :
                self._iterate(False)"""
        
    def createFisherList(self):
        fisher = []
        for pk in range(2,11): # possibles K
            self._init_rest(pk)
            self.run()
            fisher.append(self.fitting())
        return fisher
      
    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
        
        # Getting the best K
        
        mfisher=[]
        mdif=[]
        fisher=self.createFisherList()
        
        for i,v in enumerate(fisher[1:]):
            i+=1
            mfisher.append(abs(fisher[i-1]-v)) # valor absoluto
        for i, v in enumerate(mfisher[1:]):
            i+=1
            mdif.append((mfisher[i-1]/v))
        bestK_value=mdif.index(max(mdif))+3
        
        # Show results
        if self.options['verbose']:
            ejex=[2,3,4,5,6,7,8,9,10]
            indexK=ejex.index(bestK_value)
            value=fisher[indexK]       
            plt.plot(ejex,fisher, 'bs-', markersize=7, lw=3) 
            plt.plot(bestK_value,value, 'rD-',markersize=17, lw=3) 
            plt.xlabel('Valors K')
            plt.ylabel('Valor Discriminant Fisher')
            plt.grid(True)
            plt.show()
            
        self.K=bestK_value
        
        if self.options['verbose']:
            plt.figure(1)
            plt.imshow(self.original_image)
            plt.axis('off')
        
        self._init_rest(self.K)
        self.run()     
            
        """
        self._init_rest(4)
        self.run()        
        return 4 """
  
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
        
        if self.options['fitting'].lower() == 'fisher':
            distanceIntraClass=0
            distanceInterClass=0
            
            for k_vintec in range(self.K): # k_vintrc means k_value_inter_class
                distanceInterClass+=np.sqrt(np.sum(((self.centroids[k_vintec,:])-(np.mean(self.X,axis=0)))**2))    
            
            for k_vintrc in range(self.K): # k_vintrc means k_value_intra_class
                value=0
                k=(np.sum(self.clusters==k_vintrc))
                for i in range(k):
                    value+=np.sqrt(np.sum(((self.X[self.clusters==k_vintrc,:][i,:])-(self.centroids[k_vintrc,:]))**2))
                value/=k
                distanceIntraClass+=value
                
            distanceInterClass/=self.K
            distanceIntraClass/=self.K
            
            return distanceIntraClass/distanceInterClass # retorna discriminant de fisher
            
        else:
            return np.random.rand(1)
        

    def plot(self, first_time=True):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=self.X / 255, marker='.', alpha=0.3)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], c=self.centroids / 255, marker='o', s=5000, alpha=0.75, linewidths=1, edgecolors="k")

        textdict = "K: "+str(self.K)+"\nInit: "+str(self.options["km_init"]+"\nIter: "+str(self.num_iter))
        box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text( 200, 100, 200, textdict, transform=ax.transAxes, fontsize=16, horizontalalignment="left", verticalalignment="bottom", bbox=box)

        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        
        plt.show()
        
    def show_image(self):
        
        if self.num_iter == 0:
            plt.imshow(self.original_image)
        else:
            new_image = np.copy(self.X)
            for ctr in range(self.K):
                if new_image[self.clusters == ctr].shape[0] > 0:
                    new_image[self.clusters == ctr] = self.centroids[ctr]
            plt.imshow(new_image.reshape(self.original_image.shape))
            plt.show()
            plt.pause(1)