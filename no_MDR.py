
import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from tkinter import *
from tkinter import ttk, messagebox, filedialog, simpledialog
import matplotlib
from matplotlib import pyplot as plt

from os import path



'''
Version 0.4 del programa. 

- Nueva codificación de las incognitas en el fichero de entrada: 0j1d es la incógnita
de desplaz (d) de la barra 0, extremo j, gdl1 (uy)
- por cierto el concepto de gdl se ha eliminado del manual. Hay 3+3 prametros en cada
extremo de barra
- Implementada salida grafica de desplazamientos y M flectores


Heredado de versiones anteriores:

Incorpora comentarios en las ecuaciones adicionales:
- en cualquier linea posterior a la "i_gdl   coef..." (hasta aqui respetar el layout)
- empiezan con "#" 
- pueden estar al ppio de linea (con o sin espacios)
- pueden estar al final de una linea de datos
- las lineas en blanco o que empiezan por (espacios y) "#" no cuentan
- el programa 'sabe' que tiene que leer 6*n_elem ecs
Incorpora en las barras:
- movimientos de fuerza cero ux, uy & fi ("todo" para cubrir grad T transversal).

Resultados en locales para facilitar diagramas:
- el calculo (matrices & su resolucion) se hace siempre en ejes pedidos
- posteriormente se transforma a ejes locales de cada barra
- se guardan  self.a_pedidos  self.a_locales  self.f_pedidos  self.f_locales 

Cambios de ejes. Son los ejes en cada extremo de barra en los que 
voy a dar las ecuaciones. Lo comodo sera hacer coincidir todos los ejes de
extremo de barra que comparten un nudo. Si no habra que andar con senos y 
cosenos como en la v1.0. 

El convenio en el fichero de entrada (ejes_i & ejes_j) es:
- Si es un numero: grados antihorario de los nuevos ejes respecto a los de la barra.
- Si se quieren usar los ejes de la propia barra, poner 0.0 (como en la v0.1)
- Si se quieren usar los ejes de otra barra, poner su numero con el sufijo "b".
- Si se quieren usar ejes xy globales poner simplemente "g"

Si "p" no es nulo, siempre se da en coordenada "y" local de la barra. La carga "F"
se ha eliminado (usar dos barras).

Se entiende como gdl cada movimiento posible de un extremo de barra (hay 3 gdl
en cada extremo). Cada gdl tiene asociado un parametro de desplazamiento y otro
de fuerza. En las ecuaciones se pone el nº del gdl con el sufijo "d" en el primer
caso, y con "f" en el segundo (columnas "i_gdl"). Los ejes a usar son los 
especificados en giro_i & giro_j.

La estructura de una ecuacion es [ [[i_gdl, su_coef]...], [igual_a] ], por ej:
  5f   1.   8f   1.      0.
'''


class Elem:
    # Elemento barra i-j usual. Eje x de i->j; eje y hacia arriba. M+ antihorario 
    # No hay nodos, son pto_i & pto_j, simples coordenadas
    # d0x, d0y, fi0 son movimientos (incrementales entre extremos) de f nula
    def __init__(self, EI, EA, pto_i, pto_j, p, d0x, d0y, fi0):
        self.EI=EI
        self.EA=EA
        self.pto_i= np.array(pto_i, dtype='float64')
        self.pto_j= np.array(pto_j, dtype='float64')
        self.p= p
        self.d0x= d0x
        self.d0y= d0y
        self.fi0= fi0
        self.giro_i= 0.
        self.giro_j= 0. # calculo estos giros desde fuera, con los elem creados
        
        self.vec= self.pto_j - self.pto_i
        self.L= np.linalg.norm(self.vec)
        self.cen=  (self.pto_i+self.pto_j)/2.
        self.beta= np.arctan2(self.vec[1], self.vec[0]) # angulo con horiz, en i
        
        self.a_pedidos= np.zeros(6) # estos son para guardar los resultados, como
        self.f_pedidos= np.zeros(6) # se calculan (ejes pedidos), y en ejes locales
        self.a_locales= np.zeros(6) # del elemento (util para diagramas etc)
        self.f_locales= np.zeros(6)
        
        c, c1 = self.EA/self.L, self.EI/self.L
        c2=c1/self.L
        c3=c2/self.L # Matriz de rigidez en locales:
        self.K= np.array([ [   c,     0,    0,    -c,      0,      0  ],
        
                           [   0,   12*c3, 6*c2,   0,   -12*c3,   6*c2],
                           
                           [   0,    6*c2, 4*c1,   0,   -6*c2,    2*c1],
                           
                           [  -c,     0,    0,     c,      0,      0  ],
                           
                           [   0,  -12*c3, -6*c2,  0,    12*c3,  -6*c2],
                           
                           [   0,    6*c2,  2*c1,  0,    -6*c2,   4*c1] ])
        
      

        # por ahora f0 solo con carga p sencillita (poner un nudo si hay F):
        c, c1 = self.p*self.L/2, self.p*self.L**2/12
        self.f0= np.array([ 0., -c, -c1, 
                            0., -c,  c1 ])
        # si hay T etc, los datos son movs y pongo su carga equivalente en f0:
        a0= np.array([0., 0., 0., self.d0x, self.d0y, self.fi0])
        ''' Vale tb con un MSR superpuesto
        a0= np.array([-self.d0x/2, -self.d0y/2, -self.fi0/2, self.d0x/2, self.d0y/2, self.fi0/2])
        '''
        self.f0 -= self.K @ a0
        

    def k_f0_girados(self):
        # venimos aqui con los giros (i,j) que queremos ya calculados
        si,ci = np.sin(self.giro_i), np.cos(self.giro_i)
        bi=np.array([[ci,si],[-si,ci]])
        sj,cj = np.sin(self.giro_j), np.cos(self.giro_j)
        bj=np.array([[cj,sj],[-sj,cj]])
        k_girada = self.K.copy() 
        f0_girada = self.f0.copy() # copias, pa mantener lo que el giro no cambia
        
        k_girada[:, :2] =  k_girada[:,:2] @ bi.T
        k_girada[:, 3:5] = k_girada[:, 3:5] @ bj.T
        k_girada[:2, :] =  bi @ k_girada[:2, :]
        k_girada[3:5, :] = bj @ k_girada[3:5, :]
        
        f0_girada[:2] = bi @ f0_girada[:2]
        f0_girada[3:5]= bj @ f0_girada[3:5]
        
        return (k_girada, f0_girada)


    def a_f_alocales(self):
        # con los a&f en ejes pedidos esto pone su valor en ejes locales
        si,ci = np.sin(self.giro_i), np.cos(self.giro_i)
        bi=np.array([[ci,-si],[si,ci]])
        sj,cj = np.sin(self.giro_j), np.cos(self.giro_j)
        bj=np.array([[cj,-sj],[sj,cj]])
        self.a_locales = self.a_pedidos.copy()
        self.f_locales = self.f_pedidos.copy()
        
        self.a_locales[:2] = bi @ self.a_pedidos[:2]
        self.a_locales[3:5]= bj @ self.a_pedidos[3:5]
        self.f_locales[:2] = bi @ self.f_pedidos[:2]
        self.f_locales[3:5]= bj @ self.f_pedidos[3:5]



def salida_result():
    global elems

    '''
    K_imprimir= K_coo.todense()
    print('\nMatriz en formaro denso:\n')
    for i in range(12*n_elem):
        for j in range(12*n_elem):
            print('{:3.0f} '.format(K_imprimir[i,j]), end='')
        print()

    print('\nColumna igual_a: ')
    for i in range(12*n_elem):
        print('{:3.1f} '.format(igual_a[i]), end='')
    '''
    
    # Una salida de texto por terminal
    for ie in range(len(elems)):
        el=elems[ie]
        print('\nBarra ', ie)
        print('        despl_i      despl_j       fuerza_i        fuerza_j')
        print('# ejes pedidos: #')
        for i in range(3):
            print('    {:10.3e}    {:10.3e}     {:10.3e}      {:10.3e}    '.format(
                el.a_pedidos[i], el.a_pedidos[i+3], el.f_pedidos[i], el.f_pedidos[i+3]))
        print('# ejes locales de la barra: #')
        for i in range(3):
            print('    {:10.3e}    {:10.3e}     {:10.3e}      {:10.3e}    '.format(
                el.a_locales[i], el.a_locales[i+3], el.f_locales[i], el.f_locales[i+3]))

    # La salida grafica, que es la de enjundia
    # primero una con desplazam & deformada

    max_u, long_tipo = 0., 0.
    n_elems= len(elems)
    for el in elems:
        a= max(el.a_locales[0], el.a_locales[1], el.a_locales[3], el.a_locales[4], key=abs)
        a= abs(a)
        if a > max_u: max_u=a
        long_tipo += el.L
    long_tipo /= n_elems
    scal_u= long_tipo/(9*max_u) # la u max se dibujara con 1!9 de la long tipo
    n_tics=21 # numero de tics en x para dibujos de barra  
    c=np.zeros(5) # coefs para polinomio de uy
    uy = lambda x: c[0]+ c[1]*x+ c[2]*x*x+ c[3]*x**3+ c[4]*x**4

    plt.close(fig='Configuracion deformada')
    plt.figure('Configuracion deformada')
    for el in elems:
        plt.plot([el.pto_i[0], el.pto_j[0]], [el.pto_i[1], el.pto_j[1]], color='g',
                linestyle='dashed', linewidth=1.4, alpha=0.5)

        c[0]=  el.a_locales[1]
        c[1]=  el.a_locales[2]
        c[2]= -el.f_locales[2]/(2*el.EI)
        c[3]=  el.f_locales[1]/(6*el.EI)
        c[4]=  el.p/(24*el.EI)

        x_, y_ = [],[] # para los puntos que forman la grafica, en globales
        vec1x = el.vec/el.L
        vec1y = np.array([-vec1x[1], vec1x[0]])
        for x in np.linspace(0., el.L, n_tics): # x en indeformada
            ux_loc = el.a_locales[0] + x*(el.a_locales[3] - el.a_locales[0])/el.L
            uy_loc= uy(x)
            # vec = punto del trazado 
            vec= el.pto_i + vec1x*(x+ux_loc*scal_u) + vec1y*uy(x)*scal_u
            x_.append(vec[0])
            y_.append(vec[1])
        plt.plot(x_,y_, '-', linewidth=2, color='b', alpha=0.3)#, solid_capstyle='round'))
        
        # veo util marcar los extremos de barra desplazados
        vec= el.pto_i + el.a_locales[0]*vec1x*scal_u + el.a_locales[1]*vec1y*scal_u
        plt.plot(vec[0],vec[1], '+b', markersize=8)
        vec= el.pto_j + el.a_locales[3]*vec1x*scal_u + el.a_locales[4]*vec1y*scal_u
        plt.plot(vec[0],vec[1], '+b', markersize=8)
    
    texto = 'Escala de desplazamiento: {:10.3e}'.format(scal_u)
    plt.text( long_tipo/5, -long_tipo/5 , texto ,fontsize=9, ha='left')
    plt.grid(True)
    plt.axis('equal')
    

    # ahora la de momentos flectores

    max_M = 0. # long_tipo, n_elems, n_tics  valen como estan
    for el in elems:
        a= max(el.f_locales[2], el.f_locales[5], key=abs)
        a= abs(a)
        if a > max_M: max_M=a
    scal_M= long_tipo/(6*max_M) # el M max se dibujara con 1!6 de la long tipo
    eps_M, eps = max_M/1.e6, long_tipo/20
    c=np.zeros(3) # coefs para polinomio de Mz
    Mz= lambda x: c[0]+ c[1]*x+ c[2]*x*x

    plt.close(fig='Momentos flectores')
    plt.figure('Momentos flectores')
    for el in elems:
        plt.plot([el.pto_i[0], el.pto_j[0]], [el.pto_i[1], el.pto_j[1]], color='g',
                linestyle='dashed', linewidth=1.4, alpha=0.5)

        c[0]= -el.f_locales[2]  
        c[1]=  el.f_locales[1]
        c[2]=  el.p/2
        
        x_, y_ = [],[] # para los puntos que forman la grafica, en globales
        vec1x = el.vec/el.L
        vec1y = np.array([-vec1x[1], vec1x[0]])
        for x in np.linspace(0., el.L, n_tics): # x en indeformada
            Mz_loc= Mz(x)
            # vec = punto del trazado 
            vec= el.pto_i + vec1x*x - vec1y*Mz_loc*scal_M # "-" para del lado de tracc
            x_.append(vec[0])
            y_.append(vec[1]) 
        plt.plot(x_,y_, '-', linewidth=1.2, color='r', alpha=0.7)

        # rayicas que cierren el diagrama
        if abs(el.f_locales[2])>eps_M:
            plt.plot([el.pto_i[0], x_[0]],[el.pto_i[1], y_[0]], '-',
                    linewidth=1.2, color='r', alpha=0.7)
        if abs(el.f_locales[5])>eps_M:
            plt.plot([el.pto_j[0], x_[-1]],[el.pto_j[1], y_[-1]], '-',
                    linewidth=1.2, color='r', alpha=0.7)
        
        # esta previsto que las cotas de un extremo compartido se superpongan si las
        # barras están alineadas. Va bien en casos "normales", pero quiza no en 
        # todos los casos (M_ext aplicado por ej)
        if (abs(el.f_locales[2])>eps_M): # si Mz != 0 acotar en i
            texto='{:10.3e}'.format(abs(el.f_locales[2]))
            plt.text(x_[0]+eps, y_[0]-eps, texto ,fontsize=9, 
                    ha='center', va='center', rotation=el.beta*180/np.pi)
        if (abs(el.f_locales[5])>eps_M): # idem en j
            texto='{:10.3e}'.format(abs(el.f_locales[5]))
            plt.text(x_[-1]+eps, y_[-1]-eps, texto ,fontsize=9, 
                    ha='center', va='center', rotation=el.beta*180/np.pi)

        # acotar maximo (si hay p)
        if el.p:
            if np.sign(el.f_locales[1])-np.sign(el.f_locales[4]) == 0:
                a= abs(el.f_locales[1])+abs(el.f_locales[4])
                x_corte= abs(el.f_locales[1]) * el.L / a
                # solo acota max si esta un poco lejos del extremo:
                if (abs(x_corte)>el.L/20) and (abs(x_corte-el.L)>el.L/20):
                    M_extr= Mz(x_corte)
                    vec= el.pto_i + vec1x*x_corte - vec1y*M_extr*scal_M
                    texto='max:\n{:10.3e}'.format(Mz(x_corte))
                    plt.text(vec[0]+eps, vec[1]-eps, texto ,fontsize=9, 
                        ha='center', va='center', rotation=el.beta*180/np.pi)


            
            
            
            
            
            
            
            
                    

    plt.axis('equal')
    plt.grid(True)
    
    plt.show()



def pon_giros(el):
    global elems
    
    def saca_giro(a): 
        # una funcion de conveniencia. Usa "el" heredado. Da el giro que hay que girar
        # al xy local para obtener los ejes deseados.
        try:
            num=float(a)*np.pi/180.
        except:
            code= a[-1]
            if code=='g':
                num= -el.beta
            elif code=='b':
                i_el= int(a[:-1])
                num= elems[i_el].beta - el.beta
            else:
                kk=input(f'Error leyendo ejes en elem {i_el}. Abort.')
        return(num)
    el.giro_i= saca_giro(el.giro_i)
    el.giro_j= saca_giro(el.giro_j)


def dime_nom_fich():
    nfcompleto=filedialog.askopenfilename(parent=v0, title='Nombre del archivo')
    if not nfcompleto: return(0)
    n_f= path.basename(nfcompleto)
    v0.title('noMDR - '+n_f)
    return nfcompleto
    
    
def a_resolver(): # se llama con el boton "resolver" para que no haya nombre 
                     # de fich y lo pregunte. El ejemplo llama a resolver() con
                     # argumento, para que no pregunte
    resolver('')


def resolver(nfcompleto):

    global elems, ecs
    elems, ecs = [], []

    if not nfcompleto:  # Obtiene el fichero de datos si no se ha proporcionado
        nfcompleto= dime_nom_fich()
    
    f=open(nfcompleto, 'r')

    # Pone en variables el contenido del fichero, procesado como sigue:
    # - los ejes_i, ejes_j del elem se guardan como el.giro_i, el.giro_j a dar
    #   desde los ejes locales del elem.
    # - los "param" de las ecs se guardan en la ec como la posicion (columna) en 
    #   la matriz global (i ó i+6*n_elem)
    
    # La distincion entre ecuaciones de apoyos/conexiones (usualmente en despl) y
    # ecuaciones de equilibrio (usualmente en fuerzas) es un mindset tuyo. 
    # Nada impide poner ecuaciones que liguen fuerzas con despl (por ej un resorte).

    
    kk= f.readline()    # Titulo, comentario o lo que quieras
    kk= f.readline()    # linea en blanco
    kk= f.readline().split()    # texto "Elems barra" & Num elems
    n_elem=int(kk[-1])
    kk= f.readline() # texto  Nº   EI    EA   x(i)... etc
    for i in range (n_elem):
        kk=f.readline().split()
        EI, EA = float(kk[1]), float(kk[2])
        pto_i, pto_j = [float(kk[3]),float(kk[4])], [float(kk[5]), float(kk[6])]
        p = float(kk[7])
        d0x, d0y, fi0 = float(kk[8]), float(kk[9]), float(kk[10])
        el=Elem(EI, EA, pto_i, pto_j, p, d0x, d0y, fi0)

        el.giro_i, el.giro_j = kk[11], kk[12] 
                                # por ahora guardo los codigos de los giros; cuando tenga todas las barras podre poner los angulos (enseguida)
        elems.append(el)

    # ahora: pongo los giros de ejes en cada extremo de elemento:
    for el in elems:  pon_giros(el)

    kk= f.readline()    # linea en blanco
    kk= f.readline()    # texto "Ecuaciones" (n_ecs se induce)
    n_ecs= 6 * n_elem
    kk= f.readline() # texto "param   coef  ...etc...     igual_a" 
    i_ecs=0
    while i_ecs < n_ecs :
        una_ec=[]
        kk=f.readline()
        try:
            kk = kk[ : kk.index('#')]
        except ValueError:
            pass
        kk=kk.split()
        if len(kk)>2:  # que no sea linea en blanco
            for j in range(0,len(kk)-1, 2):
                param= kk[j]
                error=False # para comprobacion de sintaxix
                try:
                    iel=int(param[:-3])
                except (TypeError, ValueError):
                    error=True
                    break
                if param[-3] not in 'ij':
                    error=True
                    break
                ij= 3 if param[-3]=='j' else 0
                try:
                    i_gdl=int(param[-2])
                    if i_gdl>2: error=True
                except (TypeError, ValueError):
                    error=True
                    break
                if param[-1] not in 'fd':
                    error=True
                    break
                fd= n_ecs if param[-1]=='f' else 0

                a = fd + iel*6 + ij + i_gdl 
                una_ec.append(a)
                una_ec.append(float(kk[j+1]))

            if (error):
                print(f'Error en la ecuación {i_ecs}, incog {j/2}. Abort')
                f.close()
                return(1)

            una_ec.append(float(kk[-1]))
            ecs.append(una_ec)
            i_ecs += 1

    if (i_ecs != n_ecs) :
        print('Error - Numero de ecuaciones no consistente.')
        f.close()
        return(1)
    f.close()

    # construimos las matrices dispersas y el termino de cargas

    n_elem= len(elems)
    ldatos, li, lj = [],[],[]
    igual_a= np.zeros(12*n_elem)


    for ie in range(n_elem):
        el=elems[ie]
        k_girada, f0_girada = el.k_f0_girados()
        for i in range(6):
            for j in range(6):
                ldatos.append(k_girada[i,j])
                li.append(i+6*ie)
                lj.append(j+6*ie)
            igual_a[i+6*ie]= -f0_girada[i]
            ldatos.append(-1.)
            li.append(i+6*ie)
            lj.append(i+6*(ie+n_elem))

    for i in range(len(ecs)):
        ec=ecs[i]
        for j in range(0, len(ec)-1, 2):
            colu= ec[j] # + 6*n_elem ya viene sumado, si procede 
            ldatos.append(ec[j+1])
            li.append(i+6*n_elem)
            lj.append(colu)
        igual_a[i+6*n_elem]= ec[-1]


    # resolucion del sistema de ecuaciones disperso
    K_coo = coo_matrix((ldatos,(li,lj)))
    K_csr = K_coo.tocsr()
    a_f = spsolve(K_csr, igual_a)

    # guardar resultados en elementos 
    for ie in range(n_elem):
        # primero en ejes pedidos:
        el=elems[ie]
        el.a_pedidos= a_f[6*ie : 6*ie+6]
        el.f_pedidos= a_f[6*(n_elem+ie) : 6*(n_elem+ie)+6]
        # y ahora en ejes locales del elemento
        el.a_f_alocales()

    salida_result()
    return(0)


def salida_comprob():
    global elems, ecs

    print('\nBarras:')
    for elem in elems:
        print(elem)
    print('\nEcuaciones:')
    for ec in ecs:
        print(ec)



def presenta_elige():
    global elige,n_f, nfcompleto


    ##### Ventana hija para licencia #####

    def licencia():
        v1=Toplevel(v0)

        texto='''Este programa es Software Libre (Free Software). Como tal se le aplican los términos de la "GNU General Public License", en su versión 2 o bien (como usted prefiera) en una versión posterior. Básicamente, usted puede:

* Usar libremente el programa 
* Realizar copias del mismo y distribuirlas libremente 
* Estudiar su código para aprender cómo funciona 
* Realizar modificaciones/mejoras del programa 

Bajo las siguientes condiciones:  

* Las modificaciones realizadas al programa deben hacerse públicas bajo una licencia como la presente. Así el software puede mejorar con aportaciones realizadas sobre el trabajo de otros, como se hace en la ciencia.
* Las modificaiones y trabajos derivados en general deben incluir el reconocimiento al autor original (y no puede decir que es usted quien ha escrito este programa).
* En este caso, debe mencionar al autor original como: Juan Carlos del Caño, profesor en la Escuela de Ingenierías Industriales de la Universidad de Valladolid (actualmente jubilado).  

Este programa se distribuye con la esperanza de que sea útil, pero SIN NINGUNA GARANTIA, ni siquiera la garantía de comerciabilidad o de adecuación para un propósito 
particular. Lea la GNU General Public License para más detalles.
Usted debería haber recibido una copia de la GNU General Public License junto con este programa. Si no es así, escriba a la Free Software Foundation: 
Inc., 51 Franklin Street, Fifth Floor, 
Boston, MA 02110-1301, USA.'''

        ttk.Label(v1, text='Resumen de términos de licencia', background='#EDECEB',
            font=('', 16)).grid(row=0, column=0, columnspan=2, pady=5)

        tcaja = Text(v1, width=45, height=30,wrap='word', font=('Sans',9),
            background='#EDECEB', foreground='green', border=None, padx=20, pady=12)
        tcaja.grid(column=0, row=1, padx=8, sticky=(N,W,E,S))
        tcaja.insert('1.0',texto)

        scb = ttk.Scrollbar(v1, orient=VERTICAL, command=tcaja.yview)
        scb.grid(column=1, row=1, sticky='ns')
        
        tcaja['yscrollcommand'] = scb.set

        ttk.Button(v1, text='Entendido', width=9, command=v1.destroy).grid(
            column=0, row=2, pady=4, columnspan=2)

        tcaja['state']='disabled'

        v1.grid_columnconfigure(0, weight=1)
        v1.grid_rowconfigure(0, weight=1)
        v1.grid_rowconfigure(1, weight=4)
        v1.grid_rowconfigure(2, weight=1)
        v1.geometry('+240+60')

        v1.focus()
        v1.mainloop()


    ##### Ventana hija para breviario #####

    def breviario():

        v2=Toplevel(v0)
            
        texto='''- Qué hace este programa:

"noMDR" es un programa concebido para analizar estructuras de barras cuyas conexiones (a las que evitaremos llamar "nudos") sean muy atípicas. Lo anterior incluye conexiones mixtas entre barras, barras que apoyan sobre otras barras, etc. Por supuesto puede usarse para analizar estructuras más convencionales pero hay muchos otros programas que serán más amigables para ese fin. Actualmente su ámbito se limita a estructuras planas en régimen lineal.

El programa construye internamente las ecuaciones de comportamiento de las barras. Cualquier otro condicionante que afecte a las fuerzas, momentos, desplazamientos o giros de los extremos de las barras debe ser especificado por el usuario como una ecuación lineal adicional. Así, tanto los apoyos de cualquier tipo como las interconexiones de las barras como las relaciones de equilibrio oportunas, deben proporcionarse en forma de ecuaciones. Esta implementación es necesaria para conseguir la total fexibilidad operativa que se requiere.


- Convenciones generales:

Cada barra tendrá asignado un número que lo identifica en el programa. La numeración comienza en cero y será consecutiva en el orden en que se aporten las barras al programa. Los extremos de una barra se denominan genéricamente "i" (primer extremo), "j" (segundo extremo). El orden es nuevamente relevante, y es aquel en que sean proporcionados en la definición de la barra.

Se asume un eje "x" local en la barra con dirección desde su primer extremo "i" hacia su segundo extremo "j", y un eje "y" local girado 90º en sentido antihorario respecto del anterior. Nótese que i,j son denominaciones genéricas que nunca se concretarán ya que el programa no usa el concepto de nudo. 

Cada extremo de barra tiene asociados tres parámetros de tipo desplazamiento (dos desplazamientos y un giro) y tres parámetros de tipo fuerza (dos fuerzas y un momento). Los giros y momentos son positivos en sentido antihorario.

Por tanto una barra tiene asociados 12 de estos parámetros, que son precisamente las incógnitas a calcular en primera instancia por el programa. Dado que el comportamiento de una barra aporta 6 ecuaciones, cabe concluir que el usuario debe aportar otras ecuaciones adicionales en un número igual a 6 veces el número de barras.


- Cómo se usa:

Se requiere que el usuario edite un fichero de datos con un editor de texto plano de su elección. No se ha implementado una interfaz gráfica de entrada porque la tipología de ecuaciones que deben aportarse es virtualmente ilimitada. Tenga en cuenta que este pretende ser un programa "especializado en uniones complicadas", en que la generalidad debe primar sobre la facilidad de uso. La práctica totalidad de este apartado se dedicará por tanto a la edición del fichero de datos. 

Puede comenzar usando el botón "plantilla", que genera un fichero con valores nulos adecuado para el número de barras que especifique. Puede cambiar a su gusto el espaciado entre cantidades dentro de una linea. Tanto si usted usa esta pequeña ayuda como si no, el fichero debe respetar las siguientes pautas:

-- La primera linea será ignorada. Puede usarla para comentarios legibles de identificación del problema etc.
-- La segunda linea será también ignorada. Se sugiere dejarla en blanco por legibilidad.
-- La tercera linea debe concluir con un número, que será el número de barras del problema. Puede comenzar con un texto arbitrario como "Nº de barras = ", que será ignorado. Separe el número final del texto antecedente con al menos un espacio.
-- La cuarta linea será ignorada. Su uso previsto es servir de cabecera legible para la definición de las barras. 

-- La quinta línea y siguientes (tantas lineas como barras) especifican las propiedades de cada barra. Cada cantidad debe separarse de las demás por uno o más espacios. Las propiedades requeridas son, por este orden: 
-- -- Número de barra. Esta columna siempre debe comenzar por cero e incrementarse en uno cada nueva línea. La plantilla proporciona tal numeración, que en realidad es sólo un recordatorio visual ignorado por el programa (recuerde que el orden en que se definen las barras es en realidad lo que condiciona su número identificativo).
-- -- EI: módulo de Young por el momento de inercia
-- -- EA: módulo de Young por el área de la sección 
-- -- x(i), y(i), x(j), y(j), coordenadas x,y globales de cada extremo i,j, de la barra
-- -- p: carga transversal constante en la barra (positiva según "y" local)
-- -- d0x, d0y, fi0:  desplazamientos de tensión nula en la barra, en coordenadas locales. d0x según el eje "x" local, d0y según el eje "y" local, y fi0 el giro. Se entiende que son los desplazamientos relativos del extremo j respecto del extremo i, lo que nos ahorra aportar tres números. Por ejemplo, en caso de un incremento de temperatura uniforme sólo tendrá valor d0x, pero en caso de una variación de temperatura transversal a la barra, los tres parámetros serán no nulos. Por supuesto estos "desplazamientos iniciales" pueden tener un origen que no sea la temperatura, y que el programa no cuestiona. Para entender con precisión estos términos, sepa que el programa asume internamente una ecuación de comportamiento para la barra del tipo
                        K(a-a0) = f-f0
en donde "K" es la matriz de rigidez de la barra y "a0" representa cualquier movimiento de los extremos que no produzca "f" en ausencia de cargas internas (representadas por los términos de empotramiento perfecto "f0"). Como se ha indicado, de las infinitas posibilidades para "a0" se ha elegido aquella en que "i" tiene movimientos nulos.
-- -- ejes_i, ejes_j: aquí especificamos en qué ejes deseamos operar en los extremos de esta barra. Pueden ser ejes diferentes para ambos extremos. Se asume que las ecuaciones adicionales que aportaremos más tarde usarán esos ejes. El programa resuelve en primera instancia las incógnitas en estos ejes. En general querremos usar los mismos ejes para todos los extremos de barra que participen de una conexión particular (ello simplifica las ecuaciones). Los códigos para especificar estos ejes son "g" para usar x-y globales, "2b" para usar el sistema de ejes local de la barra 2 (que en general será otra barra distinta de la que estamos definiendo). Cambiése 2 por la barra deseada. Finalmente puede ser un número decimal, que expresará el ángulo en grados desde el eje "x" local hasta el eje "x" deseado. Si se desea operar en ejes locales de la barra pondremos 0.0, ó bien 3b si se tratase de la barra 3.

Una vez especificados los datos de las barras puede guardar provisionalmente el fichero y pulsar el botón "ver barras" para obtener un dibujo en el que figura la numeración de cada barra, sus ejes locales (en negro) y sus ejes deseados en cada extremo (en magenta). Puede identificar cuál es el extremo "i" por la orientación del eje local colineal con la barra y porque los ejes locales se dibujan más próximos a ese extremo. Solo se leerán del fichero los datos de las barras (las ecuaciones adicionales etc serán ignoradas si están presentes). Puede por tanto usar esta funcionalidad con seguridad esté el fichero totalmente editado o no lo esté, ya que el mismo no se modifica. Seguidamente puede continuar con la edición de las ecuaciones adicionales teniendo a la vista dicho dibujo.

-- Tras la definición de las barras siguen tres lineas cuyo contenido será ignorado. Se sugiere dejar la primera en blanco por legibilidad. La segunda puede contener un texto como "Ecuaciones adicionales". La tercera puede usarse como cabecera legible para las ecuaciones adicionales que se aportarán seguidamente.

A partir de esta línea definiremos las ecuaciones adicionales. La edición en esta zona permite incluir comentarios y lineas en blanco según las siguientes pautas:
-- En una linea de datos, lo que siga al carácter "#" será ignorado.
-- Una linea en blanco, o que comience por cero o más espacios segidos de "#", será ignorada. 
Con ello se pretende facilitar (mediante comentarios legibles) la identificación de lo que cada ecuación o grupo de ecuaciones adicionales representa físicamente.

                La sintaxis de una ecuación adicional es:
      parámetro  coeficiente   parámetro  coeficiente ... igual_a 

Donde puede haber cualquier número de parejas [parametro  coeficiente]. Como cabe esperar, "parámetro" indica un parámetro incógnita de los existentes en el problema, y "coeficiente" es el número que multiplica al parámetro en esta ecuación. El número "igual_a" es el término independiente de la ecuación. La codificación de los parámetros incógnita comienza por el número de barra, le sigue una letra "i" ó "j" que indica el extremo de la barra, después un número que puede ser "0", "1", ó "2" indicativo de la dirección (eje x, eje y, eje z, el último para giros o momentos), seguido de una letra que puede ser "d" ó "f" para indicar si es parámetro tipo desplazamiento o tipo fuerza. Por ejemplo una ecuación como
  2j0f  1.0  3i0f  1.0    23.7
asumiendo que hemos elegido coordenadas globales para el extremo j de la barra 2 & para el extremo i de la barra 3, expresaría que la suma de las fuerzas en x de ambos extremos de barra debe sumar 23.7. Esto se corresponde con una situación típica de un nudo que conecta dos barras y tiene aplicada exteriormente una fuerza x de valor 23.7.

Cuando haya proporcionado las 6*barras ecuaciones requeridas habrá completado la edición del fichero. Puede añadirle cualquier número de lineas arbitrarias (por ejemplo para guardar resultados ó comentarios adicionales), ya que serán ignoradas.
Por supuesto es responsabilidad del usuario proporcionar el número de ecuaciones adicionales requerido, que las mismas no sean dependientes, etc.

Completado y guardado el fichero, puede pulsar el botón "resolver" para lanzar el análisis del problema.

AUNQUE USTED NO NECESITA SABERLO para usar el programa, en previsión de que quiera modificarlo o adaptarlo (por favor tenga en cuenta la licencia en ese caso), sepa que el programa asigna internamente un número global a cada parámetro incógnita de cada barra, número que obtiene a partir de la codificación usada en el fichero de entrada. El resultado es que la matriz columna de parámetros incógnita tiene en sus 6*n primeras posiciones las incógnitas de desplazamiento, y en las 6*n siguientes las incógnitas de fuerza, siendo n el número de barras. Cada grupo de 6 parámetros contiene una terna con incógnitas del extremo i seguida de otra terna con incógnitas del extremo j. La matriz del sistema de ecuaciones que se resuelve tiene por tanto el siguiente aspecto:

| K_e1                -1             |
|      K_e2                -1        |
|           ...               ...    |
|               K_en              -1 |
|               ...                  |
|        (ecs adicionales)           |
|               ...                  |

donde las Kei son las submatrices de rigidez 6x6 de cada barra (transformadas según los ejes deseados en cada extremo), & los "-1" son submatrices unidad de 6x6. Las ecuaciones adicionales se usan sin transformaciones tal como las especificó el usuario. En general se trata de una matriz muy dispersa ya que las ecuaciones adicionales también suelen serlo. El término independiente del sistema de ecuaciones se construye en base a las cargas y desplazamientos iniciales coherentemente. 


- Unidades:

Como el autor suele decir, "el programa es agnóstico en cuanto a unidades". El usuario debe proporcionar los datos en un sistema coherente de unidades de su elección, y los resultados se obtendrán en ese mismo sistema de unidades.


- Resultados:

-- La salida de texto proporciona los desplazamientos (y giros) y las fuerzas (y momentos) en los extremos de cada barra, en los ejes solicitados y en lo ejes locales de la barra.

-- La figura "Configuración deformada" dibuja la estructura deformada. Por conveniencia se marcan con "+" las nuevas posiciones de los extremos de las barras. En la parte inferior se informa del factor de escala con el que se han dibujado los desplazamientos.

-- La figura "Momentos flectores" muestra las gráficas de momentos flectores de las barras. Los momentos se dibujan del lado de las tracciones. Se acotan valores en los extremos de las barras y en los máximos locales si existen (puede haberlos si hay carga distribuida en la barra). Si en su problema aparecen superpuestas algunas cotas, hacer zoom sobre la zona solventará la situación.

-- Como detalle de conveniencia, el título de la ventana principal cambia para incluir el nombre del fichero especificado.


- El ejemplo pre programado:

Pulsando el botón "Ejemplo" se lanza la resolución de un caso que ilustra el modo de operar y las capacidades del programa. Se le pedirá un nombre de archivo para poder guardar los datos del mismo y que ud. pueda hacer las pruebas que desee posteriormente.
En el documento pdf adjunto a este programa puede encontrar la configuración que corresponde al ejemplo junto con una explicación detallada del mismo, así como las configuraciones que corresponden a algunos otros ejemplos cuyos ficheros también se adjuntan. Se recomienda vivamente que practique con los ejemplos tras leer estas breves notas.
Este botón servirá también para comprobar si la instalación de noMDR fue correcta, es decir si el programa encuentra el intérprete Python y las librerías que necesita. 


- Estado de desarrollo del programa:

Esta versión 04 tiene todas las capacidades que el autor se planteó en la fase de concepción del programa. No obstante pueden incorporarse nuevas capacidades en versiones futuras. 
El autor agradecerá ser informado si se aprecia algún error en el programa. 

Espero que noMDR le sea útil.
JC del Caño
____________________________________
'''
        
        ttk.Label(v2, text='Notas de uso', background='#EDECEB',
            font=('', 16)).grid(row=0, column=0, columnspan=2, pady=5)

        tcaja = Text(v2, width=72, height=35,wrap='word', font=('Mono',9),
            background='#EDECEB', foreground='green', border=None, padx=20, pady=12)
        tcaja.grid(column=0, row=1, padx=8, sticky=(N,W,E,S))
        tcaja.insert('1.0',texto)

        scb = ttk.Scrollbar(v2, orient=VERTICAL, command=tcaja.yview)
        scb.grid(column=1, row=1, sticky='ns')
        
        tcaja['yscrollcommand'] = scb.set

        ttk.Button(v2, text='Entendido', width=9, command=v2.destroy).grid(
            column=0, row=2, pady=4, columnspan=2)

        tcaja['state']='disabled'

        v2.grid_columnconfigure(0, weight=1)
        v2.grid_rowconfigure(0, weight=1)
        v2.grid_rowconfigure(1, weight=4)
        v2.grid_rowconfigure(2, weight=1)
        v2.geometry('+250+70')

        v2.focus()
        v2.mainloop()



    ##### Para el prb por defecto #####

    def el_ejemplo():
        nfcompleto=filedialog.asksaveasfilename(parent=v0, title='Nombre para guardar el archivo de datos')
        n_f= path.basename(nfcompleto)
        v0.title('noMDR - '+n_f)
        f=open(nfcompleto, 'w')
        texto= '''Ejemplo del manual. Barra que apoya en otra con empotr elástico.

 Numero de barras = 3
  Nº   EI    EA      x(i)   y(i)     x(j)  y(j)    p   d0x  d0y  fi0    ejes_i    ejes_j
  0   5.e3   1.e5    0.0    0.0      2.0   1.0   -5.0  0.0  0.0  0.0     0.        0.
  1   5.e3   1.e5    2.0    1.0      4.6   2.3   -5.0  0.0  0.0  0.0     0.        g
  2   5.e3   1.e5    2.0    0.0      2.0   1.0    0.0  0.0  0.0  0.0     g         0b

 Ecuaciones adicionales
 incog  coef     incog  coef   ...etc...     igual_a

  0i0f   1.                                     0. # extr izdo (libre)
  0i1f   1.                                     0.
  0i2f   1.                                     0.
  1j0d   1.                                     0. # extremo empotr movil
  1j1f   1.                                     0.
  1j2d   1.                                     0.
  2i0d   1.                                     0. # apoyo abajo (con resorte)
  2i1d   1.                                     0.
  2i2d  6.e3     2i2f    1.                     0.
  
 # Lo que sigue es relativo a la conexión central
 
  0j0d   1.    1i0d  -1.                        0. # continuidad barra superior
  0j1d   1.    1i1d  -1.                        0.
  0j2d   1.    1i2d  -1.                        0.
  0j0f   1.    1i0f   1.                        0. 
  0j1f   1.    1i1f   1.    2j1f   1.           0.
  0j2f   1.    1i2f   1.                        0.
  
                            2j0f   1.           0. # deslizamiento barra 2
               1i1d  -1.    2j1d   1.           0.
                            2j2f   1.           0.
__fin__

        '''
        print(texto, file=f)
        f.close()
        resolver(nfcompleto)
        
        
        

    # Para la plantilla

    def plantilla():
        while True:
            n_elem= simpledialog.askstring(title='Entrada', prompt='Número de barras?: ')
            try:
                n_elem = int(n_elem)
                break
            except (TypeError, ValueError):
                print('Error. Se requiere un número entero.')
            
        nfcompleto=filedialog.asksaveasfilename(parent=v0, title='Nombre para el archivo')
        #n_f= path.basename(nfcompleto)
        f=open(nfcompleto, 'w')
        
        texto=  ' Linea para una descripción legible del problema\n\n'
        texto +=f' Numero de barras = {n_elem}\n'
        texto +='  Nº   EI    EA      x(i)   y(i)     x(j)  y(j)    p   d0x  d0y  fi0    ejes_i    ejes_j'
        print(texto, file=f)
        for i in range(n_elem):
            print('  '+str(i)+'   0.0    0.0     0.0    0.0      0.0   0.0    0.0  0.0  0.0  0.0     g         g', file=f)
        print('\n Ecuaciones adicionales', file=f)
        print(' incog  coef     incog  coef   ...etc...     igual_a', file=f)
        print('       # puede poner comentarios (como este) desde esta linea inclusive', file=f)
        print(f'       # recuerde que debe aportar {6*n_elem} ecuaciones para este problema', file=f)
        print('       # puede borrar estos comentarios (recomendado)', file=f)
        print('\n\n', file=f)
        
        f.close()
        texto=f'Fichero generado correctamente en \n{nfcompleto}'
        messagebox.showinfo(parent=v0,message=texto,title='Confirmación')


    # para ver_barras

    def ver_barras():
        global elems
        elems, ecs = [], [] # reiniciamos todo por si acaso
        
        # Obtiene barras del fichero de datos, codigo corta-pegado de a_resolver():
        
        nfcompleto=filedialog.askopenfilename(parent=v0, title='Archivo con datos de barras')
        if not nfcompleto: return(0)
        n_f= path.basename(nfcompleto)
        v0.title('noMDR - '+n_f)
        f=open(nfcompleto, 'r')

        kk= f.readline()    # Titulo, comentario o lo que quieras
        kk= f.readline()    # linea en blanco
        kk= f.readline().split()    # texto "Elems barra" & Num elems
        n_elem=int(kk[-1])
        kk= f.readline() # texto  Nº   EI    EA   x(i)... etc
        for i in range (n_elem):
            kk=f.readline().split()
            EI, EA = float(kk[1]), float(kk[2])
            pto_i, pto_j = [float(kk[3]),float(kk[4])], [float(kk[5]), float(kk[6])]
            p = float(kk[7])
            d0x, d0y, fi0 = float(kk[8]), float(kk[9]), float(kk[10])
            el=Elem(EI, EA, pto_i, pto_j, p, d0x, d0y, fi0)

            el.giro_i, el.giro_j = kk[11], kk[12] 
                                    # por ahora guardo los codigos de los giros; cuando tenga todos las barras podre poner los angulos (enseguida)
            elems.append(el)
        f.close()
        for el in elems: pon_giros(el)
        
        # elementos cargados y puestos sus giros deseados
        # empieza el dibujeo:
        
        n_elem= len(elems)
        cen_tot, long_tipo = np.zeros(2), 0.
        for el in elems:
            cen_tot += el.cen
            long_tipo += el.L
        cen_tot   /= n_elem # centro de todo
        l_flecha = long_tipo/(n_elem*7)     # ~1/7 de long de barra, pa las flechas
        eps= l_flecha/7                    # ~para una cabeza de flecha
        
        #plt.close(fig='Configuracion deformada')
        #plt.close(fig='Momentos flectores')
        plt.close(fig='Barras definidas')
        plt.figure('Barras definidas')
        for i in range(n_elem):
            el=elems[i]
            traslac= (el.cen-cen_tot)*0.5
            x=np.array([el.pto_i[0], el.pto_j[0]])+ traslac[0]
            y=np.array([el.pto_i[1], el.pto_j[1]])+ traslac[1]
            plt.plot(x,y, '-', linewidth=6, color='b', alpha=0.3, solid_capstyle='round')
            texto= plt.text(el.cen[0]+traslac[0], el.cen[1]+traslac[1], str(i),
                    fontsize=9)
            texto.set_bbox(dict(facecolor='#BEF0BE', alpha=0.5, edgecolor='#BEF0BE'))

            # ejes locales:
            vec1= el.vec/np.linalg.norm(el.vec) # unitario del elem
            plt.arrow(x[0]+el.vec[0]/3.8, y[0]+el.vec[1]/3.8, 
                vec1[0]*l_flecha*1.2, vec1[1]*l_flecha*1.2,
                linestyle='solid', linewidth=1, head_width=eps,
                overhang=0.2, color='k', length_includes_head=True)
            plt.arrow(x[0]+el.vec[0]/3.8, y[0]+el.vec[1]/3.8, 
                -vec1[1]*l_flecha*1.2, vec1[0]*l_flecha*1.2,
                linestyle='solid', linewidth=1,  head_width=eps,
                overhang=0.2, color='k', length_includes_head=True)
            
            # ejes pedidos en i:
            c,s= np.cos(el.beta+el.giro_i), np.sin(el.beta+el.giro_i)
            vec1  = np.array([c,s])
            plt.arrow(x[0], y[0], vec1[0]*l_flecha, vec1[1]*l_flecha,
               linestyle='solid', linewidth=1,  head_width= 1.2*eps,
               overhang=0.4, color='m', length_includes_head=True)
            plt.arrow(x[0], y[0], -vec1[1]*l_flecha, vec1[0]*l_flecha,
               linestyle='solid', linewidth=1, head_width=1.2*eps, 
               overhang=0.4, color='m', length_includes_head=True)
            
            # ejes pedidos en j:
            c,s= np.cos(el.beta+el.giro_j), np.sin(el.beta+el.giro_j)
            vec1= np.array([c,s])
            plt.arrow(x[1], y[1], vec1[0]*l_flecha, vec1[1]*l_flecha,
               linestyle='solid', linewidth=1, head_width=1.2*eps,
               overhang=0.4, color='m', length_includes_head=True)
            plt.arrow(x[1], y[1], -vec1[1]*l_flecha, vec1[0]*l_flecha,
               linestyle='solid', linewidth=1, head_width=1.2*eps, 
               overhang=0.4, color='m', length_includes_head=True)
        
        plt.margins(0.15)
        plt.grid(True)
        plt.axis('equal')
        plt.show()
            
            
            
    ##### La ventana de inicio #####

    # v0.title("noMDR vX.X") lo hago fuera

    cuadro = ttk.Frame(v0, padding='9 3 3 3') 
    cuadro.grid(column=0, row=0, sticky=(N, W, E, S))

    ttk.Label(cuadro, text='noMDR v0.4', font=('', 40)).grid(row=0,
        column=0, columnspan=4)
    ttk.Label(cuadro, text='Conexiones complejas en estructuras', 
        font=('Courier', 16)).grid(row=1, column=0, columnspan=4)
    ttk.Label(cuadro, text='by:   Juan Carlos del Caño\n').grid(row=2,
        column=0, columnspan=4)

    # hago la parte izda de la ventana

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(column=0, row=3,
        columnspan=4, sticky='ew')
        
    texto=  'Esto es Software Libre (Free Software), lo cual conlleva\n'
    texto +='derechos y también obligaciones. Por favor lea la licencia.'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=4, 
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=5, column=0,
        columnspan=3, sticky='ew')
        
    texto=  'Valgan estas breves notas como manual de usuario. Se sugiere\n'
    texto +='que las consulte y después juegue con el ejemplo.'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=6, 
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=7, column=0,
        columnspan=3, sticky='ew')
        
    texto = 'Obtenga un archivo de texto que sirva como plantilla para\n'
    texto +='su problema. Se genera con todos los valores nulos.'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=8,
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=9, column=0,
        columnspan=3, sticky='ew')
        
    texto = 'Vea las barras que haya definidas en un archivo. Puede usar\n'
    texto +='un archivo incompleto (que solo especifique las barras).'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=10,
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=11, column=0,
        columnspan=3, sticky='ew')
        
    texto = 'Resuelva un problema cuyos datos estén en un archivo. Se le\n'
    texto +='pedirá que especifique la ubicación del archivo.'
    ttk.Label(cuadro, text=texto, foreground='green').grid(row=12,
        column=0, columnspan=3, sticky='w')

    ttk.Separator(cuadro,orient=HORIZONTAL).grid(row=13,column=0,sticky='ew')
    ttk.Separator(cuadro,orient=HORIZONTAL).grid(row=13,column=2,sticky='ew')
    ttk.Label(cuadro, text='o bien:').grid(row=13,column=1)
        
    texto = 'Puede resolver un ejemplo preprogramado internamente y más\n'
    texto +='tarde jugar con su fichero de datos. O simplemente puede\n'
    texto +='usarlo para comprobar si la instalación de noMDR fue correcta.'

    ttk.Label(cuadro, text=texto, foreground='green').grid(row=14,
        column=0, columnspan=3, sticky='w')
        
    ttk.Separator(cuadro, orient=HORIZONTAL).grid(row=15, column=0,
        columnspan=4, sticky='ew')


    # ahora hago la parte derecha

    ttk.Separator(cuadro,orient=VERTICAL).grid(row=3,column=3,
        rowspan=13, sticky='ns')
    ttk.Button(cuadro, text='Licencia', command=licencia).grid(
        row=4, column=3)

    ttk.Button(cuadro, text='Notas de uso', command=breviario).grid(
        row=6, column=3)

    ttk.Button(cuadro, text='Plantilla', command=plantilla).grid(
        row=8, column=3)

    ttk.Button(cuadro, text='Ver barras', command=ver_barras).grid(
        row=10, column=3)

    ttk.Button(cuadro, text='Resolver', command=a_resolver).grid(
        row=12, column=3)

    ttk.Button(cuadro, text='Ejemplo', command=el_ejemplo).grid(
        row=14, column=3)

    for hijo in cuadro.winfo_children():
        hijo.grid_configure(padx=12, pady=8)

    v0.geometry('+70-70')
    v0.focus()
    v0.mainloop()
    
    return()

###################### Fin de ventana de inicio #####################


elems, ecs = [], []

v0=Tk()
v0.title('noMDR v0.4')
i_exit= presenta_elige()
print('i_exit= ', i_exit)







