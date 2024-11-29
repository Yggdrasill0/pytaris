from itertools import product
import numpy as np
import os
from itertools import combinations_with_replacement
import pandas as pd


def obtener_valores_numericos(texto):
  """Funcion que permite obtener los valores numéricos de un texto"""


  palabras = texto.split()
  valores_numericos = np.array(())

  for palabra in palabras:
    try:
      valor_float = float(palabra)
      valores_numericos = np.append(valores_numericos, [valor_float])

    except ValueError:
      pass

  return valores_numericos


def obtener_nodos_aristas(nombre_archivo):
  """Funcion que a partir de un archivo .gml me permite generar
  el array de nodos y aristas que caracterizan a un árbol"""


  with open(nombre_archivo, "r") as fo:
    string_arbol = fo.readlines()
    peso_nodos_arbol = []
    aristas_arbol = []
    for num_linea, linea in enumerate(string_arbol):
      if linea[:2] == "id":
        peso_nodos_arbol.extend([
            obtener_valores_numericos(string_arbol[num_linea+4]),
            obtener_valores_numericos(string_arbol[num_linea+5]),
            obtener_valores_numericos(string_arbol[num_linea+6]),
            obtener_valores_numericos(string_arbol[num_linea+7]),
            obtener_valores_numericos(string_arbol[num_linea+8]),
            obtener_valores_numericos(string_arbol[num_linea+9])])

      elif linea[:4] == "edge":
        aristas_arbol.extend([
            obtener_valores_numericos(string_arbol[num_linea+1]),
            obtener_valores_numericos(string_arbol[num_linea+2])])

    peso_nodos_arbol = np.reshape(np.array(peso_nodos_arbol).ravel(), (-1,6))
    aristas_arbol = np.reshape(np.array(aristas_arbol, dtype = int).ravel(), (-1,2))

    return peso_nodos_arbol, aristas_arbol

def encontrar_hijos(nodo, array_aristas):
  """Funcion que me permite encontrar hijos dado el id de un nodo padre,
  si no es padre la función devuelve una lista vacía"""

  indices = np.where(array_aristas[:,1] == nodo)[0]

  return array_aristas[indices][:,0]


def obtener_hermanos_ordenados(nodo, array_aristas, array_pesos):
  """obtengo la lista de hermanos de un nodo ordenados"""

  padre_nodo = [arista[1] for arista in array_aristas if arista[0] == nodo]
  try:
    hermanos = [arista[0] for arista in array_aristas if arista[1] == padre_nodo[0]] #Solo tenemos un padre
  except:
    return []
  hermanos_array = np.array((hermanos))
  pesos_seleccionados = array_pesos[hermanos_array]
  indices = np.argsort(pesos_seleccionados[:,-1])
  hermanos_ordenados = np.take_along_axis(hermanos_array, indices, axis=0)

  return hermanos_ordenados


def nodos_keynodes(array_aristas, array_pesos):
  """Funcion que me proporciona los nodos keynodes de un arbol""" #es necesario poner los pesos en el gml?

  nodos_key_nodes = np.array([])

  for nodo in range(len(array_pesos)):
    hermanos_nodo = obtener_hermanos_ordenados(nodo, array_aristas, array_pesos)
    if (len(hermanos_nodo) > 1) :
      nodos_key_nodes = np.append(nodos_key_nodes, np.array((hermanos_nodo))[1:])

  nodos_key_nodes = np.append(nodos_key_nodes, len(array_pesos)-1)



  return list(set(nodos_key_nodes))

def obtener_hijo(nodo, array_aristas):
  """Funcion que permite obtener una lista con los hijos del respectivo nodo"""

  hijos_nodo = [i[0] for i in array_aristas if i[1] == nodo]

  return hijos_nodo

def obtener_padre(nodo, array_aristas):
  """Funcion que permite obtener una lista con el padre del respectivo nodo"""

  padre_nodo = [i[1] for i in array_aristas if i[0] == nodo]

  return padre_nodo


def obtener_hoja_profunda_izquierda(nodo, array_aristas, array_pesos):
  """"""


  hijos = encontrar_hijos(nodo, array_aristas)

  indices = np.argsort(array_pesos[hijos][:,-1])
  hijos_ordenados = np.take_along_axis(hijos, indices, axis=0)

  if len(hijos_ordenados) != 0:
    return obtener_hoja_profunda_izquierda(hijos_ordenados[0], array_aristas, array_pesos)
  else:
    return nodo

def izq_der_post_ord(nodo, array_aristas, array_pesos, key_nodes, recorrido=None): #Meter los visitados por aca
  if recorrido is None:
    recorrido = []

  if len(recorrido) != len(array_pesos):
    if (nodo in key_nodes) and (nodo not in recorrido):
      recorrido.append(obtener_hoja_profunda_izquierda(nodo, array_aristas, array_pesos))

      if len(obtener_hermanos_ordenados(recorrido[-1], array_aristas, array_pesos)) < 2:
        nodo = recorrido[-1]
        recorrido += obtener_padre(nodo, array_aristas)
#        print(recorrido)
        izq_der_post_ord(obtener_padre(nodo, array_aristas), array_aristas, array_pesos, key_nodes, recorrido)

      else:
        hermanos_que_no = np.setdiff1d(obtener_hermanos_ordenados(recorrido[-1], array_aristas, array_pesos), recorrido)
#        print(f"los hermanos_que_no {hermanos_que_no}")

        if len(hermanos_que_no) == 0:
          nodo = obtener_padre(nodo, array_aristas)
          recorrido += nodo
          izq_der_post_ord(nodo, array_aristas, array_pesos, key_nodes, recorrido)

        else:
          nodo = hermanos_que_no[0]
#          print(recorrido)
          izq_der_post_ord(nodo, array_aristas, array_pesos, key_nodes, recorrido)

      recorrido += [nodo]

    else:
      if len(obtener_hermanos_ordenados(recorrido[-1], array_aristas, array_aristas)) < 2:
        nodo = recorrido[-1]
        recorrido += obtener_padre(nodo, array_aristas)
#        print(recorrido)
        izq_der_post_ord(obtener_padre(nodo, array_aristas), array_aristas, array_pesos, key_nodes, recorrido)

      else:
        hermanos_que_no = np.setdiff1d(obtener_hermanos_ordenados(recorrido[-1], array_aristas, array_aristas), recorrido)
#        print(f"los hermanos_que_no {hermanos_que_no}")

        if len(hermanos_que_no) == 0:
          nodo = obtener_padre(nodo, array_aristas)
          recorrido += nodo
          izq_der_post_ord(nodo, array_aristas, array_pesos, key_nodes, recorrido)

        else:
          nodo = hermanos_que_no[0]
#          print(recorrido)
          izq_der_post_ord(nodo, array_aristas, array_pesos, key_nodes, recorrido)
#  else:
#    return recorrido
#    print(recorrido)
  return np.array((recorrido[0:len(array_pesos)]))

def subarray(nodo, left_right_array, array_aristas, array_pesos):
  index_hoja = np.where(left_right_array == obtener_hoja_profunda_izquierda(nodo, array_aristas, array_pesos))[0]
  index_key_node = np.where(left_right_array == nodo)[0]

  index_inicio = index_hoja[0]
  index_final = index_key_node[0]

  subarray = left_right_array[index_inicio:index_final+1]

  return subarray

lista_directorio = list(os.listdir("/home/ricardomr/Desktop/gml_calc"))
matriz_comparaciones = np.full((len(lista_directorio) +1,len(lista_directorio) +1), np.nan)
matriz_comparaciones[0,:] = np.arange(len(lista_directorio) +1) -1
matriz_comparaciones[:,0] = np.arange(len(lista_directorio) +1) -1
lista_labels = [archivo[:-4] for archivo in lista_directorio]
print(lista_labels)


comb = combinations_with_replacement(os.listdir("/home/ricardomr/Desktop/gml_calc"), 2)
for mole in list(comb):

  NOM_ARCHIVO1 = r"/home/ricardomr/Desktop/gml_calc/" + mole[0]
  pesos1, aristas1 = obtener_nodos_aristas(NOM_ARCHIVO1)
  NOM_ARCHIVO2 = r"/home/ricardomr/Desktop/gml_calc/" + mole[1]
  pesos2, aristas2 = obtener_nodos_aristas(NOM_ARCHIVO2)

  elementos_max1 = np.amax(pesos1, axis=0)
  elementos_min1 = np.amin(pesos1, axis=0)
  pesos_norm1 = (pesos1 - elementos_min1) / (elementos_max1 - elementos_min1)

  elementos_max2 = np.amax(pesos2, axis=0)
  elementos_min2 = np.amin(pesos2, axis=0)
  pesos_norm2 = (pesos2 - elementos_min2) / (elementos_max2 - elementos_min2)

  key_nodos1 = np.array((nodos_keynodes(aristas1,pesos1)))
  key_nodos2 = np.array((nodos_keynodes(aristas2,pesos2)))
  combinaciones_keynodes = list(product(np.sort(key_nodos1), np.sort(key_nodos2)))
  combinaciones_keynodes = np.array((combinaciones_keynodes))

  distancia_arboles = np.full((len(pesos1)+1,len(pesos2)+1), np.nan)
  distancia_arboles[0,:] = np.arange(len(pesos2)+1) -1
  distancia_arboles[:,0] = np.arange(len(pesos1)+1) -1

  orden1 = izq_der_post_ord(len(pesos1)-1, aristas1, pesos1, key_nodos1)
  orden2 = izq_der_post_ord(len(pesos2)-1, aristas2, pesos2, key_nodos2)

  for combinacion in combinaciones_keynodes[:-1]:

    longitud_filas = len(subarray(combinacion[0], orden1, aristas1, pesos1)) + 2
    longitud_columnas = len(subarray(combinacion[1], orden2, aristas2, pesos2)) +2
    distancia_subarboles = np.zeros((longitud_filas, longitud_columnas))

    identidad_columnas = np.hstack((np.array([-2,-1]),  subarray(combinacion[1], orden2, aristas2, pesos2) ))
    identidad_filas = np.hstack((np.array([-2,-1]),  subarray(combinacion[0], orden1, aristas1, pesos1) ))

    distancia_subarboles[0,:] = identidad_columnas
    distancia_subarboles[:,0] = identidad_filas
    distancia_subarboles[1,1] = 0

    for i in range(len(subarray(combinacion[1], orden2, aristas2, pesos2))):
      distancia_subarboles[1,i+2] = np.linalg.norm(pesos_norm2[int(distancia_subarboles[0,i+2])][:2]) + distancia_subarboles[1,i+1]

    for j in range(len(subarray(combinacion[0], orden1, aristas1, pesos1))):
      distancia_subarboles[j+2,1] = np.linalg.norm(pesos_norm1[int(distancia_subarboles[j+2,0])][:2]) + distancia_subarboles[j+1,1]

#Esto se puede optimizar
    for i in range(2, longitud_filas):
      for j in range(2, longitud_columnas):
        valor_fila_anterior = distancia_subarboles[i-1,j] + np.linalg.norm(pesos_norm1[int(distancia_subarboles[i,0])][:2])   #cambiar valor anterior
        valor_columna_anterior = distancia_subarboles[i,j-1] + np.linalg.norm(pesos_norm2[int(distancia_subarboles[0,j])][:2])  #cambiar valor anterior
        valor_diagonal_anterior = distancia_subarboles[i-1,j-1] + np.linalg.norm((pesos_norm1[int(distancia_subarboles[i,0])][:2]) - (pesos_norm2[int(distancia_subarboles[0,j])][:2]))

        distancia_subarboles[i,j] = min(valor_fila_anterior, valor_columna_anterior, valor_diagonal_anterior)

    for i in range(2, longitud_filas):
      for j in range(2, longitud_columnas):
        if (distancia_subarboles[i-1,0] == -1) and (distancia_subarboles[i-1,0] == -1) and (np.isnan(distancia_arboles[int(distancia_subarboles[i,0]+1), int(distancia_subarboles[0,j]+1)])):
          distancia_arboles[int(distancia_subarboles[i,0]+1), int(distancia_subarboles[0,j]+1)] = distancia_subarboles[i,j] # cambie esto distancia_arboles[int(distancia_subarboles[i,0]+1), int(distancia_subarboles[0,j]+1)] = distancia_subarboles[i,j]

        elif (set(distancia_subarboles[2:i+1,0]).issubset(subarray(distancia_subarboles[i,0], orden1, aristas1, pesos1))) and (set(distancia_subarboles[0,2:j+1]).issubset(subarray(distancia_subarboles[0,j], orden2, aristas2, pesos2))) and (np.isnan(distancia_arboles[int(distancia_subarboles[i,0]+1), int(distancia_subarboles[0,j]+1)])) :
          distancia_arboles[int(distancia_subarboles[i,0]+1), int(distancia_subarboles[0,j]+1)] = distancia_subarboles[i,j]

  longitud_filas = len(subarray(combinaciones_keynodes[-1][0], orden1, aristas1, pesos1)) + 2
  longitud_columnas = len(subarray(combinaciones_keynodes[-1][1], orden2, aristas2, pesos2)) +2

  distancia_subarboles = np.zeros((longitud_filas, longitud_columnas))

  identidad_columnas = np.hstack((np.array([-2,-1]),  orden2 ))
  identidad_filas = np.hstack((np.array([-2,-1]),  orden1 ))

  distancia_subarboles[0,:] = identidad_columnas
  distancia_subarboles[:,0] = identidad_filas
  distancia_subarboles[1,1] = 0

  for i in range(len(orden2)):
    distancia_subarboles[1,i+2] = np.linalg.norm(pesos_norm2[int(distancia_subarboles[0,i+2])][:2]) + distancia_subarboles[1,i+1]

  for j in range(len(orden1)):
    distancia_subarboles[j+2,1] = np.linalg.norm(pesos_norm1[int(distancia_subarboles[j+2,0])][:2]) + distancia_subarboles[j+1,1]

  for i in range(2, longitud_filas):
    for j in range(2, longitud_columnas):
      valor_fila_anterior = distancia_subarboles[i-1,j] + np.linalg.norm(pesos_norm1[int(distancia_subarboles[i,0])][:2])   #cambiar valor anterior
      valor_columna_anterior = distancia_subarboles[i,j-1] + np.linalg.norm(pesos_norm2[int(distancia_subarboles[0,j])][:2])
      valor_diagonal_anterior = distancia_subarboles[i-1,j-1] + np.linalg.norm((pesos_norm1[int(distancia_subarboles[i,0])][:2]) - (pesos_norm2[int(distancia_subarboles[0,j])][:2]))
      distancia_subarboles[i,j] = min(valor_fila_anterior, valor_columna_anterior, valor_diagonal_anterior)
      if np.isnan(distancia_arboles[int(distancia_subarboles[i,0]+1), int(distancia_subarboles[0,j]+1)]):
        distancia_arboles[int(distancia_subarboles[i,0]+1), int(distancia_subarboles[0,j]+1)] = distancia_subarboles[i,j]


  metrica = distancia_subarboles[-1,-1]
  print(f"La distancia de edición entre {mole[0]} y {mole[1]} es {metrica}")
  matriz_comparaciones[lista_directorio.index(mole[0])+1, lista_directorio.index(mole[1])+1] = metrica
  matriz_comparaciones[lista_directorio.index(mole[1])+1, lista_directorio.index(mole[0])+1] = metrica

matrix_sin_norm = pd.DataFrame(matriz_comparaciones[1:,1:])
matrix_sin_norm_data = matrix_sin_norm.to_csv("/home/ricardomr/Desktop/distance_matrices/fly_molc_distance_3_neg.csv", index=False)
max_valor = np.max(matriz_comparaciones[1:,1:])
matriz_comparaciones_norm = matriz_comparaciones[1:,1:] / max_valor
matriz_similitudes = 1 - matriz_comparaciones_norm
matriz_comparaciones_norm_df = pd.DataFrame(matriz_comparaciones_norm)
matrix_comparaciones_norm_data = matriz_comparaciones_norm_df.to_csv("/home/ricardomr/Desktop/distance_matrices/fly_molc_distance_3_neg.csv", index=False)
